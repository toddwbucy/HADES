//! Human-guided reading: source archives, revisioned drafts, and atomic acceptance.
mod model;
use crate::{
    config::HadesConfig,
    db::{ArangoClient, ArangoError, ArangoPool, crud, transaction},
    dispatch::HandlerError,
    persephone::extraction::{ExtractOptions, ExtractionClient},
};
use base64::Engine;
pub use model::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
const SESSIONS: &str = "hades_reading_sessions";
const EVENTS: &str = "hades_reading_events";
const SOURCES: &str = "hades_reading_sources";
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
// ArangoDB can return integral doubles as JSON integers. Normalize that spelling
// before hashing so page-edge figure coordinates do not invalidate an export.
fn record_hash(value: &Value) -> String {
    fn normalize(value: &mut Value) {
        match value {
            Value::Array(items) => items.iter_mut().for_each(normalize),
            Value::Object(fields) => fields.values_mut().for_each(normalize),
            Value::Number(number) if number.is_f64() => {
                if let Some(n) = number.as_f64()
                    && n.fract() == 0.0
                    && n.abs() < 9_007_199_254_740_992.0
                {
                    *value = json!(n as i64);
                }
            }
            _ => {}
        }
    }
    let mut normalized = value.clone();
    normalize(&mut normalized);
    hash(&crate::canonical_json::to_vec(&normalized))
}
fn invalid(message: impl Into<String>) -> HandlerError {
    HandlerError::InvalidParameter {
        name: "reading".into(),
        reason: message.into(),
    }
}
fn db_error(source: ArangoError) -> HandlerError {
    HandlerError::Query {
        context: source.to_string(),
        source,
    }
}
fn conflict(message: impl Into<String>) -> ArangoError {
    ArangoError::Api {
        status: 409,
        error_num: 1200,
        message: message.into(),
        body: Value::Null,
    }
}
fn hash_valid(s: &str) -> Result<(), HandlerError> {
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        Err(invalid("invalid source SHA-256"))
    } else {
        Ok(())
    }
}
async fn ensure(pool: &ArangoPool) -> Result<(), HandlerError> {
    for name in [SESSIONS, EVENTS, SOURCES] {
        match crud::create_collection(pool, name, Some(2)).await {
            Ok(_)
            | Err(ArangoError::Api {
                error_num: 1207, ..
            }) => {}
            Err(e) => return Err(db_error(e)),
        }
        let properties = pool
            .writer()
            .get(&format!("collection/{name}/properties"))
            .await
            .map_err(db_error)?;
        if properties["type"] != 2 {
            return Err(invalid("reading store must use document collections"));
        }
    }
    Ok(())
}
async fn optional(client: &ArangoClient, path: &str) -> Result<Option<Value>, ArangoError> {
    match client.get(path).await {
        Ok(v) => Ok(Some(v)),
        Err(e) if e.is_not_found() => Ok(None),
        Err(e) => Err(e),
    }
}
async fn rows(client: &ArangoClient, query: &str, bind: Value) -> Result<Vec<Value>, ArangoError> {
    let v=client.post("cursor",&json!({"query":query,"bindVars":bind,"batchSize":2001,"ttl":10,"options":{"maxRuntime":10}})).await?;
    Ok(crate::db::query::completed_rows(&v)?.to_vec())
}

pub async fn capabilities(config: &HadesConfig) -> Result<Value, HandlerError> {
    let c = ExtractionClient::connect_at(&config.extraction.service.socket)
        .await
        .map_err(|e| HandlerError::ServiceError(e.to_string()))?;
    let info = c
        .capabilities()
        .await
        .map_err(|e| HandlerError::ServiceError(e.to_string()))?;
    Ok(
        json!({"format":FORMAT,"extensions":info.reading_extensions,"features":info.features,"max_source_bytes":8*1024*1024,"max_session_events":2000,"palette":{"version":"reading/v1","labels":["node","is","is_not","question"]}}),
    )
}

/// Archive first. Extraction failure leaves the original available for retry/recovery.
pub async fn upload(pool: &ArangoPool, p: SourceRequest) -> Result<Value, HandlerError> {
    if p.filename.is_empty() || p.filename.len() > 255 || p.filename.contains(['/', '\\', '\0']) {
        return Err(invalid("supply a filename, not a path"));
    }
    if p.content_base64.len() > 12 * 1024 * 1024 {
        return Err(invalid("source exceeds 8 MiB upload limit"));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&p.content_base64)
        .map_err(|_| invalid("invalid base64"))?;
    if bytes.is_empty() || bytes.len() > 8 * 1024 * 1024 {
        return Err(invalid("source must contain 1 byte to 8 MiB"));
    }
    let id = hash(&bytes);
    ensure(pool).await?;
    let result=transaction::run(pool,vec![SOURCES.into()],move|client|async move{
        if let Some(existing)=optional(&client,&format!("document/{SOURCES}/{id}")).await? {
            if existing["content_base64"]!=p.content_base64{return Err(conflict("source hash collision"))}
            return Ok(json!({"source_hash":id,"status":existing["status"],"unchanged":true}));
        }
        client.post(&format!("document/{SOURCES}"),&json!({"_key":id,"filename":p.filename,"content_base64":p.content_base64,"status":"archived","bytes":bytes.len(),"archived_at":chrono::Utc::now().to_rfc3339()})).await?;
        Ok(json!({"source_hash":id,"status":"archived","embedded":false,"searchable":false}))
    }).await.map_err(db_error)?;
    Ok(result)
}

pub async fn extract(
    pool: &ArangoPool,
    config: &HadesConfig,
    id: String,
) -> Result<Value, HandlerError> {
    hash_valid(&id)?;
    let source = pool
        .writer()
        .get(&format!("document/{SOURCES}/{id}"))
        .await
        .map_err(db_error)?;
    if source["status"] == "ready" {
        return Ok(json!({"source_hash":id,"status":"ready","unchanged":true}));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(
            source["content_base64"]
                .as_str()
                .ok_or_else(|| invalid("source bytes missing"))?,
        )
        .map_err(|_| invalid("source archive corrupted"))?;
    if hash(&bytes) != id {
        return Err(invalid("source archive hash mismatch"));
    }
    let filename = source["filename"]
        .as_str()
        .ok_or_else(|| invalid("source filename missing"))?;
    let result = async {
        let c = ExtractionClient::connect_at(&config.extraction.service.socket).await?;
        c.extract_bytes(
            filename,
            bytes,
            ExtractOptions {
                include_reading_document: true,
                ..ExtractOptions::all()
            },
        )
        .await
    }
    .await;
    let patch = match result {
        Ok(r) if r.reading_document_json.trim().is_empty() => {
            json!({"status":"failed","error":"extractor returned no structured reading surface for this source"})
        }
        Ok(r) if r.reading_document_json.len() > 4 * 1024 * 1024 => {
            json!({"status":"failed","error":"structured document exceeds 4 MiB limit; original is archived"})
        }
        Ok(r) => match serde_json::from_str::<Value>(&r.reading_document_json) {
            Ok(structured)
                if structured["format"] == "docling-blocks-v1"
                    && structured["blocks"].is_array() =>
            {
                json!({"status":"ready","structured":structured,"extractor_metadata":r.metadata,"representation_hash":hash(r.reading_document_json.as_bytes()),"error":null,"embedded":false,"searchable":false})
            }
            _ => {
                json!({"status":"failed","error":"extractor returned an invalid structured reading document"})
            }
        },
        Err(e) => json!({"status":"failed","error":e.to_string()}),
    };
    // Only publish the first successful extraction. Anchors cannot drift on retry.
    transaction::run(pool,vec![SOURCES.into()],move|client|async move{
        let mut current=client.get(&format!("document/{SOURCES}/{id}")).await?;
        if current["status"]!="ready" {
            current.as_object_mut().unwrap().extend(patch.as_object().unwrap().clone());
            client.put(&format!("document/{SOURCES}/{id}"),&current).await?;
        }
        Ok(json!({"source_hash":id,"status":current["status"],"error":current["error"],"embedded":false,"searchable":false}))
    }).await.map_err(db_error)
}

pub async fn open(pool: &ArangoPool, p: OpenRequest) -> Result<Value, HandlerError> {
    hash_valid(&p.source_hash)?;
    let mut s = Session::new(&p).map_err(invalid)?;
    if s.graph.contains('-') {
        return Err(invalid("session IDs must use underscores, not hyphens"));
    }
    ensure(pool).await?;
    let src = pool
        .writer()
        .get(&format!("document/{SOURCES}/{}", p.source_hash))
        .await
        .map_err(db_error)?;
    if src["status"] != "ready" {
        return Err(invalid(
            "extract the archived source before opening a reading session",
        ));
    }
    // Freeze allowed endpoint collections at session creation. Existing graph content is never rewritten.
    if let Some(existing) = optional(pool.writer(), &format!("document/{SESSIONS}/{}", s.id))
        .await
        .map_err(db_error)?
    {
        if existing["state"]["source_hash"] != p.source_hash
            || existing["state"]["title"] != p.title
        {
            return Err(invalid(
                "session identity already belongs to different content",
            ));
        }
        return Ok(existing["state"].clone());
    }
    let graphs = crate::dispatch::db_graph_list(pool).await?;
    let mut collections = std::collections::BTreeSet::from([format!("{}_nodes", s.graph)]);
    if let Some(gs) = graphs["graphs"].as_array() {
        for g in gs {
            if let Some(defs) = g["edge_definitions"].as_array() {
                for def in defs {
                    for end in ["from", "to"] {
                        if let Some(names) = def[end].as_array() {
                            for name in names.iter().filter_map(Value::as_str) {
                                if !name.starts_with('_') && !name.starts_with("hades_reading_") {
                                    collections.insert(name.to_owned());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if collections.len() > 200 {
        return Err(invalid(
            "too many graph endpoint collections for a reading workspace",
        ));
    }
    s.graph_collections = collections.into_iter().collect();
    for (name, kind) in [
        (format!("{}_nodes", s.graph), 2),
        (format!("{}_edges", s.graph), 3),
    ] {
        match crud::create_collection(pool, &name, Some(kind)).await {
            Ok(_)
            | Err(ArangoError::Api {
                error_num: 1207, ..
            }) => {}
            Err(e) => return Err(db_error(e)),
        }
        let count = pool
            .writer()
            .get(&format!("collection/{name}/count"))
            .await
            .map_err(db_error)?;
        if count["count"] != 0 {
            return Err(invalid(
                "reading graph has content but no session record; inspect before recovery",
            ));
        }
    }
    let definition = json!({"name":s.graph,"edgeDefinitions":[{"collection":format!("{}_edges",s.graph),"from":s.graph_collections,"to":s.graph_collections}]});
    match pool.writer().post("gharial", &definition).await {
        Ok(_) => {}
        Err(ArangoError::Api {
            error_num: 1925, ..
        }) => {
            let existing = pool
                .writer()
                .get(&format!("gharial/{}", s.graph))
                .await
                .map_err(db_error)?;
            if existing["graph"]["edgeDefinitions"] != definition["edgeDefinitions"] {
                return Err(invalid("existing reading graph definition differs"));
            }
        }
        Err(e) => return Err(db_error(e)),
    }
    transaction::run(pool,vec![SESSIONS.into(),EVENTS.into()],move|client|async move{
        if let Some(existing)=optional(&client,&format!("document/{SESSIONS}/{}",s.id)).await?{
            if existing["state"]["source_hash"]!=p.source_hash || existing["state"]["title"]!=p.title{return Err(conflict("session identity already belongs to different content"))}
            return Ok(existing["state"].clone());
        }
        client.post(&format!("document/{SESSIONS}"),&json!({"_key":s.id,"state":s,"opened_by":p.actor,"opened_at":chrono::Utc::now().to_rfc3339(),"format":FORMAT})).await?;
        Ok(serde_json::to_value(s).unwrap())
    }).await.map_err(db_error)
}

pub async fn mutate(pool: &ArangoPool, p: Mutation, agent: bool) -> Result<Value, HandlerError> {
    token(&p.session).map_err(invalid)?;
    token(&p.request_id).map_err(invalid)?;
    let initial = pool
        .writer()
        .get(&format!("document/{SESSIONS}/{}", p.session))
        .await
        .map_err(db_error)?;
    let graph = initial["state"]["graph"]
        .as_str()
        .ok_or_else(|| invalid("invalid session graph"))?;
    let (nodes, edges) = crate::dispatch::kg_names(graph)?;
    let expected = json!([{"collection":edges,"from":initial["state"]["graph_collections"],"to":initial["state"]["graph_collections"]}]);
    let actual = pool
        .writer()
        .get(&format!("gharial/{graph}"))
        .await
        .map_err(db_error)?;
    if actual["graph"]["edgeDefinitions"] != expected {
        return Err(invalid(
            "reading graph layout changed; inspect before writing",
        ));
    }
    let digest = record_hash(&json!({"request":p,"agent":agent}));
    let read_collections = initial["state"]["graph_collections"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .filter(|c| *c != nodes)
        .map(str::to_owned)
        .collect();
    transaction::run_with_reads(pool,vec![SESSIONS.into(),EVENTS.into(),SOURCES.into(),nodes.clone(),edges.clone()],read_collections,move|client|async move{
        let key=format!("{}_{}",p.session,p.request_id);
        if let Some(event)=optional(&client,&format!("document/{EVENTS}/{key}")).await?{
            if event["request_hash"]!=digest {return Err(conflict("request ID reused with a different payload"))}
            return Ok(event["receipt"].clone());
        }
        let mut row=client.get(&format!("document/{SESSIONS}/{}",p.session)).await?;
        let mut s:Session=serde_json::from_value(row["state"].clone()).map_err(|e|conflict(e.to_string()))?;
        if let Action::Highlight{anchor,..}=&p.action && anchor.representation=="docling-blocks-v1" {
                let src=client.get(&format!("document/{SOURCES}/{}",s.source_hash)).await?;
                let block=src["structured"]["blocks"].as_array().and_then(|blocks|blocks.iter().find(|b|b["id"].as_str()==anchor.block.as_deref())).ok_or_else(||conflict("anchor block missing"))?;
                let id=anchor.block.as_deref().unwrap();
                if s.passage_revision(id)!=anchor.passage_revision {return Err(conflict("passage changed; reload before highlighting"));}
                let body=s.passage_text(id,anchor.passage_revision,block["text"].as_str().unwrap_or("")).map_err(conflict)?;
                let needle=format!("{}{}{}",anchor.prefix,anchor.exact,anchor.suffix);
                if needle.is_empty() || body.match_indices(&needle).count()!=1{return Err(conflict("anchor is missing or ambiguous in the archived representation"))}
        }
        if let Action::CorrectPassage{block,passage_revision,text,restore_revision}=&p.action {
            let src=client.get(&format!("document/{SOURCES}/{}",s.source_hash)).await?;
            let item=src["structured"]["blocks"].as_array().and_then(|items|items.iter().find(|v|v["id"].as_str()==Some(block.as_str()))).ok_or_else(||conflict("source passage missing"))?;
            if !matches!(item["kind"].as_str(),Some("text"|"title"|"section_header"|"caption"|"list_item"|"footnote"|"page_header"|"page_footer"|"reference")) {return Err(conflict("only plain-text passages can be corrected; table, equation, and figure editing is not supported"));}
            let original=item["text"].as_str().unwrap_or("");
            let previous=s.passage_text(block,*passage_revision,original).map_err(conflict)?;
            if text==previous {return Err(conflict("correction does not change the passage"));}
            if let Some(revision)=restore_revision && s.passage_text(block,*revision,original).map_err(conflict)?!=text {return Err(conflict("restore text does not match the requested historical revision"));}
        }
        let writes=s.apply(&p,agent).map_err(conflict)?;
        if let Action::Accept{id,..}=&p.action && s.proposals[id].draft.from.is_none() {s.outline_order.push(id.clone());}
        for mut value in writes {
            let is_edge=value.get("_from").is_some();
            if is_edge {for field in ["_from","_to"]{let endpoint=value[field].as_str().unwrap();client.get(&format!("document/{endpoint}")).await?;}}
            value["_hades"]=json!({"basis":"authored","source":"guided-reading","session":s.id,"reviewer":p.actor,"request_id":p.request_id});
            let collection=if is_edge{&edges}else{&nodes};
            client.post(&format!("document/{collection}"),&value).await?;
        }
        let ids=match &p.action{Action::Accept{id,..}=>s.proposals[id].created_ids.clone(),_=>vec![]};
        let receipt=json!({"format":FORMAT,"session":s.id,"revision":s.revision,"created_ids":ids,"embedded":false,"searchable":false});
        let mut event=json!({"_key":key,"session":s.id,"sequence":s.revision,"actor":p.actor,"channel":if agent{"agent"}else{"human"},"request":p,"request_hash":digest,"at":chrono::Utc::now().to_rfc3339(),"previous_hash":s.last_event_hash,"receipt":receipt});
        let event_hash=record_hash(&event);
        event["event_hash"]=json!(event_hash); s.last_event_hash=event_hash;
        s.history_bytes+=serde_json::to_vec(&event).unwrap().len()+256;
        if s.history_bytes>8*1024*1024 {return Err(conflict("reading history exceeds 8 MiB; export before continuing in another session"));}
        let state=serde_json::to_value(&s).unwrap();
        if serde_json::to_vec(&state).unwrap().len()>2*1024*1024{return Err(conflict("reading session exceeds 2 MiB; export before continuing in another session"))}
        row["state"]=state;
        client.post(&format!("document/{EVENTS}"),&event).await?;
        client.put(&format!("document/{SESSIONS}/{}",s.id),&row).await?;
        Ok(receipt)
    }).await.map_err(db_error)
}

pub async fn read(pool: &ArangoPool, p: ReadRequest) -> Result<Value, HandlerError> {
    if let Some(id) = p.source_hash {
        hash_valid(&id)?;
        let mut s = pool
            .writer()
            .get(&format!("document/{SOURCES}/{id}"))
            .await
            .map_err(db_error)?;
        if p.view != "original" {
            s.as_object_mut().unwrap().remove("content_base64");
        }
        return Ok(s);
    }
    if p.view == "sources" {
        let result=rows(pool.writer(),"FOR s IN hades_reading_sources SORT s.archived_at DESC LIMIT 101 RETURN {source_hash:s._key,filename:s.filename,status:s.status,error:s.error,bytes:s.bytes}",json!({})).await;
        return match result {
            Ok(v) => {
                Ok(json!({"sources":v.iter().take(100).collect::<Vec<_>>(),"partial":v.len()>100}))
            }
            Err(e) if e.is_not_found() => Ok(json!({"sources":[]})),
            Err(e) => Err(db_error(e)),
        };
    }
    let Some(id) = p.session else {
        let result=rows(pool.writer(),"FOR s IN hades_reading_sessions SORT s.opened_at DESC LIMIT 101 RETURN {id:s._key,title:s.state.title,source_hash:s.state.source_hash,stage:s.state.stage,revision:s.state.revision}",json!({})).await;
        return match result {
            Ok(v) => Ok(
                json!({"sessions":v.iter().take(100).collect::<Vec<_>>(),"partial":v.len()>100,"database":pool.database()}),
            ),
            Err(e) if e.is_not_found() => Ok(json!({"sessions":[],"database":pool.database()})),
            Err(e) => Err(db_error(e)),
        };
    };
    token(&id).map_err(invalid)?;
    let row = pool
        .writer()
        .get(&format!("document/{SESSIONS}/{id}"))
        .await
        .map_err(db_error)?;
    let state = row["state"].clone();
    if p.view == "context" {
        let s: Session = serde_json::from_value(state).map_err(|e| invalid(e.to_string()))?;
        let refs:Vec<_>=s.focus.iter().map(|id|{
            if let Some(h)=s.highlights.get(id){let mut h=h.clone();h.anchor.exact=h.anchor.exact.chars().take(1200).collect();json!({"id":id,"highlight":h,"current_passage_revision":h.anchor.block.as_deref().map(|b|s.passage_revision(b)),"passage_changed":h.anchor.block.as_deref().is_some_and(|b|s.passage_revision(b)!=h.anchor.passage_revision),"truncated":s.highlights[id].anchor.exact.chars().count()>1200})}
            else if let Some(pr)=s.proposals.get(id){json!({"id":id,"title":pr.draft.title,"status":pr.status,"revision":pr.revision})}
            else{json!({"id":id})}
        }).collect();
        let mut packet = json!({"format":FORMAT,"database":pool.database(),"session":s.id,"revision":s.revision,"stage":s.stage,"graph":s.graph,"focus_actor":s.focus_actor,"references":refs,"pins":s.pins,"decisions":s.proposals.values().map(|p|json!({"id":p.id,"revision":p.revision,"status":p.status,"created_ids":p.created_ids})).collect::<Vec<_>>(),"truncated":false});
        if serde_json::to_vec(&packet).unwrap().len() > 12000 {
            packet["references"] = json!(
                s.focus
                    .iter()
                    .map(|id| json!({"id":id}))
                    .collect::<Vec<_>>()
            );
            packet["pins"] = json!({});
            packet["decisions"] = json!([]);
            packet["truncated"] = json!(true);
            packet["expand"] = json!(
                "Read the session state explicitly for full evidence, pin sets, and decisions."
            );
        }
        return Ok(packet);
    }
    if p.view == "export" {
        let events=rows(pool.writer(),"FOR e IN hades_reading_events FILTER e.session == @session SORT e.sequence LIMIT 2001 RETURN e",json!({"session":id})).await.map_err(db_error)?;
        let mut previous = String::new();
        let mut chain_valid = events.len() as u64 == state["revision"].as_u64().unwrap_or(u64::MAX);
        for (index, event) in events.iter().enumerate() {
            let mut unsigned = event.clone();
            let object = unsigned.as_object_mut().unwrap();
            object.remove("_id");
            object.remove("_rev");
            object.remove("event_hash");
            let computed = record_hash(&unsigned);
            chain_valid &= event["previous_hash"] == previous
                && event["sequence"] == (index + 1)
                && event["event_hash"] == computed;
            previous = event["event_hash"].as_str().unwrap_or("").to_owned();
        }
        chain_valid &= state["last_event_hash"] == previous;
        let session: Session =
            serde_json::from_value(state.clone()).map_err(|e| invalid(e.to_string()))?;
        let reference_warnings = session.reference_warnings();
        return Ok(
            json!({"chain_valid":chain_valid,"reference_warnings":reference_warnings,"format":FORMAT,"session":state,"opened_by":row["opened_by"],"opened_at":row["opened_at"],"events":events,"integrity":"Hash chain detects changes relative to a trusted saved head. Database administrators can rewrite history.","outline":outline(&state)}),
        );
    }
    let reference_warnings: Session =
        serde_json::from_value(state.clone()).map_err(|e| invalid(e.to_string()))?;
    let reference_warnings = reference_warnings.reference_warnings();
    let keys = state["proposals"]
        .as_object()
        .unwrap()
        .values()
        .filter(|p| p["status"] == "accepted" && p["draft"]["from"].is_null())
        .map(|p| p["id"].clone())
        .collect::<Vec<_>>();
    let graph = state["graph"].as_str().unwrap();
    let connectivity=rows(pool.writer(),"FOR key IN @keys LET n=DOCUMENT(@nodes,CONCAT('note_',key)) LET e=DOCUMENT(@edges,CONCAT('basis_',key)) LET root=e==null?null:DOCUMENT(e._to) RETURN {proposal:key,connected:n!=null AND e!=null AND e._from==n._id AND root!=null}",json!({"keys":keys,"nodes":format!("{graph}_nodes"),"edges":format!("{graph}_edges")})).await.map_err(db_error)?;
    Ok(
        json!({"format":FORMAT,"database":pool.database(),"state":state,"connectivity":connectivity,"reference_warnings":reference_warnings,"outline":outline(&state)}),
    )
}
fn outline(state: &Value) -> String {
    let mut out = format!(
        "# {}\n\nIS: {}\n\nIS NOT: {}\n",
        state["title"].as_str().unwrap_or("Reading"),
        state["bound"]["is"].as_str().unwrap_or("Unresolved"),
        state["bound"]["is_not"].as_str().unwrap_or("Unresolved")
    );
    if let (Some(ps), Some(order)) = (
        state["proposals"].as_object(),
        state["outline_order"].as_array(),
    ) {
        for id in order.iter().filter_map(Value::as_str) {
            let Some(p) = ps.get(id) else { continue };
            if p["status"] != "accepted" {
                continue;
            }
            out.push_str(&format!(
                "\n- {} [{}]\n",
                p["draft"]["title"].as_str().unwrap_or("Note"),
                p["created_ids"]
                    .as_array()
                    .map(|a| a
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(", "))
                    .unwrap_or_default()
            ));
        }
    }
    out
}
