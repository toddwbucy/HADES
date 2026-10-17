use super::{Error, MAPPING_VERSION, Plan, Result, hash, key, source::Source};
use base64::Engine;
use regex::Regex;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
struct Entry {
    id: String,
    title: String,
    start: usize,
    header: usize,
    end: usize,
    level: Option<u8>,
    attrs: BTreeMap<String, String>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
fn scalar(value: &Value, name: &str) -> Result<String> {
    match &value[name] {
        Value::String(s) if !s.trim().is_empty() => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        _ => Err(invalid(format!("missing or invalid front matter {name}"))),
    }
}
fn keywords(text: &str) -> Vec<String> {
    text.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}
fn attributes(text: &str, kind: &str) -> Result<BTreeMap<String, String>> {
    let body = text
        .strip_prefix("<!-- ")
        .and_then(|s| s.strip_suffix(" -->"))
        .ok_or_else(|| invalid("malformed node header"))?;
    let mut fields = BTreeMap::new();
    for part in body.split('|') {
        let (name, value) = part
            .trim()
            .split_once(':')
            .ok_or_else(|| invalid("invalid header field"))?;
        if fields
            .insert(name.trim().to_owned(), value.trim().to_owned())
            .is_some()
        {
            return Err(invalid(format!("duplicate header field {name}")));
        }
    }
    if fields.get(kind).is_none_or(|s| s.is_empty()) || !fields.contains_key("keywords") {
        return Err(invalid(format!(
            "{kind} header needs identity and keywords"
        )));
    }
    let allowed = if kind == "node" {
        &["node", "keywords", "xref"][..]
    } else {
        &["illustration", "keywords", "type", "subject", "xref"][..]
    };
    if let Some(field) = fields.keys().find(|name| !allowed.contains(&name.as_str())) {
        return Err(invalid(format!("unsupported {kind} header field {field}")));
    }
    Ok(fields)
}

pub fn map(bytes: &[u8], source: Source, graph: &str) -> Result<Plan> {
    let blocks = &source.blocks;
    let (metadata_text, first_heading) = {
        if blocks.first().is_none_or(|b| b.text != "---") {
            return Err(invalid("missing YAML front matter"));
        }
        let end = blocks
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, b)| b.text == "---")
            .map(|(i, _)| i)
            .ok_or_else(|| invalid("unclosed YAML front matter"))?;
        let first_heading = blocks
            .iter()
            .enumerate()
            .skip(end + 1)
            .find(|(_, b)| b.heading_level.is_some())
            .map(|(i, _)| i)
            .ok_or_else(|| invalid("no heading found"))?;
        let metadata_text = blocks[1..end]
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        (metadata_text, first_heading)
    };
    let mut metadata: Value =
        serde_yaml::from_str(&metadata_text).map_err(|e| invalid(format!("front matter: {e}")))?;
    // Versions are identifiers, not floating point values (0.10 != 0.1).
    if let Some(raw) = metadata_text
        .lines()
        .find_map(|l| l.strip_prefix("version:"))
    {
        let raw = raw.split(" #").next().unwrap_or(raw).trim();
        if metadata["version"].is_number() {
            metadata["version"] = json!(raw);
        }
    }
    let doc_id = scalar(&metadata, "doc_id")?;
    let version = scalar(&metadata, "version")?;
    let title = scalar(&metadata, "title")?;
    let author = scalar(&metadata, "author")?;
    if scalar(&metadata, "kind")? != "graphable-document" {
        return Err(invalid("kind must be graphable-document"));
    }
    for field in [
        "date",
        "source",
        "source_status",
        "budget_tokens",
        "estimated_tokens",
    ] {
        scalar(&metadata, field)?;
    }
    let front_keywords = metadata["keywords"]
        .as_array()
        .ok_or_else(|| invalid("front matter keywords must be a list"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid("keyword must be text"))
        })
        .collect::<Result<Vec<_>>>()?;
    let source_sha256 = hash(bytes);
    let mut entries = Vec::new();
    let mut ids = BTreeSet::new();
    let mut heading_starts = BTreeSet::new();
    for (i, b) in blocks.iter().enumerate().skip(first_heading) {
        if b.literal {
            continue;
        }
        let text = b.text.trim();
        let kind = if text.starts_with("<!-- node:") {
            "node"
        } else if text.starts_with("<!-- illustration:") {
            "illustration"
        } else {
            continue;
        };
        if kind == "illustration" && !b.quoted {
            return Err(invalid("illustration headers must be in a blockquote"));
        }
        if entries.len() >= 10_000 {
            return Err(invalid("source exceeds 10,000 declared nodes"));
        }
        let attrs = attributes(text, kind)?;
        let id = attrs[kind].clone();
        if !ids.insert(id.clone()) {
            return Err(invalid(format!("duplicate logical ID {id}")));
        }
        let previous = (first_heading..i)
            .rev()
            .find(|j| !blocks[*j].text.trim().is_empty())
            .ok_or_else(|| invalid(format!("{id} has no title")))?;
        if kind == "illustration" && !blocks[previous].quoted {
            return Err(invalid(
                "illustration captions must be in the same blockquote",
            ));
        }
        let level = if kind == "node" {
            let level = blocks[previous]
                .heading_level
                .filter(|x| *x <= 3)
                .ok_or_else(|| invalid(format!("{id} must follow an H1, H2, or H3")))?;
            heading_starts.insert(previous);
            Some(level)
        } else {
            None
        };
        if !id.starts_with(&format!("{doc_id}/")) && id != doc_id {
            return Err(invalid(format!(
                "{id} is outside document identity {doc_id}"
            )));
        }
        let title = if level.is_some() {
            blocks[previous]
                .text
                .trim_start()
                .trim_start_matches('#')
                .trim()
                .to_owned()
        } else {
            blocks[previous].text.trim().to_owned()
        };
        entries.push(Entry {
            id,
            title,
            start: previous,
            header: i,
            end: blocks.len() - 1,
            level,
            attrs,
        });
    }
    for (i, b) in blocks.iter().enumerate().skip(first_heading) {
        if b.heading_level.is_some_and(|l| l <= 3) && !heading_starts.contains(&i) {
            return Err(invalid(format!(
                "heading at block {} has no node header",
                b.index
            )));
        }
    }
    if entries
        .first()
        .is_none_or(|e| e.level != Some(1) || e.id != doc_id)
        || entries.iter().filter(|e| e.level == Some(1)).count() != 1
    {
        return Err(invalid(
            "expected exactly one document H1 whose ID equals doc_id",
        ));
    }
    let glossary_index = entries
        .iter()
        .rposition(|e| e.level == Some(2))
        .filter(|i| entries[*i].title.eq_ignore_ascii_case("Glossary"))
        .ok_or_else(|| invalid("last H2 must be Glossary"))?;
    if entries[glossary_index + 1..]
        .iter()
        .any(|e| e.level.is_some())
    {
        return Err(invalid("glossary subsections are unsupported"));
    }
    for i in 0..entries.len() {
        entries[i].end = if entries[i].level.is_some() {
            entries[i + 1..]
                .iter()
                .find(|e| e.level.is_some())
                .map_or(blocks.len() - 1, |e| e.start - 1)
        } else {
            (entries[i].header + 1..blocks.len())
                .take_while(|j| blocks[*j].quoted)
                .last()
                .unwrap_or(entries[i].header)
                .min(entries.get(i + 1).map_or(blocks.len() - 1, |e| e.start - 1))
        };
    }
    let mut builder = Builder {
        plan: Plan {
            graph: graph.into(),
            doc_id: doc_id.clone(),
            version: version.clone(),
            source_sha256,
            nodes: vec![],
            edges: vec![],
            warnings: vec![],
            manifest: Value::Null,
        },
        source: &source,
        registry: BTreeMap::new(),
    };
    let mut stack: Vec<(u8, String)> = Vec::new();
    let mut illustrations = BTreeMap::new();
    let number_regex = Regex::new(r"Illustration (\d+)\.").unwrap();
    for entry in &entries {
        let own: Vec<_> = (entry.header + 1..=entry.end)
            .filter(|j| {
                entry.level.is_none()
                    || !entries
                        .iter()
                        .any(|e| e.level.is_none() && e.start <= *j && *j <= e.end)
            })
            .collect();
        let content = own
            .iter()
            .map(|i| blocks[*i].text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let standing = content
            .split("\n\n")
            .filter(|p| p.trim_start().starts_with("Standing:"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut data = json!({"source_node_id":entry.id,"content":content,"standing_text":standing,"source_declaration":entry.attrs,
            "keywords":keywords(&entry.attrs["keywords"]),"source_location":source.location(entry.start,entry.end),
            "header_location":source.location(entry.header,entry.header),
            "content_blocks":own.iter().map(|i|blocks[*i].index).collect::<Vec<_>>()});
        if let Some(level) = entry.level {
            data["heading_level"] = json!(level);
            if level == 1 {
                data["document_metadata"] = metadata.clone();
                data["source_text"] = json!(source.text);
                data["source_format"] = json!(source.format);
                data["source_blocks"] = serde_json::to_value(blocks)?;
            }
            builder.node(
                if level == 1 { "document" } else { "section" },
                &entry.id,
                &entry.title,
                data,
            )?;
            while stack.last().is_some_and(|(l, _)| *l >= level) {
                stack.pop();
            }
            if level > 1 {
                let (parent_level, parent) = stack
                    .last()
                    .ok_or_else(|| invalid("heading has no parent"))?;
                if *parent_level + 1 != level {
                    return Err(invalid("heading hierarchy skips a level"));
                }
                builder.edge(
                    parent,
                    &entry.id,
                    "contains",
                    entry.header,
                    "declared_structure",
                )?;
            }
            stack.push((level, entry.id.clone()));
        } else {
            let number: u64 = number_regex
                .captures(&entry.title)
                .and_then(|c| c[1].parse().ok())
                .ok_or_else(|| invalid(format!("illustration {} has no number", entry.id)))?;
            if number != illustrations.len() as u64 + 1 {
                return Err(invalid("illustration numbers must be consecutive from 1"));
            }
            let medium = entry
                .attrs
                .get("type")
                .filter(|s| !s.is_empty())
                .ok_or_else(|| invalid("illustration has no type"))?;
            data["number"] = json!(number);
            data["medium"] = json!(medium);
            data["non_normative"] = json!(true);
            data["media_status"] = json!(if medium == "text" {
                "inline"
            } else {
                "source_reference_only"
            });
            builder.node("illustration", &entry.id, &entry.title, data)?;
            illustrations.insert(number, entry.id.clone());
        }
    }
    // Definitions remain scoped to this source revision. A matching spelling in
    // another document never establishes equivalence automatically.
    let glossary = &entries[glossary_index];
    let definition_regex = Regex::new(r"^(?:-\s+)?(?:\*\*([^*]+)\*\*|([^:]+)):\s+(.+)$").unwrap();
    let alias_regex = Regex::new(r" \([A-Za-z][A-Za-z0-9_]*\)$").unwrap();
    let mut terms = BTreeMap::new();
    let mut i = glossary.header + 1;
    while i < blocks.len() {
        let line = blocks[i].text.trim();
        if line.is_empty() || line.starts_with("<!-- ingestion:") {
            i += 1;
            continue;
        }
        let captures = definition_regex.captures(line).ok_or_else(|| {
            invalid(format!(
                "invalid glossary entry at block {}",
                blocks[i].index
            ))
        })?;
        let label = captures
            .get(1)
            .or_else(|| captures.get(2))
            .unwrap()
            .as_str();
        let term = alias_regex.replace(label, "").to_string();
        let start = i;
        let mut definition = captures[3].to_owned();
        {
            while i + 1 < blocks.len()
                && !blocks[i + 1].text.trim().is_empty()
                && !blocks[i + 1].text.trim_start().starts_with("- ")
                // Indented lines continue the current definition, including
                // "Rejected synonyms: ...". An unindented definition starts
                // a new entry even when it has no list marker.
                && (blocks[i + 1].text.starts_with(char::is_whitespace)
                    || !definition_regex.is_match(&blocks[i + 1].text))
            {
                i += 1;
                definition.push('\n');
                definition.push_str(&blocks[i].text);
            }
        }
        let id = format!("{doc_id}/concept/{term}");
        if terms.insert(term.clone(), id.clone()).is_some() {
            return Err(invalid(format!("duplicate glossary term {term}")));
        }
        builder.node(
            "concept",
            &id,
            &term,
            json!({"term":term,"source_term_label":label,"definition":definition,
            "definition_status":"source_defined","definition_scope":format!("{doc_id}@{version}"),
            "source_location":source.location(start,i)}),
        )?;
        builder.edge_span(
            &glossary.id,
            &id,
            "defines",
            start..=i,
            "declared_definition",
        )?;
        i += 1;
    }
    for entry in &entries {
        let mut words = keywords(&entry.attrs["keywords"]);
        if entry.level == Some(1) {
            words.extend(front_keywords.clone());
        }
        for term in words {
            let id = if let Some(id) = terms.get(&term) {
                id.clone()
            } else {
                let id = format!("{doc_id}/keyword/{term}");
                builder.node(
                    "unresolved_keyword",
                    &id,
                    &term,
                    json!({"term":term,"definition_status":"missing_from_glossary",
                    "source_location":source.location(entry.header,entry.header)}),
                )?;
                terms.insert(term.clone(), id.clone());
                builder
                    .plan
                    .warnings
                    .push(format!("missing glossary definition: {term}"));
                id
            };
            let evidence =
                if entry.level == Some(1) && !keywords(&entry.attrs["keywords"]).contains(&term) {
                    0..=first_heading - 1
                } else {
                    entry.header..=entry.header
                };
            builder.edge_span(&entry.id, &id, "uses_keyword", evidence, "declared_keyword")?;
        }
        if let Some(subject) = entry.attrs.get("subject") {
            let matches: Vec<_> = entries
                .iter()
                .filter(|e| {
                    e.level.is_some()
                        && (e.id == *subject || e.id.ends_with(&format!("/{subject}")))
                })
                .collect();
            if matches.len() != 1 {
                return Err(invalid(format!(
                    "illustration subject {subject} is missing or ambiguous"
                )));
            }
            builder.edge(
                &entry.id,
                &matches[0].id,
                "illustrates",
                entry.header,
                "declared_structure",
            )?;
        } else if entry.level.is_none() {
            return Err(invalid("illustration has no subject"));
        }
        if let Some(target) = entry.attrs.get("xref") {
            let target_id = if builder.registry.contains_key(target) {
                target.clone()
            } else {
                let id = format!("{doc_id}/reference/{target}");
                if !builder.registry.contains_key(&id) {
                    builder.node(
                        "unresolved_reference",
                        &id,
                        target,
                        json!({"target_source_node_id":target,"resolution_status":"unresolved",
                        "source_location":source.location(entry.header,entry.header)}),
                    )?;
                    builder
                        .plan
                        .warnings
                        .push(format!("unresolved declared reference: {target}"));
                }
                id
            };
            builder.edge(
                &entry.id,
                &target_id,
                "references",
                entry.header,
                "declared_reference",
            )?;
        }
    }
    let cite_regex = Regex::new(r"\bIllustration (\d+)\b").unwrap();
    for entry in &entries {
        for (i, block) in blocks
            .iter()
            .enumerate()
            .take(entry.end + 1)
            .skip(entry.header + 1)
        {
            if block.literal
                || entries
                    .iter()
                    .any(|e| e.id != entry.id && e.start <= i && i <= e.end && e.level.is_none())
            {
                continue;
            }
            for cap in cite_regex.captures_iter(&block.text) {
                let n = cap[1].parse::<u64>().unwrap_or(0);
                if let Some(target) = illustrations.get(&n) {
                    builder.edge(
                        &entry.id,
                        target,
                        "cites_illustration",
                        i,
                        "declared_reference",
                    )?;
                } else {
                    builder
                        .plan
                        .warnings
                        .push(format!("unknown Illustration {n} in {}", entry.id));
                }
            }
        }
    }
    builder.plan.warnings.sort();
    builder.plan.warnings.dedup();
    let manifest_key = key("revision", &[graph, &doc_id, &version]);
    builder.plan.manifest = json!({"_key":manifest_key,"doc_id":doc_id,"source_version":version,"title":title,
        "source_sha256":builder.plan.source_sha256,"mapping_version":MAPPING_VERSION,
        "source_format":source.format,"source_base64":base64::engine::general_purpose::STANDARD.encode(bytes),
        "source_author":author,"mapping_origin":"hades-core mechanical importer",
        "nodes":builder.plan.nodes.len(),"edges":builder.plan.edges.len(),
        "payload_sha256":super::payload_hash(&builder.plan.nodes,&builder.plan.edges)});
    let size = serde_json::to_vec(&(
        &builder.plan.nodes,
        &builder.plan.edges,
        &builder.plan.manifest,
    ))?
    .len();
    if size > 12 * 1024 * 1024 {
        return Err(invalid("mapped payload exceeds 12 MiB transaction budget"));
    }
    Ok(builder.plan)
}

struct Builder<'a> {
    plan: Plan,
    source: &'a Source,
    registry: BTreeMap<String, String>,
}
impl Builder<'_> {
    fn node(&mut self, kind: &str, id: &str, title: &str, data: Value) -> Result<()> {
        if self.plan.nodes.len() >= 20_000 {
            return Err(invalid("mapping exceeds 20,000 nodes"));
        }
        if self.registry.contains_key(id) {
            return Err(invalid(format!(
                "derived identity collides with declared identity {id}"
            )));
        }
        let key = key(
            kind,
            &[
                &self.plan.doc_id,
                &self.plan.version,
                &self.plan.source_sha256,
                id,
            ],
        );
        let mut row = json!({"_key":key,"type":kind,"logical_id":id,"label":title,"title":title,
            "doc_id":self.plan.doc_id,"source_version":self.plan.version,"source_sha256":self.plan.source_sha256,
            "mapping_version":MAPPING_VERSION,"_hades":{"basis":"authored"},
            "attribution":{"mapper":"hades-core","role":"mechanical_source_mapping"}});
        row.as_object_mut()
            .unwrap()
            .extend(data.as_object().unwrap().clone());
        self.registry.insert(id.into(), key);
        self.plan.nodes.push(row);
        Ok(())
    }
    fn edge(
        &mut self,
        from: &str,
        to: &str,
        relation: &str,
        block: usize,
        origin: &str,
    ) -> Result<()> {
        self.edge_span(from, to, relation, block..=block, origin)
    }
    fn edge_span(
        &mut self,
        from: &str,
        to: &str,
        relation: &str,
        span: std::ops::RangeInclusive<usize>,
        origin: &str,
    ) -> Result<()> {
        let (start, end) = (*span.start(), *span.end());
        let a = self
            .registry
            .get(from)
            .ok_or_else(|| invalid(format!("missing endpoint {from}")))?;
        let b = self
            .registry
            .get(to)
            .ok_or_else(|| invalid(format!("missing endpoint {to}")))?;
        let key = key("edge", &[a, relation, b]);
        let evidence = json!({"location":self.source.location(start,end),"quote":&self.source.text[self.source.blocks[start].start_byte..self.source.blocks[end].end_byte]});
        if let Some(edge) = self.plan.edges.iter_mut().find(|e| e["_key"] == key) {
            let passages = edge["evidence"].as_array_mut().unwrap();
            if !passages.contains(&evidence) {
                passages.push(evidence);
            }
            return Ok(());
        }
        if self.plan.edges.len() >= 50_000 {
            return Err(invalid("mapping exceeds 50,000 edges"));
        }
        self.plan.edges.push(json!({"_key":key,"_from":format!("{}_nodes/{a}",self.plan.graph),
            "_to":format!("{}_nodes/{b}",self.plan.graph),"type":relation,"relation":relation,"label":relation,
            "doc_id":self.plan.doc_id,"source_version":self.plan.version,"source_sha256":self.plan.source_sha256,
            "mapping_version":MAPPING_VERSION,"_hades":{"basis":"authored"},"assertion_origin":origin,
            "evidence":[evidence]}));
        Ok(())
    }
}
