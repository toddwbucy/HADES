//! Re-ingestion preserves references without guessing overload identity or editing foreign edges.
use super::*;
use hades_core::db::crud;

#[path = "symbols/review.rs"]
mod review;

pub(super) fn shifted_input(path: &str, body: &str) -> CodeInput {
    let mut next = input(path, body);
    next.text.insert_str(0, "// inserted above every symbol\n");
    next.analysis = code::analyze(&next.text, "fixture.rs").unwrap();
    next
}

fn source(text: &str) -> CodeInput {
    CodeInput {
        namespace: "fixture/project".into(),
        path: "one.rs".into(),
        text: text.into(),
        analysis: code::analyze(text, "fixture.rs").unwrap(),
        relationships: vec![],
    }
}

fn symbol<'a>(stored: &'a Value, name: &str) -> &'a Value {
    stored[CODEBASE.symbols]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["qualified_name"] == name)
        .unwrap()
}

fn call(input: &mut CodeInput, from: usize, to: usize) {
    input.relationships.push(LocalRelationship {
        kind: RelationshipKind::Calls,
        from: LocalEndpoint::Symbol(from),
        to: LocalEndpoint::Symbol(to),
    });
}

#[tokio::test]
async fn shifted_and_reordered_symbols_keep_ids_and_all_external_references() {
    with_temp_db("code_moved_symbols", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        let receipt = code
            .commit(code.prepare(input("one.rs", ""), false).await.unwrap())
            .await
            .unwrap();
        let original = rows(&pool).await;
        let apple = symbol(&original, "apple")["_id"].as_str().unwrap();
        let banana = symbol(&original, "banana")["_id"].as_str().unwrap();
        let other = code
            .commit(code.prepare(input("other.rs", ""), false).await.unwrap())
            .await
            .unwrap();
        // Authored collections and another file's relationships are outside this
        // file's ownership, even when both endpoints belong to this file.
        crud::create_collection(&pool, "authored_links", Some(3))
            .await
            .unwrap();
        let mut references = Vec::new();
        for collection in ["authored_links", CODEBASE.calls_edges] {
            pool.writer()
                .post(
                    &format!("document/{collection}"),
                    &json!({
                        "_key":"external_reference","_from":apple,"_to":banana,
                        "file_key":other.file_key,"note":"keep this authored relationship"
                    }),
                )
                .await
                .unwrap();
            let path = format!("document/{collection}/external_reference");
            let before = pool.writer().get(&path).await.unwrap();
            references.push((path, before));
        }
        for (text, apple_line) in [
            (
                "// moved\nfn banana() { apple(); }\nfn apple() { let x = 1; }\n",
                3,
            ),
            (
                concat!(
                    "// moved again\n// café 日本語\n",
                    "fn banana() { apple(); }\nfn apple() { let x = 2; }\nfn cherry() {}\n"
                ),
                4,
            ),
        ] {
            let mut next = source(text);
            call(&mut next, 0, 1);
            let before = rows(&pool).await;
            let prepared = code.prepare(next, true).await.unwrap();
            assert_eq!(rows(&pool).await, before, "reconciliation is read-only");
            code.commit(prepared).await.unwrap();
            let after = rows(&pool).await;
            // Select this file explicitly: the second file uses the same names.
            let own: Vec<_> = after[CODEBASE.symbols]
                .as_array()
                .unwrap()
                .iter()
                .filter(|row| row["file_key"] == receipt.file_key)
                .collect();
            let find = |name: &str| *own.iter().find(|row| row["name"] == name).unwrap();
            assert_eq!(find("apple")["_id"], apple);
            assert_eq!(find("banana")["_id"], banana);
            assert_eq!(find("apple")["start_line"], apple_line);
            let calls = after[CODEBASE.calls_edges].as_array().unwrap();
            let edge = calls
                .iter()
                .find(|row| row["file_key"] == receipt.file_key)
                .unwrap();
            assert_eq!(edge["_from"], banana);
            assert_eq!(edge["_to"], apple);
            assert_eq!(edge["_key"], original[CODEBASE.calls_edges][0]["_key"]);
            for (path, before) in &references {
                assert_eq!(pool.writer().get(path).await.unwrap(), *before);
            }
            for (collection, _) in CODEBASE.all_collections() {
                let unrelated = |snapshot: &Value| {
                    snapshot[collection]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|row| {
                            row["file_key"] == other.file_key || row["_key"] == other.file_key
                        })
                        .cloned()
                        .collect::<Vec<_>>()
                };
                assert_eq!(unrelated(&before), unrelated(&after));
            }
        }
    })
    .await;
}

fn module(name: &str, body: &str) -> String {
    format!(
        "mod {name} {{\n    struct Widget;\n    impl Widget {{\n        fn new() {{\n            {body}\n        }}\n    }}\n}}\n"
    )
}

/// Supply explicit compiler identities without depending on an installed libclang.
fn compiler_source(text: &str) -> CodeInput {
    let mut input = source(text);
    input.analysis.analyzer = "fixture.compiler".into();
    input.analysis.analysis_tier = AnalysisTier::Semantic;
    let modules: Vec<_> = input
        .analysis
        .symbols
        .iter()
        .filter(|symbol| symbol.kind == code::SymbolKind::Module)
        .map(|symbol| (symbol.name.clone(), symbol.start_line, symbol.end_line))
        .collect();
    for symbol in &mut input.analysis.symbols {
        if symbol.qualified_name() == "Widget::new" {
            let (name, _, _) = modules
                .iter()
                .find(|(_, start, end)| symbol.start_line > *start && symbol.end_line < *end)
                .unwrap();
            symbol.metadata["usr"] = json!(format!("fixture::{name}::Widget::new"));
        }
    }
    input
}

fn overload(input: &CodeInput, owner: &str) -> usize {
    input
        .analysis
        .symbols
        .iter()
        .position(|symbol| symbol.metadata["usr"] == format!("fixture::{owner}::Widget::new"))
        .unwrap()
}

fn compiler_symbol_id(stored: &Value, owner: &str) -> String {
    stored[CODEBASE.symbols]
        .as_array()
        .unwrap()
        .iter()
        .find(|symbol| symbol["metadata"]["usr"] == format!("fixture::{owner}::Widget::new"))
        .unwrap()["_id"]
        .as_str()
        .unwrap()
        .into()
}

#[tokio::test]
async fn compiler_identities_preserve_overloads_across_moves_body_swaps_and_additions() {
    with_temp_db("code_overload_moves", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        let first = module("first", "/* café apple */");
        let second = module("second", "/* banana */");
        let mut initial = compiler_source(&(first.clone() + &second));
        let from = overload(&initial, "first");
        let to = overload(&initial, "second");
        call(&mut initial, from, to);
        code.commit(code.prepare(initial, false).await.unwrap())
            .await
            .unwrap();
        let before = rows(&pool).await;
        let first_id = compiler_symbol_id(&before, "first");
        let second_id = compiler_symbol_id(&before, "second");
        let last = first.clone() + &second + &module("third", "/* cherry */");
        for text in [
            second.clone() + &first,
            module("first", "/* banana */") + &module("second", "/* café apple */"),
            last.clone(),
        ] {
            let mut next = compiler_source(&text);
            let from = overload(&next, "first");
            let to = overload(&next, "second");
            call(&mut next, from, to);
            code.commit(code.prepare(next, true).await.unwrap())
                .await
                .unwrap();
            let after = rows(&pool).await;
            assert_eq!(compiler_symbol_id(&after, "first"), first_id);
            assert_eq!(compiler_symbol_id(&after, "second"), second_id);
            let edge = &after[CODEBASE.calls_edges][0];
            assert_eq!(
                edge["_from"], first_id,
                "simultaneous key swaps must not chase remapped endpoints"
            );
            assert_eq!(edge["_to"], second_id);
            assert_eq!(edge["_key"], before[CODEBASE.calls_edges][0]["_key"]);
        }
        for changed in [json!("different-compiler-identity"), json!(null)] {
            let mut next = compiler_source(&last);
            let index = overload(&next, "first");
            next.analysis.symbols[index].metadata["usr"] = changed;
            refused_before_planning(&code, &pool, &peer, next, "symbol endpoints").await;
        }
    })
    .await;
}

async fn refused_before_planning(
    code: &ProfiledCode,
    pool: &ArangoPool,
    peer: &Peer,
    next: CodeInput,
    message: &str,
) {
    let before = rows(pool).await;
    let plans = peer.plans().len();
    let requests = peer.requests().len();
    let error = code.prepare(next, true).await.err().unwrap().to_string();
    assert!(error.contains(message), "{error}");
    assert_eq!(peer.plans().len(), plans);
    assert_eq!(peer.requests().len(), requests);
    assert_eq!(rows(pool).await, before);
}

#[tokio::test]
async fn ambiguous_declarations_and_changed_identity_never_repurpose_an_endpoint() {
    with_temp_db(
        "code_ambiguous_symbols",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::code(2);
            fixture.install_code(&pool).await;
            let peer = Peer::with_workload_description(&fixture);
            let code = connect(&pool, &peer).await;
            let text = "#[cfg(unix)]\nfn choose() {}\n#[cfg(windows)]\nfn choose() {}\n".to_owned();
            code.commit(code.prepare(source(&text), false).await.unwrap())
                .await
                .unwrap();
            let ids = rows(&pool).await[CODEBASE.symbols]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row["_id"].clone())
                .collect::<Vec<_>>();
            // Re-embedding unchanged duplicate declarations is safe at their exact positions.
            code.commit(code.prepare(source(&text), true).await.unwrap())
                .await
                .unwrap();
            assert_eq!(
                rows(&pool).await[CODEBASE.symbols]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| row["_id"].clone())
                    .collect::<Vec<_>>(),
                ids
            );
            refused_before_planning(
                &code,
                &pool,
                &peer,
                source(&("// shift\n".to_string() + &text)),
                "ambiguous symbol identity",
            )
            .await;
            refused_before_planning(
                &code,
                &pool,
                &peer,
                source(&text.replace("{}", "{ let x = 1; }")),
                "ambiguous symbol identity",
            )
            .await;

            let mut renamed = source(&text);
            for symbol in &mut renamed.analysis.symbols {
                symbol.name = "renamed".into();
            }
            refused_before_planning(&code, &pool, &peer, renamed, "symbol endpoints").await;
            let mut changed_kind = source(&text);
            for symbol in &mut changed_kind.analysis.symbols {
                symbol.kind = code::SymbolKind::Struct;
            }
            refused_before_planning(&code, &pool, &peer, changed_kind, "symbol endpoints").await;
            let mut language = source(&text);
            language.analysis.language = code::Language::Python;
            refused_before_planning(&code, &pool, &peer, language, "source language").await;
        },
    )
    .await;
}

#[tokio::test]
async fn a_new_symbol_gets_a_fresh_id_at_a_retained_endpoints_historical_line() {
    with_temp_db(
        "code_retained_collision",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::code(2);
            fixture.install_code(&pool).await;
            let peer = Peer::with_workload_description(&fixture);
            let code = connect(&pool, &peer).await;
            let first = module("first", "/* apple */");
            code.commit(code.prepare(compiler_source(&first), false).await.unwrap())
                .await
                .unwrap();
            let original_id = compiler_symbol_id(&rows(&pool).await, "first");
            let text = module("second", "/* banana */") + &first;
            let mut added_id = None;
            for prefix in ["", "// move both again\n"] {
                let mut next = compiler_source(&(prefix.to_string() + &text));
                let from = overload(&next, "second");
                let to = overload(&next, "first");
                call(&mut next, from, to);
                code.commit(code.prepare(next, true).await.unwrap())
                    .await
                    .unwrap();
                let stored = rows(&pool).await;
                assert_eq!(compiler_symbol_id(&stored, "first"), original_id);
                let fresh = compiler_symbol_id(&stored, "second");
                assert_ne!(fresh, original_id);
                if let Some(id) = &added_id {
                    assert_eq!(&fresh, id);
                } else {
                    assert!(fresh.starts_with(&format!("{}/bsym_", CODEBASE.symbols)));
                    added_id = Some(fresh.clone());
                }
                assert_eq!(stored[CODEBASE.calls_edges][0]["_from"], fresh);
                assert_eq!(stored[CODEBASE.calls_edges][0]["_to"], original_id);
            }
        },
    )
    .await;
}

#[tokio::test]
async fn malformed_compiler_identities_fail_before_planning_new_files() {
    with_temp_db("code_invalid_usr", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        for usr in [
            json!(42),
            json!(""),
            json!("x".repeat(4 * 1024 * 1024 + 1)),
            json!("bad\nidentity"),
        ] {
            let mut next = input("new.rs", "");
            next.analysis.symbols[0].metadata["usr"] = usr;
            let error = code.prepare(next, false).await.err().unwrap().to_string();
            assert!(error.contains("compiler symbol identity"), "{error}");
        }
        assert!(peer.plans().is_empty() && peer.requests().is_empty());
        assert!(
            rows(&pool)
                .await
                .as_object()
                .unwrap()
                .values()
                .all(|rows| rows.as_array().unwrap().is_empty())
        );
    })
    .await;
}

#[tokio::test]
async fn edited_missing_or_foreign_symbols_refuse_reconciliation_before_inference() {
    with_temp_db(
        "code_symbol_inventory",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::code(2);
            fixture.install_code(&pool).await;
            let peer = Peer::with_workload_description(&fixture);
            let code = connect(&pool, &peer).await;
            for change in ["edited", "missing", "foreign"] {
                let receipt = code
                    .commit(code.prepare(input(change, ""), false).await.unwrap())
                    .await
                    .unwrap();
                let stored = rows(&pool).await;
                let symbol = stored[CODEBASE.symbols]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|row| row["file_key"] == receipt.file_key)
                    .unwrap();
                let path = format!("document/{}", symbol["_id"].as_str().unwrap());
                match change {
                    "edited" => {
                        pool.writer()
                            .patch(&path, &json!({"metadata":{"authored_note":"keep"}}))
                            .await
                            .unwrap();
                    }
                    "missing" => {
                        pool.writer().delete(&path).await.unwrap();
                    }
                    _ => {
                        let orphan = json!({
                            "_key":"orphan","file_key":receipt.file_key,"authored_note":"keep"
                        });
                        pool.writer()
                            .post(&format!("document/{}", CODEBASE.symbols), &orphan)
                            .await
                            .unwrap();
                    }
                }
                refused_before_planning(
                    &code,
                    &pool,
                    &peer,
                    shifted_input(change, ""),
                    "symbol inventory differs",
                )
                .await;
            }
        },
    )
    .await;
}

#[tokio::test]
async fn symbol_edits_during_planning_are_detected_before_publication() {
    with_temp_db("code_symbol_race", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = Arc::new(connect(&pool, &peer).await);
        code.commit(code.prepare(input("one.rs", ""), false).await.unwrap())
            .await
            .unwrap();
        let (started, release) = peer.hold_next_plan();
        let task = {
            let code = code.clone();
            tokio::spawn(async move { code.prepare(shifted_input("one.rs", ""), true).await })
        };
        started.await.unwrap();
        let before = rows(&pool).await;
        let path = format!(
            "document/{}",
            symbol(&before, "apple")["_id"].as_str().unwrap()
        );
        pool.writer()
            .patch(&path, &json!({"authored_note":"preserve"}))
            .await
            .unwrap();
        release.send(()).unwrap();
        let prepared = task.await.unwrap().unwrap();
        let edited = rows(&pool).await;
        assert!(
            code.commit(prepared)
                .await
                .unwrap_err()
                .to_string()
                .contains("inventory differs")
        );
        assert_eq!(rows(&pool).await, edited);
    })
    .await;
}
