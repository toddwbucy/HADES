//! Reviewer reproductions using real structural analyzers and explicit compiler metadata.
use super::*;

fn scoped_id(stored: &Value, owner: &str) -> String {
    let parent = symbol(stored, owner);
    stored[CODEBASE.symbols]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| {
            row["qualified_name"] == "Widget::new"
                && row["start_line"].as_u64() > parent["start_line"].as_u64()
                && row["end_line"].as_u64() < parent["end_line"].as_u64()
        })
        .unwrap()["_id"]
        .as_str()
        .unwrap()
        .into()
}

#[tokio::test]
async fn structural_method_ids_follow_their_containers_not_order_or_body_text() {
    with_temp_db("code_lexical_scopes", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        let first = module("first", "/* apple */");
        let second = module("second", "/* banana */");
        code.commit(
            code.prepare(source(&(first.clone() + &second)), false)
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let before = rows(&pool).await;
        for text in [
            second + &first,
            module("first", "/* banana */") + &module("second", "/* apple */"),
        ] {
            code.commit(code.prepare(source(&text), true).await.unwrap())
                .await
                .unwrap();
            let after = rows(&pool).await;
            for owner in ["first", "second"] {
                assert_eq!(scoped_id(&after, owner), scoped_id(&before, owner));
            }
        }
    })
    .await;
}

#[tokio::test]
async fn moving_a_unique_method_between_existing_modules_refuses_before_inference() {
    with_temp_db(
        "code_container_change",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::code(2);
            fixture.install_code(&pool).await;
            let peer = Peer::with_workload_description(&fixture);
            let code = connect(&pool, &peer).await;
            let empty = |name| format!("mod {name} {{\n    struct Widget;\n}}\n");
            let initial = source(&(module("first", "/* old */") + &empty("second")));
            code.commit(code.prepare(initial, false).await.unwrap())
                .await
                .unwrap();
            let stored = rows(&pool).await;
            let id = scoped_id(&stored, "first");
            crud::create_collection(&pool, "authored_links", Some(3))
                .await
                .unwrap();
            pool.writer()
                .post(
                    "document/authored_links",
                    &json!({
                        "_key":"reference", "_from":id, "_to":id
                    }),
                )
                .await
                .unwrap();
            let authored = pool
                .writer()
                .get("document/authored_links/reference")
                .await
                .unwrap();
            let next = source(&(empty("first") + &module("second", "/* unrelated */")));
            refused_before_planning(&code, &pool, &peer, next, "symbol endpoints").await;
            assert_eq!(
                pool.writer()
                    .get("document/authored_links/reference")
                    .await
                    .unwrap(),
                authored
            );
            // The closing-brace line can also contain a sibling impl. Treating
            // inclusive line containment as ownership would silently transfer
            // the outer Widget's method ID to first::Widget's method here.
            let mut boundary = source(concat!(
                "struct Widget;\nmod first {\n    struct Widget;\n",
                "} impl Widget { fn new() {} }\n"
            ));
            boundary.path = "boundary.rs".into();
            code.commit(code.prepare(boundary, false).await.unwrap())
                .await
                .unwrap();
            let mut interior = source(concat!(
                "struct Widget;\nmod first {\n    struct Widget;\n",
                "    impl Widget {\n        fn new() {}\n    }\n}\n"
            ));
            interior.path = "boundary.rs".into();
            refused_before_planning(&code, &pool, &peer, interior, "symbol endpoints").await;
            // Free functions already carry module paths; the real gap was impl methods.
            let free = source("mod v1 { pub fn parse() {} }\n");
            assert!(
                free.analysis
                    .symbols
                    .iter()
                    .any(|s| s.qualified_name() == "v1::parse")
            );
        },
    )
    .await;
}

#[tokio::test]
async fn indistinguishable_same_line_containers_refuse_source_edits() {
    with_temp_db("code_same_line_scope", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        let text = "mod first { struct Widget; impl Widget { fn new() {} } }\n";
        code.commit(code.prepare(source(text), false).await.unwrap())
            .await
            .unwrap();
        code.commit(code.prepare(source(text), true).await.unwrap())
            .await
            .unwrap();
        refused_before_planning(
            &code,
            &pool,
            &peer,
            source(&format!("// shift\n{text}")),
            "ambiguous symbol identity",
        )
        .await;
    })
    .await;
}

/// Match libclang's shared-USR, distinct-is_definition output without requiring
/// a workstation libclang installation in this storage contract.
fn declarations(prefix: &str) -> CodeInput {
    let mut input = source(&format!(
        "{prefix}fn helper() {{}}\nfn helper() {{ /* definition */ }}\n"
    ));
    input.analysis.language = code::Language::Cpp;
    input.analysis.analyzer = "fixture.libclang".into();
    input.analysis.analysis_tier = AnalysisTier::Semantic;
    for (index, symbol) in input.analysis.symbols.iter_mut().enumerate() {
        symbol.metadata["usr"] = json!("c:@F@helper#");
        symbol.metadata["is_definition"] = json!(index == 1);
    }
    input
}

#[tokio::test]
async fn declaration_and_definition_with_shared_usr_keep_separate_endpoints() {
    with_temp_db(
        "code_declaration_roles",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::code(2);
            fixture.install_code(&pool).await;
            let peer = Peer::with_workload_description(&fixture);
            let code = connect(&pool, &peer).await;
            code.commit(code.prepare(declarations(""), false).await.unwrap())
                .await
                .unwrap();
            let before = rows(&pool).await;
            code.commit(code.prepare(declarations("// move\n"), true).await.unwrap())
                .await
                .unwrap();
            let after = rows(&pool).await;
            for role in [false, true] {
                let find = |rows: &Value| {
                    rows[CODEBASE.symbols]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|r| r["metadata"]["is_definition"] == role)
                        .unwrap()["_id"]
                        .clone()
                };
                assert_eq!(find(&before), find(&after));
            }
            let mut changed = declarations("// move\n");
            changed.analysis.symbols[0].metadata["is_definition"] = json!(true);
            refused_before_planning(&code, &pool, &peer, changed, "symbol endpoints").await;
            let mut malformed = declarations("");
            malformed.analysis.symbols[0].metadata["is_definition"] = json!("yes");
            refused_before_planning(&code, &pool, &peer, malformed, "definition role").await;
        },
    )
    .await;
}

#[tokio::test]
async fn python_last_methods_keep_their_container_identity_across_edits() {
    with_temp_db("code_python_scope", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        let text = "class C:\n    def f(self):\n        return 1\n";
        let python = |text: &str| CodeInput {
            namespace: "fixture/project".into(),
            path: "scope.py".into(),
            text: text.into(),
            analysis: code::analyze(text, "scope.py").unwrap(),
            relationships: vec![],
        };
        code.commit(code.prepare(python(text), false).await.unwrap())
            .await
            .unwrap();
        let before = rows(&pool).await;
        code.commit(
            code.prepare(python(&format!("# move\n{text}")), true)
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let after = rows(&pool).await;
        assert_eq!(symbol(&before, "C.f")["_id"], symbol(&after, "C.f")["_id"]);
    })
    .await;
}

#[tokio::test]
async fn real_python_property_and_overload_duplicates_require_unchanged_source() {
    with_temp_db("code_python_duplicates", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        for (path, text, name) in [
            ("property.py", "class C:\n    @property\n    def x(self):\n        return 1\n    @x.setter\n    def x(self, value):\n        pass\n", "C.x"),
            ("overloads.py", "from typing import overload\n@overload\ndef f(x: int) -> int: ...\n@overload\ndef f(x: str) -> str: ...\ndef f(x):\n    return x\n", "f"),
        ] {
            let python = |text: &str| CodeInput {
                namespace: "fixture/project".into(), path: path.into(), text: text.into(),
                analysis: code::analyze(text, path).unwrap(), relationships: vec![],
            };
            let initial = python(text);
            assert!(initial.analysis.symbols.iter().filter(|s| s.qualified_name() == name).count() > 1);
            code.commit(code.prepare(initial, false).await.unwrap()).await.unwrap();
            code.commit(code.prepare(python(text), true).await.unwrap()).await.unwrap();
            refused_before_planning(&code, &pool, &peer, python(&format!("# move\n{text}")), "ambiguous symbol identity").await;
            let mut upgrade = python(text);
            upgrade.analysis.analysis_tier = AnalysisTier::Semantic;
            for (index, symbol) in upgrade.analysis.symbols.iter_mut().enumerate() {
                if symbol.qualified_name() == name {
                    symbol.metadata["usr"] = json!(format!("compiler:{index}"));
                }
            }
            refused_before_planning(&code, &pool, &peer, upgrade, "symbol endpoints").await;
        }
    }).await;
}

#[tokio::test]
async fn unique_analyzer_upgrade_accepts_long_usrs_and_preserves_ids() {
    with_temp_db("code_usr_upgrade", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        code.commit(code.prepare(input("one.rs", ""), false).await.unwrap())
            .await
            .unwrap();
        let before = rows(&pool).await;
        let upgraded = |shift| {
            let mut next = if shift {
                shifted_input("one.rs", "")
            } else {
                input("one.rs", "")
            };
            next.analysis.analysis_tier = AnalysisTier::Semantic;
            next.analysis.analyzer = "fixture.compiler".into();
            next.analysis.symbols[0].metadata["usr"] = json!("template".repeat(1024));
            next.analysis.symbols[0].metadata["is_definition"] = json!(true);
            next
        };
        for shift in [false, true] {
            code.commit(code.prepare(upgraded(shift), true).await.unwrap())
                .await
                .unwrap();
            let after = rows(&pool).await;
            assert_eq!(
                symbol(&before, "apple")["_id"],
                symbol(&after, "apple")["_id"]
            );
            assert_eq!(
                symbol(&after, "apple")["metadata"]["usr"]
                    .as_str()
                    .unwrap()
                    .len(),
                8192
            );
        }
        for usr in [json!(null), json!("changed")] {
            let mut next = upgraded(true);
            next.analysis.symbols[0].metadata["usr"] = usr;
            refused_before_planning(&code, &pool, &peer, next, "symbol endpoints").await;
        }
    })
    .await;
}
