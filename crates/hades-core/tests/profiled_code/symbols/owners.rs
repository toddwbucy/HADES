//! Full Rust implementation identities and upgrades through verified owner chains.
use super::*;
use std::collections::BTreeMap;

fn implementation_ids(stored: &Value) -> BTreeMap<String, Value> {
    stored[CODEBASE.symbols]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| {
            row["metadata"]["impl_identity"]
                .as_str()
                .map(|identity| (identity.into(), row["_id"].clone()))
        })
        .collect()
}

#[tokio::test]
async fn generic_trait_and_reference_impls_survive_reordering_and_body_edits() {
    with_temp_db("code_impl_headers", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        for (path, intro, first, second) in [
            ("from.rs", "struct MyError;\n",
                "impl From<io::Error> for MyError {\n    fn from(_: io::Error) -> Self { Self }\n}\n",
                "impl From<serde_json::Error> for MyError {\n    fn from(_: serde_json::Error) -> Self { Self }\n}\n"),
            ("refs.rs", "struct A;\nstruct B;\ntrait T {}\n",
                "impl T for &A {\n    fn run(&self) {}\n}\n",
                "impl T for &mut B {\n    fn run(&self) {}\n}\n"),
        ] {
            let make = |text: String| { let mut input = source(&text); input.path = path.into(); input };
            let initial = make(format!("{intro}{first}{second}"));
            let methods: Vec<_> = initial.analysis.symbols.iter().filter(|s| s.metadata.get("impl_identity").is_some()).collect();
            assert_eq!(methods.len(), 2);
            assert_eq!(methods[0].qualified_name(), methods[1].qualified_name(), "legacy labels intentionally remain compatible");
            assert_ne!(methods[0].metadata["impl_identity"], methods[1].metadata["impl_identity"]);
            code.commit(code.prepare(initial, false).await.unwrap()).await.unwrap();
            let before = implementation_ids(&rows(&pool).await);
            let reordered = format!("// moved\n{intro}{second}{first}")
                .replace("From<", "From < ")
                .replace("{ Self }", "{ /* changed body */ Self }")
                .replace("fn run(&self) {}", "fn run(&self) { /* changed body */ }");
            code.commit(code.prepare(make(reordered), true).await.unwrap()).await.unwrap();
            assert_eq!(implementation_ids(&rows(&pool).await), before);
            // A new impl argument must not inherit the old implementation's ID.
            let changed = format!("{intro}{first}{second}").replace("serde_json::Error", "String").replace("&mut B", "&mut A");
            refused_before_planning(&code, &pool, &peer, make(changed), "symbol endpoints").await;
        }
    }).await;
}

/// Explicit C++ analyzer snapshots keep these storage contracts independent of
/// libclang installation, while Rust coverage above exercises the real parser.
fn cpp(text: String, symbols: Vec<code::Symbol>, tier: AnalysisTier) -> CodeInput {
    let lines = text.lines().count();
    CodeInput {
        namespace: "fixture/project".into(),
        path: "owners.cpp".into(),
        text,
        analysis: code::FileAnalysis {
            language: code::Language::Cpp,
            symbols,
            metrics: code::CodeMetrics {
                total_lines: lines,
                lines_of_code: lines,
                blank_lines: 0,
                comment_lines: 0,
                cyclomatic_complexity: 1,
                max_nesting_depth: 0,
            },
            symbol_hash: code::compute_content_hash("fixture"),
            top_level_defs: vec![],
            analysis_tier: tier,
            analyzer: "fixture.cpp".into(),
            fallback_reason: None,
        },
        relationships: vec![],
    }
}

fn cpp_symbol(name: &str, kind: code::SymbolKind, start: usize, end: usize) -> code::Symbol {
    code::Symbol {
        name: name.rsplit("::").next().unwrap().into(),
        kind,
        start_line: start,
        end_line: end,
        metadata: json!({"qualified_name":name}),
    }
}

fn nested_cpp(stage: u8) -> CodeInput {
    let prefix = if stage > 0 { "// moved\n" } else { "" };
    let offset = usize::from(stage > 0);
    let mut symbols = vec![
        cpp_symbol("N", code::SymbolKind::Module, 1, 7),
        cpp_symbol("N::C", code::SymbolKind::Class, 2, 6),
        cpp_symbol("N::C::Inner", code::SymbolKind::Class, 3, 5),
        cpp_symbol("N::C::Inner::f", code::SymbolKind::Function, 4, 4),
    ];
    for symbol in &mut symbols {
        symbol.start_line += offset;
        symbol.end_line += offset;
        if stage > 0 && (stage == 2 || symbol.kind != code::SymbolKind::Function) {
            symbol.metadata["usr"] = json!(format!("fixture:{}", symbol.qualified_name()));
            symbol.metadata["is_definition"] = json!(true);
        }
    }
    cpp(
        format!(
            "{prefix}namespace N {{\nclass C {{\nclass Inner {{\nint f() {{ return 1; }}\n}};\n}};\n}}\n"
        ),
        symbols,
        if stage == 0 {
            AnalysisTier::Structural
        } else {
            AnalysisTier::Semantic
        },
    )
}

#[tokio::test]
async fn nested_container_and_method_identity_upgrades_preserve_every_endpoint() {
    with_temp_db("code_nested_upgrade", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        code.commit(code.prepare(nested_cpp(0), false).await.unwrap())
            .await
            .unwrap();
        let ids = |stored: &Value| {
            stored[CODEBASE.symbols]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    (
                        r["qualified_name"].as_str().unwrap().to_string(),
                        r["_id"].clone(),
                    )
                })
                .collect::<BTreeMap<_, _>>()
        };
        let before = ids(&rows(&pool).await);
        // First upgrade only the owners, then the method; a fresh file also
        // exercises upgrading all nested levels in a single replacement below.
        for stage in [1, 2] {
            code.commit(code.prepare(nested_cpp(stage), true).await.unwrap())
                .await
                .unwrap();
            assert_eq!(ids(&rows(&pool).await), before);
        }
        let mut changed_owner = nested_cpp(2);
        changed_owner.analysis.symbols[1].metadata["usr"] = json!("different-class");
        refused_before_planning(&code, &pool, &peer, changed_owner, "symbol endpoints").await;
        for stage in [0, 2] {
            let mut input = nested_cpp(stage);
            input.path = "simultaneous.cpp".into();
            let receipt = code
                .commit(code.prepare(input, stage > 0).await.unwrap())
                .await
                .unwrap();
            assert_eq!(receipt.symbols, 4);
        }
    })
    .await;
}

#[tokio::test]
async fn compiler_distinct_containers_cannot_exchange_a_unique_weak_child() {
    with_temp_db("code_ambiguous_owner", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        let make = |moved| {
            let (text, first_end, second_start, child_line) = if moved {
                ("template<class T> struct Box;\ntemplate<> struct Box<int> {\n};\ntemplate<> struct Box<char> {\nint helper() { return 1; }\n};\n", 3, 4, 5)
            } else {
                ("template<class T> struct Box;\ntemplate<> struct Box<int> {\nint helper() { return 1; }\n};\ntemplate<> struct Box<char> {\n};\n", 4, 5, 3)
            };
            let mut symbols = vec![cpp_symbol("Box", code::SymbolKind::Class, 2, first_end),
                cpp_symbol("Box", code::SymbolKind::Class, second_start, 6),
                cpp_symbol("Box::helper", code::SymbolKind::Function, child_line, child_line)];
            for (index, symbol) in symbols[..2].iter_mut().enumerate() {
                symbol.metadata["usr"] = json!(format!("fixture:specialization:{index}"));
                symbol.metadata["is_definition"] = json!(true);
            }
            cpp(text.into(), symbols, AnalysisTier::Semantic)
        };
        code.commit(code.prepare(make(false), false).await.unwrap()).await.unwrap();
        refused_before_planning(&code, &pool, &peer, make(true), "ambiguous symbol identity").await;
    }).await;
}
