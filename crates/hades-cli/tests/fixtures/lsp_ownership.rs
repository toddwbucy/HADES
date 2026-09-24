fn ownership_symbol(name: &str, kind: u32, line: u32, column: u32) -> Value {
    let range = json!({"start":{"line":line,"character":column},"end":{"line":line,"character":column+name.len() as u32}});
    json!({"name":name,"kind":kind,"range":range,"selectionRange":range})
}

async fn ownership_ingest(pool: &ArangoPool, embedder: &Embedder, root: &Path, scenario: &Value, go: bool) -> Value {
    std::fs::write(root.join(".lsp-scenario.json"), serde_json::to_vec(scenario).unwrap()).unwrap();
    let peer = Path::new(env!("CARGO_MANIFEST_DIR")).join("../hades-core/tests/fixtures/lsp_ownership.py");
    let output = cli_command(pool, embedder, &["codebase","ingest",root.to_str().unwrap(),"--force"])
        .env(if go {"HADES_GOPLS_PATH"} else {"HADES_RUST_ANALYZER_PATH"}, peer).output().await.unwrap();
    serde_json::from_slice(&output.stdout).unwrap_or_else(|_| panic!("{output:?}"))
}

fn assert_ownership_endpoints(graph: &Value) {
    let vertices: std::collections::HashSet<_> = ["codebase_files", "codebase_symbols"].iter()
        .flat_map(|name| graph[*name].as_array().unwrap().iter().map(|s| s["_id"].as_str().unwrap())).collect();
    for collection in ["codebase_calls_edges", "codebase_implements_edges", "codebase_defines_edges"] {
        for edge in graph[collection].as_array().unwrap() {
            assert!(vertices.contains(edge["_from"].as_str().unwrap()) && vertices.contains(edge["_to"].as_str().unwrap()), "dangling {edge}");
        }
    }
    for symbol in graph["codebase_symbols"].as_array().unwrap() {
        assert!(graph["codebase_defines_edges"].as_array().unwrap().iter().any(|edge| edge["_to"] == symbol["_id"]), "undefined {symbol}");
    }
}

#[tokio::test]
async fn ownership_null_document_and_callee_only_rename() {
    with_temp_db("ownership_callee", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::new().await;
        let tree = tempfile::tempdir().unwrap(); let root = tree.path();
        std::fs::write(root.join("Cargo.toml"), "[package]\nname=\"fixture\"\nversion=\"0.1.0\"\n[lib]\npath=\"b.rs\"\n").unwrap();
        std::fs::write(root.join("b.rs"), "mod c;\nfn b() { c::c(); }\n").unwrap();
        std::fs::write(root.join("c.rs"), "pub fn c() {}\n").unwrap();
        let b = ownership_symbol("b",12,1,3); let c = ownership_symbol("c",12,0,7);
        let mut scenario = json!({"documents":{"b.rs":[b],"c.rs":[c]},"responses":{"callHierarchy/outgoingCalls":{"b.rs:1":[{"to":c,"fromRanges":[]}]}}});
        scenario["responses"]["callHierarchy/outgoingCalls"]["b.rs:1"][0]["to"]["uri"] = json!("@/c.rs");
        assert_eq!(ownership_ingest(&pool,&embedder,root,&scenario,false).await["success"],true);
        let initial = snapshot_graph(&pool).await;
        assert_eq!(initial["codebase_calls_edges"].as_array().unwrap().len(),1);
        scenario["responses"]["textDocument/documentSymbol"] = json!({"c.rs":null});
        let report = ownership_ingest(&pool,&embedder,root,&scenario,false).await;
        assert_eq!(report["success"],false,"{report}");
        let retained = snapshot_graph(&pool).await; assert_ownership_endpoints(&retained);
        assert_eq!(retained["codebase_calls_edges"], initial["codebase_calls_edges"]);
        // Both requests now fail; C disappears from the source. B's old answer
        // cannot survive as a dangling edge merely because B's query failed too.
        scenario["responses"]["callHierarchy/outgoingCalls"]["b.rs:1"] = json!({"$error":"b unavailable"});
        std::fs::write(root.join("c.rs"), "pub fn renamed() {}\n").unwrap();
        let report = ownership_ingest(&pool,&embedder,root,&scenario,false).await;
        assert_eq!(report["success"],false);
        let renamed = snapshot_graph(&pool).await; assert_ownership_endpoints(&renamed);
        assert!(renamed["codebase_calls_edges"].as_array().unwrap().is_empty());
        scenario["responses"]["textDocument/documentSymbol"]["c.rs"]=json!([]);
        scenario["responses"]["callHierarchy/outgoingCalls"]["b.rs:1"]=json!([]);
        let report=ownership_ingest(&pool,&embedder,root,&scenario,false).await;
        assert_eq!(report["success"],true,"empty document-symbol array is valid: {report}");
        assert_ownership_endpoints(&snapshot_graph(&pool).await);

    }).await;
}

#[tokio::test]
async fn ownership_go_startup_failure_preserves_first_ingest_fallback() {
    with_temp_db("ownership_go_fallback", Fixtures::Codebase, |pool| async move {
        let embedder=Embedder::new().await; let tree=tempfile::tempdir().unwrap(); let root=tree.path();
        std::fs::write(root.join("go.mod"),"module fixture\ngo 1.22\n").unwrap();
        std::fs::write(root.join("main.go"),"package fixture\nfunc caller() { target() }\nfunc target() {}\n").unwrap();
        let scenario=json!({"documents":{},"responses":{"initialize":{"*":{"$error":"session unavailable"}}}});
        let report=ownership_ingest(&pool,&embedder,root,&scenario,true).await;
        assert_eq!(report["success"],false,"{report}");
        let graph=snapshot_graph(&pool).await; assert_ownership_endpoints(&graph);
        assert!(!graph["codebase_calls_edges"].as_array().unwrap().is_empty(),"missing tree-sitter fallback: {graph}");
        assert!(graph["codebase_calls_edges"].as_array().unwrap().iter().all(|e| e["analysis_tier"] != "semantic"));
    }).await;
}

#[tokio::test]
async fn ownership_go_receivers_have_independent_request_outcomes() {
    with_temp_db("ownership_go_receivers", Fixtures::Codebase, |pool| async move {
        let embedder=Embedder::new().await; let tree=tempfile::tempdir().unwrap(); let root=tree.path();
        std::fs::write(root.join("go.mod"),"module fixture\ngo 1.22\n").unwrap();
        let source="package fixture\ntype A struct{}\ntype B struct{}\nfunc (a A) Run() { target() }\nfunc (b B) Run() { target() }\nfunc target() {}\n";
        std::fs::write(root.join("main.go"),source).unwrap();
        let a=ownership_symbol("(A).Run",6,3,11); let b=ownership_symbol("(B).Run",6,4,11);
        let mut target=ownership_symbol("target",12,5,5); target["uri"]=json!("@/main.go");
        let mut scenario=json!({"documents":{"main.go":[a,b,target]},"responses":{"callHierarchy/outgoingCalls":{"main.go:3":[{"to":target}],"main.go:4":[{"to":target}]}}});
        assert_eq!(ownership_ingest(&pool,&embedder,root,&scenario,true).await["success"],true);
        let initial=snapshot_graph(&pool).await;
        let initial_semantic:Vec<_>=initial["codebase_calls_edges"].as_array().unwrap().iter().filter(|e|e["analysis_tier"]=="semantic").cloned().collect();
        assert_eq!(initial_semantic.len(),2,"{initial}");
        std::fs::write(root.join("main.go"),source.replace("func (b B) Run() { target() }","func (b B) Run() {}" )).unwrap();
        scenario["responses"]["callHierarchy/outgoingCalls"]["main.go:3"]=json!({"$error":"A failed"});
        scenario["responses"]["callHierarchy/outgoingCalls"]["main.go:4"]=json!([]);
        let report=ownership_ingest(&pool,&embedder,root,&scenario,true).await;
        assert_eq!(report["success"],false,"{report}");
        assert_eq!(report["data"]["gopls"]["failed_request_count"],1,"{report}");
        let graph=snapshot_graph(&pool).await; assert_ownership_endpoints(&graph);
        let fkey=keys::scoped_file_key(root.to_str().unwrap(),"main.go");
        let a_id=format!("codebase_symbols/{}",keys::symbol_key(&fkey,"Run",4));
        let b_id=format!("codebase_symbols/{}",keys::symbol_key(&fkey,"Run",5));
        assert!(graph["codebase_calls_edges"].as_array().unwrap().iter().any(|e|e["_from"]==a_id && e["analysis_tier"]=="semantic"));
        assert!(!graph["codebase_calls_edges"].as_array().unwrap().iter().any(|e|e["_from"]==b_id && e["analysis_tier"]=="semantic"));
    }).await;
}

#[tokio::test]
async fn ownership_source_macro_and_unicode_path_module_are_clean() {
    with_temp_db("ownership_cfg_macro", Fixtures::Codebase, |pool| async move {
        let embedder=Embedder::new().await; let tree=tempfile::tempdir().unwrap(); let root=tree.path();
        std::fs::write(root.join("Cargo.toml"),"[package]\nname=\"fixture\"\nversion=\"0.1.0\"\n[[bin]]\nname=\"fixture\"\npath=\"main.rs\"\n").unwrap();
        std::fs::create_dir(root.join("other")).unwrap();
        let declaration="/* é😀 */ #[cfg(feature=\"off\")] #[path=\"other/file.rs\"] mod m;";
        std::fs::write(root.join("main.rs"),format!("macro_rules! identity {{ ($x:expr) => {{$x}}; }}\n{declaration}\nfn main() {{}}\n")).unwrap();
        std::fs::write(root.join("other/file.rs"),"pub fn gated() {}\n").unwrap();
        let macro_symbol=ownership_symbol("identity",12,0,13);
        let main=ownership_symbol("main",12,2,3); let gated=ownership_symbol("gated",12,0,7);
        let start=declaration[..declaration.find("mod m").unwrap()+4].encode_utf16().count();
        let inactive=json!({"code":"inactive-code","range":{"start":{"line":1,"character":start},"end":{"line":1,"character":declaration.encode_utf16().count()}}});
        let scenario=json!({"documents":{"main.rs":[macro_symbol,main],"other/file.rs":[gated]},
            "diagnostics":{"main.rs":[inactive],"other/file.rs":[{"code":"unlinked-file","range":gated["range"]}]},
            "responses":{"textDocument/prepareCallHierarchy":{"main.rs:0":{"$error":"macro must not be queried"},"other/file.rs:0":null}}});
        let report=ownership_ingest(&pool,&embedder,root,&scenario,false).await;
        assert_eq!(report["success"],true,"{report}");
        assert_eq!(report["data"]["rust_analyzer"]["failed_request_count"],0);
        assert_eq!(report["data"]["rust_analyzer"]["no_call_count"],1);
        assert_ownership_endpoints(&snapshot_graph(&pool).await);
        let log=std::fs::read_to_string(root.join(".lsp-requests")).unwrap();
        assert!(!log.lines().map(|l|serde_json::from_str::<Value>(l).unwrap()).any(|r|r["method"]=="textDocument/prepareCallHierarchy" && r["file"]=="main.rs" && r["line"]==0));
    }).await;
}

#[tokio::test]
async fn ownership_go_implements_requires_every_interface_query() {
    with_temp_db("ownership_go_implements", Fixtures::Codebase, |pool| async move {
        let embedder=Embedder::new().await; let tree=tempfile::tempdir().unwrap(); let root=tree.path();
        std::fs::write(root.join("go.mod"),"module fixture\ngo 1.22\n").unwrap();
        std::fs::write(root.join("main.go"),"package fixture\ntype I interface {\nRun()\nStop()\n}\ntype A struct{}\nfunc (a A) Run() {}\nfunc (a A) Stop() {}\n").unwrap();
        let mut interface=ownership_symbol("I",11,1,5);
        interface["children"]=json!([ownership_symbol("Run",6,2,0),ownership_symbol("Stop",6,3,0)]);
        let a=ownership_symbol("(A).Run",6,6,11); let b=ownership_symbol("(A).Stop",6,7,11);
        let mut scenario=json!({"documents":{"main.go":[interface,a,b]},"responses":{"textDocument/implementation":{"main.go:2":[{"uri":"@/main.go","range":a["range"]}],"main.go:3":[{"uri":"@/main.go","range":b["range"]}]}}});
        assert_eq!(ownership_ingest(&pool,&embedder,root,&scenario,true).await["success"],true);
        let initial=snapshot_graph(&pool).await;
        assert_eq!(initial["codebase_implements_edges"].as_array().unwrap().len(),2,"{initial}");
        scenario["responses"]["textDocument/implementation"]["main.go:2"]=json!([]);
        scenario["responses"]["textDocument/implementation"]["main.go:3"]=json!({"$error":"Stop implementation query failed"});
        let report=ownership_ingest(&pool,&embedder,root,&scenario,true).await;
        assert_eq!(report["success"],false,"{report}");
        assert_eq!(snapshot_graph(&pool).await["codebase_implements_edges"],initial["codebase_implements_edges"]);
        scenario["responses"]["textDocument/implementation"]["main.go:3"]=json!([]);
        assert_eq!(ownership_ingest(&pool,&embedder,root,&scenario,true).await["success"],true);
        let graph=snapshot_graph(&pool).await; assert_ownership_endpoints(&graph);
        assert!(graph["codebase_implements_edges"].as_array().unwrap().is_empty());
    }).await;
}

#[tokio::test]
async fn ownership_skipped_store_still_defines_symbols_and_prunes_orphans() {
    with_temp_db("ownership_skipped", Fixtures::Codebase, |pool| async move {
        let embedder=Embedder::new().await; let tree=tempfile::tempdir().unwrap(); let root=tree.path();
        std::fs::write(root.join("Cargo.toml"),"[package]\nname=\"fixture\"\nversion=\"0.1.0\"\n[lib]\npath=\"lib.rs\"\n").unwrap();
        std::fs::write(root.join("lib.rs"),"fn caller() { target(); }\nfn target() {}\n").unwrap();
        let caller=ownership_symbol("caller",12,0,3); let mut target=ownership_symbol("target",12,1,3); target["uri"]=json!("@/lib.rs");
        let mut scenario=json!({"documents":{"lib.rs":[caller,target]},"responses":{"callHierarchy/outgoingCalls":{"lib.rs:0":[{"to":target}]}}});
        assert_eq!(ownership_ingest(&pool,&embedder,root,&scenario,false).await["success"],true);
        std::fs::write(root.join("bad.rs"),[0xff]).unwrap();
        std::fs::write(root.join("lib.rs"),"fn caller() {}\nfn renamed() {}\n").unwrap();
        scenario["documents"]["lib.rs"]=json!([caller,ownership_symbol("renamed",12,1,3),ownership_symbol("analyzer_only",12,1,3)]);
        scenario["responses"]["callHierarchy/outgoingCalls"]["lib.rs:0"]=json!([]);
        let report=ownership_ingest(&pool,&embedder,root,&scenario,false).await;
        assert_eq!(report["success"],false,"{report}");
        assert!(report["data"]["relationship_error"].as_str().unwrap().contains("deferred"));
        let graph=snapshot_graph(&pool).await; assert_ownership_endpoints(&graph);
        assert!(graph["codebase_symbols"].as_array().unwrap().iter().any(|s|s["name"]=="analyzer_only"));
        assert!(graph["codebase_calls_edges"].as_array().unwrap().is_empty(),"cleanup was skipped: {graph}");
    }).await;
}

#[tokio::test]
async fn ownership_nested_go_module_does_not_inherit_parent_failure() {
    with_temp_db("ownership_nested_go", Fixtures::Codebase, |pool| async move {
        let embedder=Embedder::new().await; let tree=tempfile::tempdir().unwrap(); let root=tree.path();
        let nested=root.join("nested"); std::fs::create_dir(&nested).unwrap();
        for path in [root,nested.as_path()] {
            std::fs::write(path.join("go.mod"),"module fixture\ngo 1.22\n").unwrap();
            std::fs::write(path.join("main.go"),"package fixture\nfunc caller() { target() }\nfunc target() {}\n").unwrap();
        }
        let caller=ownership_symbol("caller",12,1,5); let mut target=ownership_symbol("target",12,2,5); target["uri"]=json!("@/main.go");
        let nested_scenario=json!({"documents":{"main.go":[caller,target]},"responses":{"callHierarchy/outgoingCalls":{"main.go:1":[{"to":target}]}}});
        std::fs::write(nested.join(".lsp-scenario.json"),serde_json::to_vec(&nested_scenario).unwrap()).unwrap();
        let outer=json!({"documents":{},"responses":{"initialize":{"*":{"$error":"outer unavailable"}}}});
        let report=ownership_ingest(&pool,&embedder,root,&outer,true).await;
        assert_eq!(report["success"],false,"{report}");
        assert_eq!(report["data"]["gopls"]["modules_analyzed"],1,"{report}");
        let error=report["data"]["enrichment_error"].as_str().unwrap();
        assert!(!error.contains("nested/main.go"),"successful nested module misclassified: {report}");
        let graph=snapshot_graph(&pool).await; assert_ownership_endpoints(&graph);
        let fkey=keys::scoped_file_key(root.to_str().unwrap(),"nested/main.go");
        let id=format!("codebase_symbols/{}",keys::symbol_key(&fkey,"caller",2));
        assert!(graph["codebase_calls_edges"].as_array().unwrap().iter().any(|e|e["_from"]==id && e["analysis_tier"]=="semantic"));
    }).await;
}
