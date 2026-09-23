// Schema imports against the runner-owned disposable ArangoDB instance.
#[tokio::test]
async fn schema_apply_real_imports_replace_preserve_and_fail_explicitly() {
    with_temp_db("schema_outcomes", Fixtures::Empty, |pool| async move {
        use hades_core::db::crud;
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("schema.yaml");
        let mut schema = json!({
            "collections":[{"name":"docs","type":"document"},{"name":"links","type":"edge"}],
            "docs":[{"_key":"one","value":1}],
            "edge_definitions":[{"name":"links","from_collections":["docs"],"to_collections":["docs"]}],
            "named_graphs":[{"name":"example","edges":["links"]}]
        });
        let embedder = Embedder::new().await;
        std::fs::write(&file, schema.to_string()).unwrap();
        let first = cli_command(&pool, &embedder, &["schema","apply",file.to_str().unwrap()]).output().await.unwrap();
        assert!(first.status.success(), "{first:?}");
        assert_eq!(crud::get_document(&pool,"docs","one").await.unwrap()["value"],1);
        assert_eq!(crud::count_collection(&pool,"hades_schema").await.unwrap(),3);
        let graph = pool.reader().get("gharial/example").await.unwrap();
        assert_eq!(graph["graph"]["edgeDefinitions"][0]["collection"],"links");
        crud::insert_document(&pool,"docs",&json!({"_key":"untouched","value":9})).await.unwrap();
        schema["docs"][0]["value"] = json!(2);
        std::fs::write(&file, schema.to_string()).unwrap();
        let guard = cli_command(&pool, &embedder, &["schema","apply",file.to_str().unwrap()]).output().await.unwrap();
        assert!(!guard.status.success());
        assert_eq!(crud::get_document(&pool,"docs","one").await.unwrap()["value"],1);
        let replace = cli_command(&pool, &embedder, &["schema","apply",file.to_str().unwrap(),"--force"]).output().await.unwrap();
        assert!(replace.status.success(), "{replace:?}");
        let result: Value = serde_json::from_slice(&replace.stdout).unwrap();
        assert_eq!(result["data"]["result"]["named_graphs_skipped_existing"],1);
        assert_eq!(crud::get_document(&pool,"docs","one").await.unwrap()["value"],2);
        assert_eq!(crud::get_document(&pool,"docs","untouched").await.unwrap()["value"],9);
        let meta = crud::get_document(&pool,"hades_schema","meta").await.unwrap();
        schema["docs"] = json!([{"_key":"one","value":3},{"_key":"bad/key","value":4}]);
        std::fs::write(&file, schema.to_string()).unwrap();
        let rejected = cli_command(&pool, &embedder, &["schema","apply",file.to_str().unwrap(),"--force"]).output().await.unwrap();
        assert!(!rejected.status.success(), "{rejected:?}");
        assert!(rejected.stdout.is_empty());
        let diagnostic = String::from_utf8_lossy(&rejected.stderr);
        assert!(diagnostic.contains("document seeds"), "{diagnostic}");
        assert!(diagnostic.contains("earlier operations may have committed"));
        assert_eq!(crud::get_document(&pool,"docs","one").await.unwrap()["value"],2);
        assert_eq!(crud::count_collection(&pool,"docs").await.unwrap(),2);
        assert_eq!(crud::get_document(&pool,"hades_schema","meta").await.unwrap()["_rev"],meta["_rev"]);
        println!("Schema real-backend rejection: {}",json!({"exit":rejected.status.code(),"stderr":diagnostic,
            "prior_seed_value":2,"unrelated_value":9,"document_count":2,"metadata_revision_unchanged":true}));
    }).await;
}

#[tokio::test]
async fn weavertools_schema_apply_ensures_lookup_indexes_idempotently() {
    with_temp_db("lookup_schema", Fixtures::Empty, |pool| async move {
        let embedder=Embedder::new().await;
        let path=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../services/adapters/weavertools/schema.yaml");
        let mut previous=std::collections::BTreeMap::new();
        for pass in 0..2 {
            let output=cli_command(&pool,&embedder,&["schema","apply",path.to_str().unwrap()]).output().await.unwrap();
            assert!(output.status.success(),"{output:?}");
            for suffix in ["assertions","documents","vocabulary","crates","terms","axioms","artifacts","systems"] {
                let collection=format!("wt_{suffix}");
                let indexes=hades_core::db::index::list_indexes(&pool,&collection).await.unwrap();
                let matches:Vec<_>=indexes.iter().filter(|i|i.index_type=="persistent" && i.fields==["ident"]).collect();
                assert_eq!(matches.len(),1,"{collection}: {indexes:?}");
                if pass==0 { previous.insert(collection,matches[0].id.clone()); }
                else { assert_eq!(previous[&collection],matches[0].id); }
            }
        }
    }).await;
}
