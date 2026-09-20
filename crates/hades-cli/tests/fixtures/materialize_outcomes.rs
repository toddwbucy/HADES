// Real partial edge import and read-back on the runner's disposable database.
#[tokio::test]
async fn materialize_partial_import_preserves_committed_edges_and_reports_failure() {
    with_temp_db("materialize_outcomes", Fixtures::Empty, |pool| async move {
        use hades_core::db::crud;
        for (name, kind) in [("hades_schema",2),("docs",2),("edges",3)] {
            crud::create_collection(&pool,name,Some(kind)).await.unwrap();
        }
        crud::insert_document(&pool,"hades_schema",&json!([
            {"_key":"meta","schema_type":"schema_meta"},
            {"_key":"edges","schema_type":"edge_definition","name":"edges","source_field":"related",
             "from_collections":["docs"],"to_collections":["docs"]}
        ])).await.unwrap();
        // Valid document keys produce an overlong derived edge key for the
        // second reference; the first edge remains a valid import item.
        let long_key = "x".repeat(180);
        crud::insert_document(&pool,"docs",&json!([
            {"_key":"start","related":"docs/end"},{"_key":"end"},
            {"_key":long_key,"related":format!("docs/{long_key}")}
        ])).await.unwrap();
        let embedder=Embedder::new().await;
        let output=tokio::time::timeout(std::time::Duration::from_secs(10),
            cli_command(&pool,&embedder,&["db","graph","materialize"]).output()).await.unwrap().unwrap();
        let edge=crud::get_document(&pool,"edges","docs_start__docs_end").await.unwrap();
        assert_eq!(edge["_from"],"docs/start");
        assert_eq!(edge["_to"],"docs/end");
        assert_eq!(crud::count_collection(&pool,"edges").await.unwrap(),1);
        assert_eq!(crud::count_collection(&pool,"docs").await.unwrap(),3);
        assert!(!output.status.success(),"{output:?}");
        assert!(output.stdout.is_empty());
        let stderr=String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("materialization incomplete"));
        assert!(stderr.contains("1 of 2 documents failed"),"{stderr}");
        assert!(stderr.contains("\"edges_created\":1"),"{stderr}");
        println!("Materialization persisted partial outcome: {}",json!({"exit":output.status.code(),
            "stdout":String::from_utf8_lossy(&output.stdout),"stderr":stderr,
            "committed_edge_count":1,"source_document_count":3,"valid_edge_endpoints_verified":true}));
    }).await;
}
