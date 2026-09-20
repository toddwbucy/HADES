// Actual CLI batch insertion outcome against the runner's private database.
#[tokio::test]
async fn batch_insert_partial_failure_is_not_reported_as_success() {
    with_temp_db("cli_insert_outcomes", Fixtures::Empty, |pool| async move {
        use hades_core::db::crud;
        crud::create_collection(&pool, "documents", Some(2)).await.unwrap();
        crud::insert_document(&pool, "documents", &json!({"_key":"existing", "value":"original"})).await.unwrap();
        let before = crud::get_document(&pool, "documents", "existing").await.unwrap();
        let embedder = Embedder::new().await;
        let payload = json!([
            {"_key":"new", "value":"inserted"},
            {"_key":"existing", "value":"must-not-replace"}
        ]).to_string();
        let output = tokio::time::timeout(std::time::Duration::from_secs(10),
            cli_command(&pool, &embedder, &["db", "insert", "documents", "--data", &payload]).output()
        ).await.unwrap().unwrap();
        let existing = crud::get_document(&pool, "documents", "existing").await.unwrap();
        let inserted = crud::get_document(&pool, "documents", "new").await.unwrap();
        assert_eq!(existing, before, "existing row/revision must be preserved");
        assert_eq!(inserted["value"], "inserted");
        println!("Partial insert evidence: {}", json!({
            "exit_code":output.status.code(), "stdout":String::from_utf8_lossy(&output.stdout),
            "stderr":String::from_utf8_lossy(&output.stderr),
            "new_document_committed":true, "existing_document_and_revision_preserved":true
        }));
        assert!(!output.status.success(), "partially failed insert must exit nonzero");
    }).await;
}
