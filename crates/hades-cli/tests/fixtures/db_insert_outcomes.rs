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
        assert!(output.stdout.is_empty());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("1 succeeded, 1 failed") && error.contains("1210"));
        let repeated = tokio::time::timeout(std::time::Duration::from_secs(10),
            cli_command(&pool, &embedder, &["db", "insert", "documents", "--data", &payload]).output()
        ).await.unwrap().unwrap();
        assert!(!repeated.status.success());
        assert!(String::from_utf8_lossy(&repeated.stderr).contains("0 succeeded, 2 failed"));
        assert_eq!(crud::get_document(&pool, "documents", "existing").await.unwrap(), before);
        assert_eq!(crud::get_document(&pool, "documents", "new").await.unwrap(), inserted);
        let single = cli(&pool, &embedder, &["db", "insert", "documents", "--data",
            r#"{"_key":"single","error":true}"#], true).await;
        assert_eq!(single["_key"], "single");
        let batch = cli(&pool, &embedder, &["db", "insert", "documents", "--data",
            r#"[{"_key":"four"},{"_key":"five"}]"#], true).await;
        assert_eq!(batch.as_array().unwrap().len(), 2);
        for key in ["single", "four", "five"] {
            assert_eq!(crud::get_document(&pool, "documents", key).await.unwrap()["_key"], key);
        }
        println!("Insert controls: all-failed batch preserves exact rows/revisions; single and all-success batch commit expected keys");

    }).await;
}

// Execute projection and lookup against ArangoDB, not an already-projected mock (#186).
#[tokio::test]
async fn db_get_projects_and_validates_against_real_database() {
    use hades_core::dispatch::{dispatch, DaemonCommand, DbGetParams, DispatchError, HandlerError};
    use hades_core::db::crud;
    with_temp_db("get_projection", Fixtures::Empty, |pool| async move {
        crud::create_collection(&pool, "documents", Some(2)).await.unwrap();
        crud::insert_document(&pool, "documents", &json!({"_key":"doc","label":"fixture","text":"bulk","full_text":"paper","body":"body","embedding":[1,0]})).await.unwrap();
        let config = hades_core::HadesConfig::with_database(pool.database());
        for fields in [None, Some(vec!["label".into(), "full_text".into()]), Some(vec![])] {
            let result = dispatch(&pool, &config, DaemonCommand::DbGet(DbGetParams {collection:"documents".into(),key:"doc".into(),fields:fields.clone()})).await.unwrap();
            match fields {
                None => {
                    assert_eq!(result["label"], "fixture");
                    assert_eq!(result["_key"], "doc");
                    for field in ["text","full_text","body","embedding"] { assert!(result.get(field).is_none(), "{result}"); }
                }
                Some(fields) if fields.is_empty() => assert_eq!(result, json!({})),
                Some(_) => assert_eq!(result, json!({"label":"fixture","full_text":"paper"})),
            }
        }
        let missing = dispatch(&pool, &config, DaemonCommand::DbGet(DbGetParams {collection:"documents".into(),key:"missing".into(),fields:None})).await.unwrap_err();
        assert!(matches!(missing, DispatchError::Handler(HandlerError::DocumentNotFound {..})), "{missing}");
        for (collection,key,parameter) in [("", "documents/doc", "collection"), ("documents", "documents/doc", "key")] {
            let error = dispatch(&pool, &config, DaemonCommand::DbGet(DbGetParams {collection:collection.into(),key:key.into(),fields:None})).await.unwrap_err();
            assert!(matches!(error, DispatchError::Handler(HandlerError::InvalidParameter {ref name,..}) if name == parameter), "{error}");
        }
        let embedder = Embedder::new().await;
        let row = cli(&pool,&embedder,&["db","get","documents","doc","--fields","label,full_text"],true).await;
        assert_eq!(row,json!({"label":"fixture","full_text":"paper"}));
    }).await;
}
