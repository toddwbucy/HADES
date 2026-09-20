// Included by codebase_lifecycle.rs to reuse its private CLI/embedding fixtures.
#[tokio::test]
async fn changed_text_has_path_specific_structural_vector_retention() {
    with_temp_db("vector_retention", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::new().await;
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path().to_str().unwrap();
        let parsed = tree.path().join("parsed.py");
        let text = tree.path().join("text.legacy");
        std::fs::write(&parsed, "def target():\n    return 'quartz_original'\n").unwrap();
        std::fs::write(&text, "quartz_original plain text\n").unwrap();
        cli(&pool, &embedder, &["codebase", "ingest", root, "--unparsed-ext", "legacy"], true).await;
        let parsed_key = keys::scoped_file_key(root, "parsed.py");
        let text_key = keys::scoped_file_key(root, "text.legacy");
        let before_parsed = hades_core::db::crud::get_document(&pool, "codebase_files", &parsed_key).await.unwrap();
        let before_text = hades_core::db::crud::get_document(&pool, "codebase_files", &text_key).await.unwrap();
        for key in [&parsed_key, &text_key] {
            hades_core::db::crud::update_document(&pool, "codebase_files", key,
                &json!({"structural_embedding":[0.25, 0.75], "audit_marker":"prior generation"}))
                .await.unwrap();
        }
        let before = snapshot_graph(&pool).await;
        let symbols = before["codebase_symbols"].as_array().unwrap();
        assert!(!symbols.is_empty());
        for symbol in symbols {
            hades_core::db::crud::update_document(&pool, "codebase_symbols", symbol["_key"].as_str().unwrap(),
                &json!({"structural_embedding":[0.25, 0.75]})).await.unwrap();
        }
        std::fs::write(&parsed, "def target():\n    return 'sapphire_changed'\n").unwrap();
        std::fs::write(&text, "sapphire_changed plain text\n").unwrap();
        cli(&pool, &embedder, &["codebase", "ingest", root, "--unparsed-ext", "legacy"], true).await;
        let after_parsed = hades_core::db::crud::get_document(&pool, "codebase_files", &parsed_key).await.unwrap();
        let after_text = hades_core::db::crud::get_document(&pool, "codebase_files", &text_key).await.unwrap();
        assert_ne!(after_parsed["content_hash"], before_parsed["content_hash"]);
        assert_ne!(after_text["content_hash"], before_text["content_hash"]);
        assert!(after_parsed.get("structural_embedding").is_none());
        assert!(after_parsed.get("audit_marker").is_none());
        assert_eq!(after_text["structural_embedding"], json!([0.25, 0.75]));
        assert_eq!(after_text["audit_marker"], "prior generation");
        let after = snapshot_graph(&pool).await;
        let symbols = after["codebase_symbols"].as_array().unwrap();
        assert!(!symbols.is_empty());
        assert!(symbols.iter().all(|row| row.get("structural_embedding").is_none()));
        for key in [&parsed_key, &text_key] {
            let chunks: Vec<_> = after["codebase_chunks"].as_array().unwrap().iter()
                .filter(|row| row["file_key"].as_str() == Some(key.as_str())).collect();
            assert!(!chunks.is_empty());
            assert!(chunks.iter().any(|row| row.to_string().contains("sapphire_changed")));
            assert!(chunks.iter().all(|row| !row.to_string().contains("quartz_original")));
        }
        println!("Changed text verified: parsed file/symbol vectors removed; parser-free file vector retained; both content hashes and chunks refreshed.");
    }).await;
}
