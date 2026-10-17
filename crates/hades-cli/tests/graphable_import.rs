//! Real CLI and database, no embedder or extractor required. The optional source
//! is supplied explicitly by the isolated runner, never discovered on disk.
use hades_core::{
    db::crud,
    graphable::Plan,
    test_support::{Fixtures, with_temp_db},
};
use serde_json::{Value, json};
use std::path::PathBuf;
use tokio::process::Command;

#[tokio::test]
async fn graphable_cli_archives_maps_and_repeats_without_embedding() {
    with_temp_db("graphable_cli",Fixtures::Empty,|pool|async move {
        let temp=tempfile::tempdir().unwrap();
        let file=std::env::var_os("HADES_GRAPHABLE_SOURCE").map(PathBuf::from).unwrap_or_else(||{
            let p=temp.path().join("sample.md");
            std::fs::write(&p,include_str!("../../hades-core/tests/fixtures/graphable/sample.md")).unwrap();p
        });
        let archive=std::env::var_os("HADES_GRAPHABLE_ARCHIVE").map(PathBuf::from);
        let note="Reviewed mechanical export supplied by the operator; original bytes retained separately; no interpretation performed by importer";
        let mut plan=Plan::from_path(&file,"theory").unwrap();
        if let Some(path)=&archive {plan=plan.with_original_source(path,note).unwrap();}
        let run=|dry:bool|{
            let file=file.clone();let archive=archive.clone();let database=pool.database().to_owned();
            async move {
                let mut cmd=Command::new(env!("CARGO_BIN_EXE_hades"));
                cmd.args(["--db",&database,"graphable","import"]).arg(file)
                    .args(["--graph","theory","--create-graph"]);
                if dry {cmd.arg("--dry-run");}
                if let Some(path)=archive {cmd.arg("--source-archive").arg(path).args(["--extraction-note",note]);}
                let output=cmd.output().await.unwrap();
                assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
                serde_json::from_slice::<Value>(&output.stdout).unwrap()["data"].clone()
            }
        };
        let preview=run(true).await;assert_eq!(preview["mapped"],false);
        assert!(pool.reader().get("collection/theory_nodes").await.unwrap_err().is_not_found());
        let first=run(false).await;let repeat=run(false).await;
        assert_eq!(first["inserted_nodes"],plan.nodes().len());
        assert_eq!(first["inserted_edges"],plan.edges().len());
        assert_eq!(repeat["unchanged"],true);
        assert_eq!(repeat["inserted_nodes"],0);assert_eq!(repeat["inserted_edges"],0);
        for (collection,rows) in [("theory_nodes",plan.nodes()),("theory_edges",plan.edges())] {
            for row in rows {
                let mut actual=crud::get_document(&pool,collection,row["_key"].as_str().unwrap()).await.unwrap();
                actual.as_object_mut().unwrap().remove("_id");actual.as_object_mut().unwrap().remove("_rev");
                assert_eq!(&actual,row);
            }
        }
        let receipt=pool.reader().get(&format!("document/{}",first["source_archive_id"].as_str().unwrap())).await.unwrap();
        use base64::Engine;
        let decode=|v:&Value|base64::engine::general_purpose::STANDARD.decode(v.as_str().unwrap()).unwrap();
        assert_eq!(decode(&receipt["source_base64"]),std::fs::read(&file).unwrap());
        if let Some(archive)=&archive {assert_eq!(decode(&receipt["original_source_base64"]),std::fs::read(archive).unwrap());}
        let text=plan.nodes()[0]["source_text"].as_str().unwrap();
        for edge in plan.edges() {
            for evidence in edge["evidence"].as_array().unwrap() {
                let loc=&evidence["location"];
                assert_eq!(&text[loc["start_byte"].as_u64().unwrap() as usize..loc["end_byte"].as_u64().unwrap() as usize],evidence["quote"].as_str().unwrap());
            }
        }
        assert_eq!(first["embedding"],"not_requested");
        assert!(pool.reader().get("collection/embeddings").await.unwrap_err().is_not_found());
        if let Some(path)=std::env::var_os("HADES_GRAPHABLE_REPORT") {
            let path=PathBuf::from(path);
            std::fs::write(&path,serde_json::to_vec_pretty(&json!({"first":first,"repeat":repeat,
                "verification":{"all_rows_exact":true,"source_archives_byte_identical":true,
                    "all_edge_quotes_resolve":true,"embedding_not_requested":true}})).unwrap()).unwrap();
            std::fs::write(path.with_extension("plan.json"),serde_json::to_vec_pretty(&plan).unwrap()).unwrap();
        }
    }).await;
}
