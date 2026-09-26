// A successful libclang analysis replaces a symbol's prior libclang call edges,
// including with "no calls"; a failed one keeps them (#194, #179/#184).

async fn libclang_calls(pool: &ArangoPool) -> Vec<Value> {
    hades_core::db::query::query(
        pool,
        "FOR e IN codebase_calls_edges FILTER e.analyzer == 'libclang' SORT e._key RETURN {caller: e.caller, callee: e.callee}",
        None,
        None,
        false,
        hades_core::db::query::ExecutionTarget::Writer,
    )
    .await
    .unwrap()
    .results
}

async fn symbol_names(pool: &ArangoPool) -> Vec<Value> {
    hades_core::db::query::query(
        pool,
        "FOR s IN codebase_symbols SORT s.name RETURN s.name",
        None,
        None,
        false,
        hades_core::db::query::ExecutionTarget::Writer,
    )
    .await
    .unwrap()
    .results
}

#[tokio::test]
async fn removed_libclang_call_disappears_only_after_a_successful_reanalysis() {
    const WITH_CALL: &str = "int b() { return 1; }\nint a() { return b(); }\n";
    const WITHOUT_CALL: &str = "int b() { return 1; }\nint a() { return 2; }\n";
    for extension in ["cu", "cpp"] {
        // libclang is dlopened, and CI's isolated job does not install it, so
        // this is an optional analyzer probe like `clang_cuda_probe`.
        let probe = hades_core::code::analyze_with_fallback(
            WITH_CALL,
            hades_core::code::Language::Cpp,
            &format!("probe.{extension}"),
            &hades_core::code::AnalysisOptions::default(),
        );
        match probe {
            hades_core::code::AnalyzerOutcome::Success(analysis) if analysis.analyzer == "libclang" => {}
            _ => {
                eprintln!("skipping .{extension}: libclang cannot analyze the fixture here");
                continue;
            }
        }
        with_temp_db("libclang_owner", Fixtures::Codebase, move |pool| async move {
            let embedder = Embedder::new().await;
            let tree = tempfile::tempdir().unwrap();
            let source = tree.path().join(format!("calls.{extension}"));
            let root = tree.path().to_str().unwrap().to_owned();
            let ingest = |libclang: Option<&Path>| {
                let mut command = cli_command(&pool, &embedder, &["codebase", "ingest", &root]);
                if let Some(empty) = libclang {
                    // LIBCLANG_PATH is searched exclusively, so an empty
                    // directory makes libclang unavailable to the child.
                    command.env("LIBCLANG_PATH", empty);
                }
                command
            };

            std::fs::write(&source, WITH_CALL).unwrap();
            let first = ingest(None).output().await.unwrap();
            assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
            let expected = vec![json!({"caller":"a","callee":"b"})];
            assert_eq!(libclang_calls(&pool).await, expected, ".{extension}");

            // Both symbols remain; only the call is gone.
            std::fs::write(&source, WITHOUT_CALL).unwrap();

            // libclang fails: the file cannot be re-analyzed semantically, so
            // its previous answer stands rather than being erased.
            let empty = tempfile::tempdir().unwrap();
            let failed = ingest(Some(empty.path())).output().await.unwrap();
            let failed_out = String::from_utf8_lossy(&failed.stdout);
            assert_eq!(
                libclang_calls(&pool).await,
                expected,
                ".{extension}: a failed analysis removed the edge\n{failed_out}\n{}",
                String::from_utf8_lossy(&failed.stderr)
            );

            // libclang succeeds and reports no call: the edge is retired.
            let second = ingest(None).output().await.unwrap();
            assert!(second.status.success(), "{}", String::from_utf8_lossy(&second.stderr));
            assert_eq!(libclang_calls(&pool).await, Vec::<Value>::new(), ".{extension}");
            assert_eq!(symbol_names(&pool).await, vec![json!("a"), json!("b")], ".{extension}");

            // And a restored call comes back.
            std::fs::write(&source, WITH_CALL).unwrap();
            let third = ingest(None).output().await.unwrap();
            assert!(third.status.success(), "{}", String::from_utf8_lossy(&third.stderr));
            assert_eq!(libclang_calls(&pool).await, expected, ".{extension}");
        })
        .await;
    }
}
