//! The saved ontology controls endpoints and is pinned through publication.
use super::*;

async fn definition(pool: &ArangoPool, name: &str) -> Value {
    query::query(pool, "FOR d IN hades_schema FILTER d.schema_type == 'edge_definition' AND d.name == @name RETURN d",
        Some(&json!({"name":name})), None, false, query::ExecutionTarget::Writer).await.unwrap().results.remove(0)
}

#[tokio::test]
async fn endpoint_kinds_follow_saved_definitions_before_planning_and_at_publication() {
    with_temp_db("code_ontology", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        for (kind, from, to) in [
            (
                RelationshipKind::Calls,
                LocalEndpoint::File,
                LocalEndpoint::Symbol(0),
            ),
            (
                RelationshipKind::Calls,
                LocalEndpoint::Symbol(0),
                LocalEndpoint::File,
            ),
            (
                RelationshipKind::Implements,
                LocalEndpoint::File,
                LocalEndpoint::File,
            ),
            (
                RelationshipKind::Imports,
                LocalEndpoint::Symbol(0),
                LocalEndpoint::File,
            ),
        ] {
            let mut source = input("invalid.rs", "");
            source.relationships = vec![LocalRelationship { kind, from, to }];
            assert!(
                code.prepare(source, false)
                    .await
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("ontology")
            );
        }
        assert!(peer.plans().is_empty());
        assert!(peer.requests().is_empty());
        let mut source = input("valid.rs", "");
        source.relationships.extend([
            LocalRelationship {
                kind: RelationshipKind::Implements,
                from: LocalEndpoint::Symbol(1),
                to: LocalEndpoint::Symbol(0),
            },
            LocalRelationship {
                kind: RelationshipKind::Imports,
                from: LocalEndpoint::File,
                to: LocalEndpoint::File,
            },
            LocalRelationship {
                kind: RelationshipKind::Imports,
                from: LocalEndpoint::File,
                to: LocalEndpoint::Symbol(0),
            },
        ]);
        assert_eq!(
            code.commit(code.prepare(source, false).await.unwrap())
                .await
                .unwrap()
                .relationships,
            6
        );
        let saved = rows(&pool).await;
        let prepared = code.prepare(input("pending.rs", ""), false).await.unwrap();
        let imports = definition(&pool, CODEBASE.imports_edges).await;
        let path = format!(
            "document/hades_schema/{}",
            imports["_key"].as_str().unwrap()
        );
        pool.writer()
            .patch(&path, &json!({"from_collections":[CODEBASE.symbols]}))
            .await
            .unwrap();
        assert!(
            code.commit(prepared)
                .await
                .unwrap_err()
                .to_string()
                .contains("ontology changed")
        );
        assert_eq!(rows(&pool).await, saved);
        let plans = peer.plans().len();
        assert!(code.prepare(input("stale.rs", ""), false).await.is_err());
        assert_eq!(peer.plans().len(), plans);
        // A fresh admission uses the database's changed definition, not a Rust table.
        let code = connect(&pool, &peer).await;
        let mut source = input("custom.rs", "");
        source.relationships = vec![LocalRelationship {
            kind: RelationshipKind::Imports,
            from: LocalEndpoint::Symbol(1),
            to: LocalEndpoint::Symbol(0),
        }];
        code.commit(code.prepare(source, false).await.unwrap())
            .await
            .unwrap();

        // Generated defines edges obey the same data-owned endpoint contract.
        let defines = definition(&pool, CODEBASE.defines_edges).await;
        pool.writer()
            .patch(
                &format!(
                    "document/hades_schema/{}",
                    defines["_key"].as_str().unwrap()
                ),
                &json!({"from_collections":[CODEBASE.symbols]}),
            )
            .await
            .unwrap();
        let code = connect(&pool, &peer).await;
        let plans = peer.plans().len();
        assert!(
            code.prepare(input("bad_defines.rs", ""), false)
                .await
                .is_err()
        );
        assert_eq!(peer.plans().len(), plans);
    })
    .await;
}

#[tokio::test]
async fn missing_or_ambiguous_ontology_refuses_provider_admission() {
    with_temp_db(
        "code_ontology_missing",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::code(2);
            fixture.install_code(&pool).await;
            let peer = Peer::with_workload_description(&fixture);
            let mut saved = definition(&pool, CODEBASE.calls_edges).await;
            let path = format!("document/hades_schema/{}", saved["_key"].as_str().unwrap());
            saved.as_object_mut().unwrap().remove("_rev");
            saved.as_object_mut().unwrap().remove("_id");
            pool.writer().delete(&path).await.unwrap();
            assert!(
                ProfiledCode::connect(pool.clone(), &peer.catalog())
                    .await
                    .is_err()
            );
            pool.writer()
                .post("document/hades_schema", &saved)
                .await
                .unwrap();
            saved["_key"] = json!("ambiguous_calls");
            pool.writer()
                .post("document/hades_schema", &saved)
                .await
                .unwrap();
            assert!(
                ProfiledCode::connect(pool.clone(), &peer.catalog())
                    .await
                    .is_err()
            );
            assert_eq!(peer.observations(), 0);
            assert!(peer.plans().is_empty());
        },
    )
    .await;
}

#[tokio::test]
async fn prepared_size_is_bounded_before_planning_and_before_embedding() {
    with_temp_db("code_size_admission", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(8192);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        let saved = rows(&pool).await;
        let mut source = input("large.rs", "");
        source.text = format!("/*{}*/\n", "\u{0001}".repeat(1_500_000));
        source.analysis = code::analyze(&source.text, "large.rs").unwrap();
        source.relationships.clear();
        assert!(
            code.prepare(source, false)
                .await
                .err()
                .unwrap()
                .to_string()
                .contains("before planning")
        );
        assert!(peer.plans().is_empty());
        let mut source = input("wide.rs", "");
        source.text = "// many spans\n".repeat(80);
        source.analysis = code::analyze(&source.text, "wide.rs").unwrap();
        source.relationships.clear();
        assert!(
            code.prepare(source, false)
                .await
                .err()
                .unwrap()
                .to_string()
                .contains("before embedding")
        );
        assert_eq!(peer.plans().len(), 1);
        assert!(peer.requests().is_empty());
        assert_eq!(rows(&pool).await, saved);
    })
    .await;
}
