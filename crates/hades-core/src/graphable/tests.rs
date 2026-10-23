use super::*;
const SAMPLE: &str = include_str!("../../tests/fixtures/graphable/sample.md");

#[test]
fn ingestion_annotations_do_not_extend_definitions_or_their_evidence() {
    for prefix in ["", "  "] {
        let annotation = format!(
            "{prefix}<!-- ingestion: date=2026-10-17 | model=example | window=10000 | late_chunking=on | host=test -->\n"
        );
        let input = SAMPLE.replace("- **glossary**:", &format!("{annotation}- **glossary**:"))
            + &annotation;
        let plan = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
        let baseline = Plan::from_bytes(SAMPLE.as_bytes(), "md", "theory").unwrap();
        for concept in plan.nodes.iter().filter(|n| n["type"] == "concept") {
            let original = baseline
                .nodes
                .iter()
                .find(|n| n["logical_id"] == concept["logical_id"])
                .unwrap();
            assert_eq!(concept["definition"], original["definition"]);
            let location = &concept["source_location"];
            let quote = &input[location["start_byte"].as_u64().unwrap() as usize
                ..location["end_byte"].as_u64().unwrap() as usize];
            assert!(!quote.contains("<!-- ingestion:"));
        }
        for edge in plan.edges.iter().filter(|e| e["relation"] == "defines") {
            let evidence = &edge["evidence"][0];
            let location = &evidence["location"];
            let quote = &input[location["start_byte"].as_u64().unwrap() as usize
                ..location["end_byte"].as_u64().unwrap() as usize];
            assert_eq!(evidence["quote"], quote);
            assert!(!quote.contains("<!-- ingestion:"));
        }
        assert_eq!(plan.nodes[0]["source_text"], input);
    }
}

#[test]
fn many_sections_keep_illustration_content_and_citations_with_their_owner() {
    let extra = (0..1200)
        .map(|i| {
            format!(
                "## Section {i}\n<!-- node: state-notes/section-{i} | keywords: buffer -->\n{}\n",
                "Section content.\n".repeat(16)
            )
        })
        .collect::<String>();
    let base = SAMPLE.replace("## Glossary", &(extra + "## Glossary"));
    // Exercise the no-illustration case that used to scan all entries for
    // every content line, plus many adjacent illustrations in one blockquote.
    for count in [0, 30] {
        let illustrations = (1..=count).map(|i| format!(
            "> **Illustration {i}.** Example {i}\n> <!-- illustration: state-notes/ill-{i} | type: text | subject: buffer/boundary | keywords: buffer -->\n> Illustration-only sentence {i}. See Illustration 1.\n>\n"
        )).collect::<String>();
        let input = base.replace(
            "> **Illustration 1.** A shelf\n> <!-- illustration: state-notes/ill-1 | type: text | subject: buffer/boundary | keywords: buffer -->\n> A shelf holds books.", &illustrations
        ).replace("The buffer holds entries. See Illustration 1.", "The buffer holds entries.");
        let plan = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
        assert_eq!(plan.nodes.len(), 1208 + count);
        for node in plan.nodes.iter().filter(|n| n["type"] == "section") {
            assert!(
                !node["content"]
                    .as_str()
                    .unwrap()
                    .contains("Illustration-only")
            );
        }
        let citations: Vec<_> = plan
            .edges
            .iter()
            .filter(|e| e["relation"] == "cites_illustration")
            .collect();
        assert_eq!(citations.len(), count);
        for i in 1..=count {
            let id = format!("state-notes/ill-{i}");
            let node = plan.nodes.iter().find(|n| n["logical_id"] == id).unwrap();
            assert_eq!(
                node["content"]
                    .as_str()
                    .unwrap()
                    .matches("Illustration-only")
                    .count(),
                1
            );
            let handle = format!("theory_nodes/{}", node["_key"].as_str().unwrap());
            let citation = citations.iter().find(|e| e["_from"] == handle).unwrap();
            let evidence = &citation["evidence"][0];
            let loc = &evidence["location"];
            assert_eq!(
                evidence["quote"],
                &input[loc["start_byte"].as_u64().unwrap() as usize
                    ..loc["end_byte"].as_u64().unwrap() as usize]
            );
        }
    }
}

#[test]
fn mechanical_structure_preserves_claims_as_text_and_evidence_as_spans() {
    let plan = Plan::from_bytes(SAMPLE.as_bytes(), "md", "theory").unwrap();
    assert_eq!(plan.nodes.len(), 9);
    assert_eq!(plan.version, "0.10");
    let second = Plan::from_bytes(SAMPLE.as_bytes(), "md", "theory").unwrap();
    assert_eq!(plan.nodes, second.nodes);
    assert_eq!(plan.edges, second.edges);
    assert!(plan.nodes.iter().all(|n| n["logical_id"] != "fake"));
    assert!(plan.nodes.iter().any(|n| {
        n["standing_text"]
            .as_str()
            .is_some_and(|s| s.contains("candidate"))
    }));
    assert_eq!(plan.nodes[0]["source_text"], SAMPLE);
    let ids: std::collections::BTreeSet<_> = plan
        .nodes
        .iter()
        .map(|n| format!("theory_nodes/{}", n["_key"].as_str().unwrap()))
        .collect();
    for edge in &plan.edges {
        assert!(ids.contains(edge["_from"].as_str().unwrap()));
        assert!(ids.contains(edge["_to"].as_str().unwrap()));
        assert!(matches!(
            edge["relation"].as_str().unwrap(),
            "contains"
                | "defines"
                | "uses_keyword"
                | "illustrates"
                | "cites_illustration"
                | "references"
        ));
        for evidence in edge["evidence"].as_array().unwrap() {
            let loc = &evidence["location"];
            assert_eq!(
                &SAMPLE[loc["start_byte"].as_u64().unwrap() as usize
                    ..loc["end_byte"].as_u64().unwrap() as usize],
                evidence["quote"].as_str().unwrap()
            );
        }
    }
    let archive = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        plan.manifest["source_base64"].as_str().unwrap(),
    )
    .unwrap();
    assert_eq!(archive, SAMPLE.as_bytes());
}

#[test]
fn missing_headers_duplicate_ids_and_ambiguous_subjects_fail_closed() {
    for bad in [
        SAMPLE.replace("<!-- node: state-notes/buffer | keywords: buffer, unresolved | xref: other/document -->", ""),
        SAMPLE.replace("node: state-notes/buffer/boundary", "node: state-notes/buffer"),
        SAMPLE.replace("subject: buffer/boundary", "subject: missing"),
        SAMPLE.replace("### A boundary", "## A boundary\n### Orphan"),
        SAMPLE.replace("## Glossary", "## Terms"),
    ] {assert!(Plan::from_bytes(bad.as_bytes(),"md","theory").is_err());}
}

#[test]
fn version_keys_change_without_changing_logical_identity() {
    let a = Plan::from_bytes(SAMPLE.as_bytes(), "md", "theory").unwrap();
    let b = Plan::from_bytes(
        SAMPLE.replace("'0.10'", "'0.11'").as_bytes(),
        "md",
        "theory",
    )
    .unwrap();
    for (a, b) in a.nodes.iter().zip(b.nodes.iter()) {
        assert_eq!(a["logical_id"], b["logical_id"]);
        assert_ne!(a["_key"], b["_key"]);
    }
    assert_ne!(key("n", &["a|b", "c"]), key("n", &["a", "b|c"]));
}

#[test]
fn front_matter_preserves_unquoted_version_spelling() {
    let p = Plan::from_bytes(SAMPLE.replace("'0.10'", "0.10").as_bytes(), "md", "theory").unwrap();
    assert_eq!(p.version, "0.10");
}

#[test]
fn yaml_comments_are_metadata_not_document_headings() {
    let input = SAMPLE.replace(
        "title: State Notes",
        "# Reviewed source\ntitle: State Notes",
    );
    let plan = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
    assert_eq!(plan.nodes.len(), 9);
    assert_eq!(plan.nodes[0]["source_text"], input);
    assert_eq!(plan.nodes[0]["logical_id"], "state-notes");
}

#[test]
fn adjacent_unbulleted_definitions_keep_separate_concepts_and_evidence() {
    for bold in [true, false] {
        let mut input = SAMPLE.replace("- **", "**");
        if !bold {
            input = input.replace("**", "");
        }
        let plan = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
        for (term, definition) in [
            (
                "buffer",
                "The entries retained between calls.\n  Rejected synonyms: cache.",
            ),
            ("glossary", "The definitions used by this document."),
        ] {
            let node = plan
                .nodes
                .iter()
                .find(|n| n["type"] == "concept" && n["term"] == term)
                .unwrap();
            assert_eq!(node["definition"], definition);
            let id = format!("theory_nodes/{}", node["_key"].as_str().unwrap());
            let edge = plan
                .edges
                .iter()
                .find(|e| e["relation"] == "defines" && e["_to"] == id)
                .unwrap();
            let evidence = &edge["evidence"][0];
            let location = &evidence["location"];
            let quote = &input[location["start_byte"].as_u64().unwrap() as usize
                ..location["end_byte"].as_u64().unwrap() as usize];
            assert_eq!(evidence["quote"], quote);
            assert!(quote.contains(definition));
            assert!(!quote.contains(if term == "buffer" {
                "definitions used"
            } else {
                "entries retained"
            }));
        }
        assert!(!plan.nodes.iter().any(|n| n["term"] == "Rejected synonyms"));
    }
}

#[test]
fn indented_fences_keep_example_declarations_literal() {
    for marker in ["```", "~~~"] {
        for open_indent in 0..=3 {
            for close_indent in 0..=3 {
                let input = SAMPLE.replace("```markdown", &format!("{}{marker}markdown", " ".repeat(open_indent)))
                    .replace("\n```\n", &format!("\n    {marker}\n<!-- node: still-literal | keywords: fake -->\n{}{marker}\n", " ".repeat(close_indent)));
                let plan = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
                assert_eq!(plan.nodes.len(), 9);
                assert!(
                    !plan
                        .nodes
                        .iter()
                        .any(|n| n["logical_id"] == "fake" || n["logical_id"] == "still-literal")
                );
                assert!(
                    plan.nodes
                        .iter()
                        .any(|n| n["logical_id"] == "state-notes/ill-1")
                );
            }
        }
    }
}

#[test]
fn adjacent_illustrations_have_disjoint_content_and_resolvable_citations() {
    let input=SAMPLE.replace("> A shelf holds books.",
        "> A shelf holds books.\n>\n> **Illustration 2.** Another shelf\n> <!-- illustration: state-notes/ill-2 | type: text | subject: buffer/boundary | keywords: buffer -->\n> Refer to Illustration 1.");
    let p = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
    let first = p
        .nodes
        .iter()
        .find(|n| n["logical_id"] == "state-notes/ill-1")
        .unwrap();
    let second = p
        .nodes
        .iter()
        .find(|n| n["logical_id"] == "state-notes/ill-2")
        .unwrap();
    assert!(!first["content"].as_str().unwrap().contains("Another shelf"));
    assert!(p.edges.iter().any(|e| e["relation"] == "cites_illustration"
        && e["_from"] == format!("theory_nodes/{}", second["_key"].as_str().unwrap())
        && e["_to"] == format!("theory_nodes/{}", first["_key"].as_str().unwrap())));
}

#[test]
fn indented_headings_keep_titles_hierarchy_and_source_spans() {
    for indent in 1..=3 {
        let input = SAMPLE
            .lines()
            .map(|line| {
                if line.starts_with('#') {
                    format!("{}{line}\n", " ".repeat(indent))
                } else {
                    format!("{line}\n")
                }
            })
            .collect::<String>();
        let plan = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
        assert_eq!(plan.nodes.len(), 9);
        for (id, title) in [
            ("state-notes", "State Notes"),
            ("state-notes/buffer", "The buffer"),
            ("state-notes/buffer/boundary", "A boundary"),
        ] {
            let node = plan.nodes.iter().find(|n| n["logical_id"] == id).unwrap();
            assert_eq!(node["title"], title);
        }
        for edge in &plan.edges {
            for evidence in edge["evidence"].as_array().unwrap() {
                let loc = &evidence["location"];
                assert_eq!(
                    evidence["quote"],
                    &input[loc["start_byte"].as_u64().unwrap() as usize
                        ..loc["end_byte"].as_u64().unwrap() as usize]
                );
            }
        }
    }
    let code = SAMPLE.replace("This document defines a buffer.",
        "This document defines a buffer.\n\n    ## Example heading\n    <!-- node: fake | keywords: fake -->");
    assert_eq!(
        Plan::from_bytes(code.as_bytes(), "md", "theory")
            .unwrap()
            .nodes
            .len(),
        9
    );
}

#[test]
fn quoted_indented_code_does_not_create_illustrations_or_citations() {
    // After '> ', one tab reaches column four but leaves only two content
    // indentation columns. Two tabs reach the indented-code threshold.
    for indent in ["    ", "\t\t", " \t\t"] {
        let example = format!(
            ">\n> {indent}**Illustration 2.** A code example\n> {indent}<!-- illustration: fake | type: text | subject: missing | keywords: fake -->\n> {indent}See Illustration 99.\n>\n> This prose is part of the illustration."
        );
        let input = SAMPLE.replace(
            "> A shelf holds books.",
            &format!("> A shelf holds books.\n{example}"),
        );
        let plan = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
        assert_eq!(plan.nodes.len(), 9);
        assert!(!plan.warnings.iter().any(|w| w.contains("Illustration 99")));
        let illustration = plan
            .nodes
            .iter()
            .find(|n| n["logical_id"] == "state-notes/ill-1")
            .unwrap();
        assert!(
            illustration["content"]
                .as_str()
                .unwrap()
                .contains("See Illustration 99.")
        );
    }
}

#[test]
fn tabs_after_heading_markers_preserve_titles() {
    for indent in 0..=3 {
        let input = SAMPLE
            .lines()
            .map(|line| {
                if line.starts_with('#') {
                    format!("{}{}\n", " ".repeat(indent), line.replacen(' ', "\t", 1))
                } else {
                    format!("{line}\n")
                }
            })
            .collect::<String>();
        let plan = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
        assert_eq!(plan.nodes.len(), 9);
        for (id, title) in [
            ("state-notes", "State Notes"),
            ("state-notes/buffer", "The buffer"),
            ("state-notes/buffer/boundary", "A boundary"),
        ] {
            assert_eq!(
                plan.nodes.iter().find(|n| n["logical_id"] == id).unwrap()["title"],
                title
            );
        }
        assert_eq!(plan.nodes[0]["source_text"], input);
    }
}

#[test]
fn mixed_indentation_keeps_code_examples_out_of_the_graph() {
    for prefix in ["\t", " \t", "  \t", "   \t", "    ", "\t "] {
        let example = format!(
            "\n{prefix}> **Illustration 2.** Code\n{prefix}> <!-- illustration: fake | type: text | subject: missing | keywords: fake -->\n{prefix}## Fake heading\n{prefix}<!-- node: fake | keywords: fake -->\n{prefix}See Illustration 99.\n"
        );
        let input = SAMPLE.replace(
            "This document defines a buffer.",
            &format!("This document defines a buffer.\n{example}"),
        );
        let plan = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
        assert_eq!(plan.nodes.len(), 9);
        assert!(!plan.warnings.iter().any(|w| w.contains("Illustration 99")));
        assert_eq!(plan.nodes[0]["source_text"], input);
    }
}

#[test]
fn quote_markers_inside_outer_fences_are_literal_content() {
    for marker in ["```", "~~~"] {
        for prefix in ["> ", " > ", "  > ", "   > ", ">\t"] {
            let example = format!(
                "{marker}markdown\n{prefix}{marker}\n## Not a node\n<!-- node: fake | keywords: fake -->\n{prefix}**Illustration 99.** Code\n{prefix}<!-- illustration: fake | type: text | subject: missing | keywords: fake -->\n{marker}"
            );
            let input = SAMPLE.replace(
                "```markdown\n## Not a node\n<!-- node: fake | keywords: fake -->\n```",
                &example,
            );
            let plan = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
            assert_eq!(plan.nodes.len(), 9);
            assert!(!plan.warnings.iter().any(|w| w.contains("Illustration 99")));
            assert_eq!(plan.nodes[0]["source_text"], input);
        }
    }
}

#[test]
fn quoted_fences_end_at_their_container_boundary() {
    for closing in ["> ```\n", ""] {
        let example = format!(
            "> ```markdown\n> ## Not a node\n> <!-- node: fake | keywords: fake -->\n> See Illustration 99.\n{closing}"
        );
        let input = SAMPLE.replace(
            "```markdown\n## Not a node\n<!-- node: fake | keywords: fake -->\n```",
            &example,
        );
        let plan = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
        assert_eq!(plan.nodes.len(), 9);
        assert!(!plan.warnings.iter().any(|w| w.contains("Illustration 99")));
        assert!(
            plan.nodes
                .iter()
                .any(|n| n["logical_id"] == "state-notes/glossary")
        );
    }
    // A tab immediately after '>' can leave fewer than four indentation
    // columns. Such a declaration is still a real illustration.
    let input = SAMPLE.replace("> ", ">\t");
    assert_eq!(
        Plan::from_bytes(input.as_bytes(), "md", "theory")
            .unwrap()
            .nodes
            .len(),
        9
    );
}

#[test]
fn original_archive_is_separate_from_the_parsed_markdown_and_is_lossless() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("original.docx");
    let bytes = b"original file bytes\0\xff";
    std::fs::write(&file, bytes).unwrap();
    let plan = Plan::from_bytes(SAMPLE.as_bytes(), "md", "theory")
        .unwrap()
        .with_original_source(&file, "Reviewed text export")
        .unwrap();
    assert_eq!(plan.nodes[0]["source_text"], SAMPLE);
    assert_eq!(
        plan.nodes[0]["original_source"]["original_source_sha256"],
        hash(bytes)
    );
    use base64::Engine;
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(plan.manifest["original_source_base64"].as_str().unwrap())
            .unwrap(),
        bytes
    );
}

/// Cancel after an acknowledged write while the importer is held before commit.
#[tokio::test]
async fn cancelling_an_import_after_node_writes_publishes_nothing_and_retry_succeeds() {
    use crate::test_support::{Fixtures, with_temp_db};
    with_temp_db("graphable_cancel", Fixtures::Empty, |pool| async move {
        let plan = |source: &str| Plan::from_bytes(source.as_bytes(), "md", "theory").unwrap();
        let base = plan(SAMPLE);
        super::import(&pool, &base, true).await.unwrap();
        let updated = plan(&SAMPLE.replace("'0.10'", "'0.11'"));
        let cloned = pool.clone();
        let task_plan = updated.clone();
        let (written, wait_written) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            super::store::import_until_node_write(&cloned, &task_plan, written).await
        });
        // Inspect this import's real uncommitted node while its callback is
        // paused. Polling transaction inventory allowed commit to race abort().
        let scoped = tokio::time::timeout(std::time::Duration::from_secs(10), wait_written)
            .await
            .unwrap()
            .unwrap();
        let root_path = format!(
            "document/theory_nodes/{}",
            updated.nodes()[0]["_key"].as_str().unwrap()
        );
        assert_eq!(
            scoped.get(&root_path).await.unwrap()["source_version"],
            "0.11"
        );
        assert!(!task.is_finished());
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        // Taking the exclusive lock again waits for the independent abort owner.
        crate::db::transaction::run(
            &pool,
            vec!["theory_nodes".into()],
            move |client| async move {
                assert!(client.get(&root_path).await.unwrap_err().is_not_found());
                Ok(())
            },
        )
        .await
        .unwrap();
        assert_eq!(
            stored_count(&pool, "theory_nodes").await,
            base.nodes().len() as u64
        );
        assert_eq!(
            stored_count(&pool, "theory_edges").await,
            base.edges().len() as u64
        );
        assert_eq!(stored_count(&pool, "theory_imports").await, 1);
        let recovered = super::import(&pool, &updated, false).await.unwrap();
        assert_eq!(recovered["inserted_nodes"], updated.nodes().len());
        assert_eq!(recovered["inserted_edges"], updated.edges().len());
        assert_eq!(
            super::import(&pool, &updated, false).await.unwrap()["unchanged"],
            true
        );
    })
    .await;
}

async fn stored_count(pool: &crate::db::ArangoPool, collection: &str) -> u64 {
    pool.reader()
        .get(&format!("collection/{collection}/count"))
        .await
        .unwrap()["count"]
        .as_u64()
        .unwrap()
}

#[test]
fn multiline_front_keywords_and_definitions_keep_complete_evidence() {
    let input = SAMPLE
        .replace(
            "keywords: [buffer, glossary]",
            "keywords:\n  - buffer\n  - glossary",
        )
        .replace("version: '0.10'", "version: 0.10 # reviewed");
    let p = Plan::from_bytes(input.as_bytes(), "md", "theory").unwrap();
    assert_eq!(p.version, "0.10");
    let definition = p
        .edges
        .iter()
        .find(|e| {
            e["relation"] == "defines"
                && e["evidence"][0]["quote"]
                    .as_str()
                    .unwrap()
                    .contains("buffer")
        })
        .unwrap();
    assert!(
        definition["evidence"][0]["quote"]
            .as_str()
            .unwrap()
            .contains("Rejected synonyms")
    );
    let root = format!("theory_nodes/{}", p.nodes[0]["_key"].as_str().unwrap());
    let glossary = p
        .nodes
        .iter()
        .find(|n| n["type"] == "concept" && n["term"] == "glossary")
        .unwrap();
    let target = format!("theory_nodes/{}", glossary["_key"].as_str().unwrap());
    let edge = p
        .edges
        .iter()
        .find(|e| e["_from"] == root && e["_to"] == target)
        .unwrap();
    assert!(
        edge["evidence"][0]["quote"]
            .as_str()
            .unwrap()
            .contains("  - glossary")
    );
}
