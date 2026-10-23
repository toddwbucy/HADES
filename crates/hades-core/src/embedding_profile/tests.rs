use super::*;
use serde_json::{Value, json};

/// Load the synthetic contract shared with the Python identity suite.
fn fixture() -> (ArtifactManifest, EmbeddingProfile, ProviderReceipt) {
    let value: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/embedding_profile.json")).unwrap();
    (
        serde_json::from_value(value["manifest"].clone()).unwrap(),
        serde_json::from_value(value["profile"].clone()).unwrap(),
        serde_json::from_value(value["receipt"].clone()).unwrap(),
    )
}

/// Decode the fixed byte vectors so tests compare serialization independently.
fn unhex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

/// Construct an asset through the streaming API from synthetic source bytes.
fn asset(path: &str, bytes: &[u8]) -> ArtifactFile {
    ArtifactFile::from_reader(path.into(), bytes).unwrap()
}

#[test]
/// Require the Rust encoder to reproduce every approved cross-language vector.
fn shared_manifest_vectors_match_exact_bytes_and_md5() {
    let vectors: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/specs/database-embedding-manifest-vectors.json"
    )))
    .unwrap();
    for case in vectors["cases"].as_array().unwrap() {
        let files: Vec<_> = case["source_files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|file| {
                asset(
                    file["path"].as_str().unwrap(),
                    &unhex(file["content_hex"].as_str().unwrap()),
                )
            })
            .collect();
        let manifest = ArtifactManifest::new(files.clone()).unwrap();
        let bytes = manifest.canonical_bytes().unwrap();
        assert_eq!(bytes, case["canonical_json"].as_str().unwrap().as_bytes());
        assert_eq!(bytes, unhex(case["canonical_utf8_hex"].as_str().unwrap()));
        assert_eq!(
            manifest.digest().unwrap().as_str(),
            case["manifest_md5"].as_str().unwrap()
        );
        assert_eq!(
            ArtifactManifest::from_json(&bytes)
                .unwrap()
                .digest()
                .unwrap(),
            manifest.digest().unwrap()
        );
        assert_eq!(
            ArtifactManifest::new(files.into_iter().rev().collect())
                .unwrap()
                .digest()
                .unwrap(),
            manifest.digest().unwrap()
        );
    }
}

#[test]
/// Keep Unicode scalar sequences distinct and apply the specified JSON escaping.
fn manifest_escaping_preserves_controls_unicode_and_literal_backslash() {
    let path = "x/\u{1}\u{8}\t\n\u{c}\r\"\\\u{2028}\u{2029}";
    let bytes = ArtifactManifest::new(vec![asset(path, b"")])
        .unwrap()
        .canonical_bytes()
        .unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert!(text.contains("x/\\u0001\\b\\t\\n\\f\\r\\\"\\\\\u{2028}\u{2029}"));
    assert_ne!(
        ArtifactManifest::new(vec![asset("caf\u{e9}", b"")])
            .unwrap()
            .digest()
            .unwrap(),
        ArtifactManifest::new(vec![asset("cafe\u{301}", b"")])
            .unwrap()
            .digest()
            .unwrap()
    );
}

#[test]
/// Reject identities whose paths or digest encodings could be misinterpreted.
fn manifest_rejects_ambiguous_paths_missing_assets_and_bad_digests() {
    for path in [
        "",
        "/absolute",
        "./model",
        "model/../weights",
        "model//weights",
        "model/",
        "nul\0x",
    ] {
        assert!(
            ArtifactFile::from_reader(path.into(), &b""[..]).is_err(),
            "{path:?}"
        );
    }
    assert!(ArtifactManifest::new(vec![]).is_err());
    for paths in [["a", "a"], ["a", "a/b"], ["a/b", "a"]] {
        assert!(ArtifactManifest::new(paths.into_iter().map(|p| asset(p, b"")).collect()).is_err());
    }
    for digest in [
        "",
        "D41D8CD98F00B204E9800998ECF8427E",
        "g41d8cd98f00b204e9800998ecf8427e",
    ] {
        assert!(Md5Digest::try_from(digest.to_owned()).is_err());
    }
    let (manifest, _, _) = fixture();
    assert!(
        manifest
            .check_digest(&Md5Digest::of_bytes(b"wrong expected identity"))
            .is_err()
    );
}

#[test]
/// Refuse unknown schemas, invalid numeric fields, malformed Unicode, and size excesses.
fn manifest_parsing_is_strict_and_bounded() {
    let (manifest, _, _) = fixture();
    let original: Value = serde_json::to_value(manifest).unwrap();
    for (field, bad) in [
        ("format", json!("hades-model-manifest-v2")),
        ("unexpected", json!(true)),
    ] {
        let mut value = original.clone();
        value[field] = bad;
        assert!(ArtifactManifest::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
    }
    for bad in [json!(-1), json!(1.5), json!(true), json!("3"), Value::Null] {
        let mut value = original.clone();
        value["files"][0]["bytes"] = bad;
        assert!(ArtifactManifest::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
    }
    let mut value = original;
    value["files"][0]["extra"] = json!(1);
    assert!(ArtifactManifest::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
    for raw in [br#"{"format":"hades-model-manifest-v1","format":"hades-model-manifest-v1","files":[]}"#.as_slice(),
        br#"{"format":"hades-model-manifest-v1","files":[{"path":"\ud800","bytes":0,"md5":"d41d8cd98f00b204e9800998ecf8427e"}]}"#,
        b"\xff"] {
        assert!(ArtifactManifest::from_json(raw).is_err());
    }
    assert!(ArtifactManifest::from_json(&vec![b' '; MAX_MANIFEST_BYTES + 1]).is_err());
    assert!(
        ArtifactManifest::new(
            (0..=MAX_MANIFEST_FILES)
                .map(|i| asset(&format!("a{i}"), b""))
                .collect()
        )
        .is_err()
    );
    assert!(ArtifactFile::from_reader("a".repeat(4097), &b""[..]).is_err());
}

#[test]
/// Cover multiple read buffers and ensure read failures cannot yield partial digests.
fn stream_hashes_large_files_and_propagates_read_failure() {
    let bytes = vec![17; 150_000];
    let file = asset("weights.bin", &bytes);
    assert_eq!(file.bytes, bytes.len() as u64);
    assert_eq!(file.md5, Md5Digest::of_bytes(&bytes));
    struct Broken;
    impl std::io::Read for Broken {
        /// Simulate an asset read failure before a checksum can be completed.
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("fixture"))
        }
    }
    assert!(ArtifactFile::from_reader("weights.bin".into(), Broken).is_err());
}

#[test]
/// Distinguish host-local relocation from changes to any required model dependency.
fn artifact_relocation_is_stable_but_every_required_asset_changes_identity() {
    let roots = [tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap()];
    let paths = [
        "weights.bin",
        "tokenizer.json",
        "config.json",
        "adapter.bin",
    ];
    let mut digests = Vec::new();
    for root in &roots {
        let files = paths
            .iter()
            .map(|path| {
                std::fs::write(root.path().join(path), b"original").unwrap();
                ArtifactFile::from_reader(
                    (*path).into(),
                    std::fs::File::open(root.path().join(path)).unwrap(),
                )
                .unwrap()
            })
            .collect();
        digests.push(ArtifactManifest::new(files).unwrap().digest().unwrap());
    }
    assert_eq!(digests[0], digests[1]);
    for changed in paths {
        let files = paths
            .iter()
            .map(|p| {
                asset(
                    p,
                    if *p == changed {
                        b"modified"
                    } else {
                        b"original"
                    },
                )
            })
            .collect();
        assert_ne!(
            digests[0],
            ArtifactManifest::new(files).unwrap().digest().unwrap()
        );
    }
}

#[test]
/// Resolve the small-model fixture while retaining its asymmetric task pair.
fn fixture_profile_resolves_small_model_and_asymmetric_tasks_without_defaults() {
    let (manifest, profile, receipt) = fixture();
    let parsed =
        EmbeddingProfile::from_json(&serde_json::to_vec(&profile).unwrap(), &manifest).unwrap();
    let passage = parsed
        .bind(
            &manifest,
            WorkloadKind::Documents,
            EmbeddingRole::Passage,
            &receipt,
        )
        .unwrap();
    let query = parsed
        .bind(
            &manifest,
            WorkloadKind::Documents,
            EmbeddingRole::Query,
            &receipt,
        )
        .unwrap();
    assert_eq!(passage.dimension, 384);
    assert_eq!(query.model, "fixture/small-text-model");
    assert_eq!(passage.task, "retrieval.passage");
    assert_eq!(query.task, "retrieval.query");
    assert_eq!(passage.expected.space_id, query.expected.space_id);
    let value: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/embedding_profile.json")).unwrap();
    assert_eq!(
        crate::canonical_json::to_vec(&value["profile"]["workloads"]["documents"]["space"]),
        unhex(value["space_canonical_utf8_hex"].as_str().unwrap())
    );
}

#[test]
/// Refuse changed artifacts or inference behavior even when vector widths agree.
fn same_dimension_is_not_enough_for_artifact_or_behavior_compatibility() {
    let (manifest, profile, receipt) = fixture();
    let original = profile.workloads.documents.as_ref().unwrap().space.clone();
    let mut spaces = Vec::new();
    let mut space = original.clone();
    space.artifact_md5 = Md5Digest::of_bytes(b"changed weights");
    spaces.push(space);
    let mut space = original.clone();
    space.query.prompt.push(' ');
    spaces.push(space);
    let mut space = original.clone();
    space.passage.adapter = "code".into();
    spaces.push(space);
    let mut space = original.clone();
    space.representation.normalize = Normalization::None;
    spaces.push(space);
    let mut space = original.clone();
    space.representation.pooling = "cls".into();
    spaces.push(space);
    let mut space = original.clone();
    space.representation.inference_precision = "fp16".into();
    spaces.push(space);
    let mut space = original.clone();
    space.representation.implementation = "fixture-adapter-v2".into();
    spaces.push(space);
    let mut space = original.clone();
    space.tokenizer_contract = "tokenizer-v2".into();
    spaces.push(space);
    let mut space = original.clone();
    space.passage.special_tokens = "no-special-tokens".into();
    spaces.push(space);
    for space in spaces {
        let mut wrong = receipt.clone();
        wrong.space_id = space.id().unwrap();
        assert_ne!(wrong.space_id, receipt.space_id);
        assert!(
            profile
                .bind(
                    &manifest,
                    WorkloadKind::Documents,
                    EmbeddingRole::Query,
                    &wrong
                )
                .is_err()
        );
    }
    let mut wrong = receipt.clone();
    wrong.artifact_md5 = Md5Digest::of_bytes(b"changed weights");
    assert!(
        profile
            .bind(
                &manifest,
                WorkloadKind::Documents,
                EmbeddingRole::Query,
                &wrong
            )
            .is_err()
    );
}

#[test]
/// Prevent accepting a response under an identity pinned before a provider reload.
fn receipt_rejects_reload_or_identity_changes_after_binding() {
    let (manifest, profile, receipt) = fixture();
    let request = profile
        .bind(
            &manifest,
            WorkloadKind::Documents,
            EmbeddingRole::Passage,
            &receipt,
        )
        .unwrap();
    receipt.check_matches(&request.expected).unwrap();
    let mut changed = receipt.clone();
    changed.serving_generation = "load-fixture-2".into();
    assert!(changed.check_matches(&request.expected).is_err());
    let mut changed = receipt.clone();
    changed.artifact_md5 = Md5Digest::of_bytes(b"other");
    assert!(changed.check_matches(&request.expected).is_err());
    let mut changed = receipt.clone();
    changed.space_id = SpaceId::try_from("0".repeat(64)).unwrap();
    assert!(changed.check_matches(&request.expected).is_err());
    let mut missing = receipt.clone();
    missing.serving_generation.clear();
    assert!(
        profile
            .bind(
                &manifest,
                WorkloadKind::Documents,
                EmbeddingRole::Query,
                &missing
            )
            .is_err()
    );
}

#[test]
/// Separate enabled workloads from advertised capabilities and workspace readiness.
fn disabled_workloads_unready_profiles_and_unsupported_features_fail() {
    let (manifest, profile, receipt) = fixture();
    assert!(
        profile
            .bind(
                &manifest,
                WorkloadKind::Code,
                EmbeddingRole::Passage,
                &receipt
            )
            .is_err()
    );
    let mut unready = profile.clone();
    unready.state = ProfileState::Initializing;
    unready.validate(&manifest).unwrap();
    assert!(
        unready
            .bind(
                &manifest,
                WorkloadKind::Documents,
                EmbeddingRole::Query,
                &receipt
            )
            .is_err()
    );
    let mut late = profile.clone();
    late.workloads.documents.as_mut().unwrap().chunking = ChunkPolicy::Late {
        context_cap_tokens: 2048,
        minimum_window_overlap_tokens: 100,
        passages: None,
    };
    assert!(late.validate(&manifest).is_err());
    late.capabilities.late_chunking = true;
    assert!(late.validate(&manifest).is_err());
    late.capabilities.tokenizer_offsets = true;
    late.validate(&manifest).unwrap();
    let mut code = profile.clone();
    code.workloads.code = code.workloads.documents.clone();
    // Ordinary text code ingestion does not require a specialized code claim.
    code.validate(&manifest).unwrap();
    code.workloads.code.as_mut().unwrap().specialized_code = true;
    assert!(code.validate(&manifest).is_err());
    code.capabilities.specialized_code = true;
    code.validate(&manifest).unwrap();
}

#[test]
fn late_passage_policy_is_explicit_bounded_and_preserves_older_wire_records() {
    let (manifest, mut profile, _) = fixture();
    profile.capabilities.late_chunking = true;
    profile.capabilities.tokenizer_offsets = true;
    let old = json!({"mode":"late","context_cap_tokens":1000,"minimum_window_overlap_tokens":100});
    let policy: ChunkPolicy = serde_json::from_value(old.clone()).unwrap();
    assert_eq!(serde_json::to_value(&policy).unwrap(), old);
    profile.workloads.documents.as_mut().unwrap().chunking = policy;
    profile.validate(&manifest).unwrap();
    for (tokens, overlap, valid) in [
        (256, 32, true),
        (0, 0, false),
        (1001, 0, false),
        (32, 32, false),
    ] {
        let mut wire = old.clone();
        wire["passages"] = json!({"passage_tokens":tokens,"overlap_tokens":overlap});
        profile.workloads.documents.as_mut().unwrap().chunking =
            serde_json::from_value(wire).unwrap();
        assert_eq!(profile.validate(&manifest).is_ok(), valid);
    }
    let mut unknown = old;
    unknown["passages"] = json!({"passage_tokens":256,"overlap_tokens":32,"approximate":true});
    assert!(serde_json::from_value::<ChunkPolicy>(unknown).is_err());
}

#[test]
/// Fail closed on incomplete, incompatible, or forward-version profile records.
fn malformed_profile_fields_and_unknown_versions_do_not_fall_back() {
    let (manifest, profile, _) = fixture();
    let original = serde_json::to_value(profile).unwrap();
    for (pointer, bad) in [
        ("/revision", json!(0)),
        ("/generation", json!("")),
        ("/state", json!("legacy_unverified")),
        ("/format", json!("hades-embedding-profile-v2")),
        ("/model/tokenizer_files", json!([])),
        ("/model/tokenizer_files", json!(["missing.json"])),
        (
            "/model/tokenizer_files",
            json!(["tokenizer.json", "tokenizer.json"]),
        ),
        ("/capabilities/tasks", json!(["retrieval.query"])),
        ("/capabilities/text", json!(false)),
        (
            "/workloads/documents/space/representation/dimension",
            json!(0),
        ),
        (
            "/workloads/documents/space/representation/dimension",
            json!(8193),
        ),
        ("/workloads/documents/chunking/overlap_tokens", json!(500)),
        ("/workloads/documents/chunking/passage_tokens", json!(2049)),
        (
            "/workloads/documents/space/artifact_md5",
            json!("0".repeat(32)),
        ),
        (
            "/workloads/documents/space/tokenizer_manifest_md5",
            json!("0".repeat(32)),
        ),
    ] {
        let mut value = original.clone();
        *value.pointer_mut(pointer).unwrap() = bad;
        assert!(
            EmbeddingProfile::from_json(&serde_json::to_vec(&value).unwrap(), &manifest).is_err(),
            "{pointer}"
        );
    }
    for pointer in [
        "",
        "/model",
        "/capabilities",
        "/workloads/documents",
        "/workloads/documents/space",
        "/workloads/documents/chunking",
        "/workloads/documents/space/representation",
    ] {
        let mut value = original.clone();
        value
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), json!(true));
        assert!(
            EmbeddingProfile::from_json(&serde_json::to_vec(&value).unwrap(), &manifest).is_err()
        );
    }
    assert!(EmbeddingProfile::from_json(&vec![b' '; MAX_PROFILE_BYTES + 1], &manifest).is_err());
}

#[test]
/// Preserve vector-space identity while keeping operational and corpus pins distinct.
fn provider_location_and_corpus_generation_are_not_vector_space_identity() {
    let (manifest, profile, receipt) = fixture();
    let mut relocated = profile.clone();
    relocated.runtime_binding.provider_id = "other-registered-endpoint".into();
    relocated.revision = 2;
    relocated.generation = "corpus-fixture-2".into();
    let a = profile
        .bind(
            &manifest,
            WorkloadKind::Documents,
            EmbeddingRole::Query,
            &receipt,
        )
        .unwrap();
    let b = relocated
        .bind(
            &manifest,
            WorkloadKind::Documents,
            EmbeddingRole::Query,
            &receipt,
        )
        .unwrap();
    assert_eq!(a.expected.space_id, b.expected.space_id);
    assert_ne!(a.profile_revision, b.profile_revision);
    assert_ne!(a.generation, b.generation);
    assert_ne!(a.provider_id, b.provider_id);
}
