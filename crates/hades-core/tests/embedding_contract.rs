//! Wire-contract tests use a private mock socket; no embedding service or GPU.
#![cfg(unix)]

use hades_core::persephone::embedding::{
    EmbeddingClient, EmbeddingClientConfig, EmbeddingEndpoint,
};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::os::unix::net::UnixListener;
use std::time::{Duration, Instant};

struct Mock {
    client: EmbeddingClient,
    server: Option<std::thread::JoinHandle<()>>,
    _directory: tempfile::TempDir,
}

impl Mock {
    async fn new(responses: Vec<Value>, model: &str) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("mock.sock");
        let listener = UnixListener::bind(&path).unwrap();
        listener.set_nonblocking(true).unwrap();
        let server = std::thread::spawn(move || {
            for response in responses {
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline, "mock request timed out");
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("mock accept failed: {error}"),
                    }
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut incoming = Vec::new();
                let mut buffer = [0; 4096];
                loop {
                    let count = stream.read(&mut buffer).unwrap();
                    assert!(count > 0);
                    incoming.extend_from_slice(&buffer[..count]);
                    if let Some(end) = incoming.windows(4).position(|s| s == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&incoming[..end]).to_lowercase();
                        let len: usize = headers
                            .lines()
                            .find_map(|line| line.strip_prefix("content-length:"))
                            .unwrap()
                            .trim()
                            .parse()
                            .unwrap();
                        if incoming.len() >= end + 4 + len {
                            break;
                        }
                    }
                }
                let body = response.to_string();
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
            }
        });
        let client = EmbeddingClient::connect(EmbeddingClientConfig {
            endpoint: EmbeddingEndpoint::Unix(path),
            model: model.into(),
            expected_dimension: 2,
            timeout: Duration::from_secs(5),
            ..Default::default()
        })
        .await
        .unwrap();
        Self {
            client,
            server: Some(server),
            _directory: directory,
        }
    }

    fn finish(mut self) {
        self.server.take().unwrap().join().unwrap();
    }
}

fn valid() -> Value {
    json!({"model":"test-model", "data":[
        {"index":1,"embedding":[0.0,1.0]},
        {"index":0,"embedding":[1.0,0.0]}
    ]})
}

#[tokio::test]
async fn reorders_complete_response_into_input_order() {
    let mock = Mock::new(vec![valid()], "test-model").await;
    let result = mock
        .client
        .embed(&["first".into(), "second".into()], "code", None)
        .await
        .unwrap();
    assert_eq!(result.embeddings, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert_eq!(result.dimension, 2);
    mock.finish();
}

#[tokio::test]
async fn rejects_invalid_indices() {
    for index in [
        json!(0),
        json!(2),
        json!(-1),
        json!(0.5),
        Value::Null,
        json!("1"),
    ] {
        let mut response = valid();
        response["data"][0]["index"] = index.clone();
        let mock = Mock::new(vec![response], "test-model").await;
        assert!(
            mock.client
                .embed(&["a".into(), "b".into()], "", None)
                .await
                .is_err(),
            "index {index}"
        );
        mock.finish();
    }
}

#[tokio::test]
async fn rejects_missing_and_extra_rows() {
    for rows in [
        json!([]),
        json!([{"index":0,"embedding":[1,0]}]),
        json!([
            {"index":0,"embedding":[1,0]}, {"index":1,"embedding":[0,1]}, {"index":2,"embedding":[1,1]}
        ]),
    ] {
        let mock = Mock::new(
            vec![json!({"model":"test-model","data":rows})],
            "test-model",
        )
        .await;
        assert!(
            mock.client
                .embed(&["a".into(), "b".into()], "", None)
                .await
                .is_err()
        );
        mock.finish();
    }
}

#[tokio::test]
async fn rejects_bad_vectors_without_dropping_elements() {
    for vector in [
        json!([]),
        json!([1]),
        json!([1, 2, 3]),
        json!([1, "bad"]),
        json!([1, null]),
        json!([true, 1]),
        json!([1, 1e100]),
        Value::Null,
        json!("bad"),
    ] {
        let mut response = valid();
        response["data"][0]["embedding"] = vector.clone();
        let mock = Mock::new(vec![response], "test-model").await;
        assert!(
            mock.client
                .embed(&["a".into(), "b".into()], "", None)
                .await
                .is_err(),
            "vector {vector}"
        );
        mock.finish();
    }
}

#[tokio::test]
async fn requires_compatible_model_identity() {
    for model in [Value::Null, json!(""), json!("other-model")] {
        let mut response = valid();
        response["model"] = model;
        let mock = Mock::new(vec![response], "test-model").await;
        assert!(
            mock.client
                .embed(&["a".into(), "b".into()], "", None)
                .await
                .is_err()
        );
        mock.finish();
    }
}

#[tokio::test]
async fn accepts_reference_alias_but_rejects_batch_identity_drift() {
    let alias = "/models/jinaai--jina-embeddings-v4";
    let model = "jinaai/jina-embeddings-v4";
    let response = json!({"model":alias,"data":[{"index":0,"embedding":[1,0]}]});
    let mock = Mock::new(vec![response.clone()], model).await;
    assert_eq!(
        mock.client
            .embed(&["a".into()], "", None)
            .await
            .unwrap()
            .model,
        alias
    );
    mock.finish();
    let mut other = response.clone();
    other["model"] = json!(model);
    let mock = Mock::new(vec![response, other], model).await;
    assert!(
        mock.client
            .embed(&["a".into(), "b".into()], "", Some(1))
            .await
            .is_err()
    );
    mock.finish();
}

#[tokio::test]
async fn late_chunks_validate_indices_vectors_and_model() {
    let valid = json!({"model":"test-model","data":[
        {"index":0,"chunk_index":0,"char_start":0,"char_end":2,"embedding":[1,0]}
    ]});
    for (field, value) in [
        ("index", json!(1)),
        ("embedding", json!([1, "bad"])),
        ("embedding", json!([])),
        ("embedding", json!([1, 1e100])),
    ] {
        let mut response = valid.clone();
        response["data"][0][field] = value;
        let mock = Mock::new(vec![response], "test-model").await;
        assert!(
            mock.client
                .embed_late_chunked(&["ab".into()], "", &[vec![(0, 2)]])
                .await
                .is_err()
        );
        mock.finish();
    }
    let mut wrong_model = valid.clone();
    wrong_model["model"] = json!("other-model");
    let mock = Mock::new(vec![wrong_model], "test-model").await;
    assert!(
        mock.client
            .embed_late_chunked(&["ab".into()], "", &[vec![(0, 2)]])
            .await
            .is_err()
    );
    mock.finish();
    let mock = Mock::new(vec![valid], "test-model").await;
    assert!(
        mock.client
            .embed_late_chunked(&["ab".into()], "", &[vec![(0, 2)]])
            .await
            .is_ok()
    );
    mock.finish();
}

#[tokio::test]
async fn refuses_unspecified_model_or_vector_width() {
    for config in [
        EmbeddingClientConfig {
            expected_dimension: 0,
            ..Default::default()
        },
        EmbeddingClientConfig {
            model: " ".into(),
            ..Default::default()
        },
    ] {
        assert!(EmbeddingClient::connect(config).await.is_err());
    }
}
