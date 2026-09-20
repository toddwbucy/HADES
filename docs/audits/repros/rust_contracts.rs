//! Isolated audit probes. Build using the temporary manifest described in the report.
//! These contracts deliberately fail on the audited revision.

#[cfg(test)]
mod tests {
    use hades_core::db::keys;
    use hades_core::persephone::embedding::EmbeddingClient;
    use std::io::{Read, Write};
    use std::os::unix::net::UnixListener;

    #[test]
    fn distinct_source_paths_have_distinct_keys() {
        assert_ne!(keys::file_key("a/b.py"), keys::file_key("a_b.py"));
    }

    async fn embed_response(body: &str) -> bool {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mock.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let body = body.to_owned();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut incoming = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = stream.read(&mut buf).unwrap();
                assert!(n > 0);
                incoming.extend_from_slice(&buf[..n]);
                if let Some(header_end) = incoming.windows(4).position(|s| s == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&incoming[..header_end]).to_lowercase();
                    let len: usize = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length:"))
                        .unwrap()
                        .trim()
                        .parse()
                        .unwrap();
                    if incoming.len() >= header_end + 4 + len {
                        break;
                    }
                }
            }
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
        });
        let client = EmbeddingClient::connect_unix_at(path).await.unwrap();
        let rejected = client
            .embed(&["first".into(), "second".into()], "code", None)
            .await
            .is_err();
        server.join().unwrap();
        rejected
    }

    #[tokio::test]
    async fn duplicate_embedding_indices_are_rejected() {
        assert!(
            embed_response(
                r#"{"data":[{"index":0,"embedding":[1.0,0.0]},{"index":0,"embedding":[0.0,1.0]}]}"#
            )
            .await
        );
    }

    #[tokio::test]
    async fn nonnumeric_embedding_elements_are_rejected() {
        assert!(embed_response(r#"{"data":[{"index":0,"embedding":[1.0,"bad"]},{"index":1,"embedding":[0.0,"bad"]}]}"#).await);
    }
}
