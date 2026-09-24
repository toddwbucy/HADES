//! Persisted phases and detached ownership of ingestion startup/completion.
use super::{Reservation, process::Process};
use crate::config::HadesConfig;
use crate::db::{ArangoPool, crud, query};
use crate::dispatch::HandlerError;
use futures::FutureExt;
use serde_json::{Value, json};
use std::panic::AssertUnwindSafe;
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

const COLLECTION: &str = "hades_ingest_jobs";
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);

fn service(error: impl std::fmt::Display) -> HandlerError {
    HandlerError::ServiceError(error.to_string())
}

pub(crate) async fn start(
    pool: &ArangoPool,
    config: &HadesConfig,
    path: &str,
    force: bool,
    allow_degraded_enrichment: bool,
) -> Result<Value, HandlerError> {
    let exe = std::env::current_exe().map_err(service)?;
    start_with(
        pool,
        config,
        path,
        force,
        allow_degraded_enrichment,
        tokio::process::Command::new(exe),
    )
    .await
}

async fn start_with(
    pool: &ArangoPool,
    config: &HadesConfig,
    path: &str,
    force: bool,
    allow_degraded_enrichment: bool,
    mut command: tokio::process::Command,
) -> Result<Value, HandlerError> {
    let bounded_pool = ArangoPool::new(
        pool.reader()
            .clone()
            .with_response_limit(64 * 1024)
            .map_err(service)?,
        pool.writer()
            .clone()
            .with_response_limit(64 * 1024)
            .map_err(service)?,
    );
    let pool = &bounded_pool;
    let database = config.effective_database().map_err(service)?.to_owned();
    let resolved = std::fs::canonicalize(path).map_err(|e| HandlerError::InvalidParameter {
        name: "path".into(),
        reason: e.to_string(),
    })?;
    if resolved.to_str().is_none() {
        return Err(HandlerError::InvalidParameter {
            name: "path".into(),
            reason: "canonical ingestion path must be valid UTF-8".into(),
        });
    }
    // Reject before reserving, creating records or spawning: a file child cannot
    // apply the semantic override, so persisting true would be misleading (#185).
    if allow_degraded_enrichment && !resolved.is_dir() {
        return Err(HandlerError::InvalidParameter {
            name: "allow_degraded_enrichment".into(),
            reason: "applies to a directory ingest; pass the directory instead".into(),
        });
    }
    let source_git = crate::source_git::resolve(&resolved).map_err(service)?;
    let reservation = super::reserve(&resolved).map_err(service)?;
    if let Err(error) = crud::create_collection(pool, COLLECTION, Some(2)).await
        && error.kind() != crate::db::ArangoErrorKind::Conflict
    {
        return Err(service(format!("cannot prepare ingest records: {error}")));
    }

    // Database history is not a lock. Unowned unfinished rows require explicit
    // reconciliation; a numeric PID's existence cannot establish ownership.
    let rows = query::query_fold(pool,
        "FOR j IN hades_ingest_jobs FILTER j.status IN ['starting','running'] LIMIT 3 RETURN {key:j._key, current_owner:j.owner_instance == @owner}",
        json!({"owner":super::instance()}),
        query::FoldLimits { batch_size:3, response_bytes:64*1024, max_rows:3, server_memory_bytes:8*1024*1024 },
        Vec::new(), |rows, row| { rows.push(row); Ok(()) }, ()
    ).await.map_err(|e| service(format!("cannot verify ingest admission: {e}")))?;
    if rows.iter().any(|row| {
        row["current_owner"] != true
            || !row["key"]
                .as_str()
                .is_some_and(|key| super::owns(&database, key))
    }) {
        return Err(service(
            "unfinished ingest records are not owned by this service; review and reconcile them before retrying",
        ));
    }

    let job = format!("{:032x}", rand::random::<u128>());
    let snapshot = crate::config::snapshot::Snapshot::new(config).map_err(service)?;
    let fd = snapshot.inherit(&mut command);
    command.arg("--resolved-config-fd").arg(fd.to_string());
    match &config.gpu.cuda_visible_devices {
        Some(mask) => {
            command.env("CUDA_VISIBLE_DEVICES", mask);
        }
        None => {
            command.env_remove("CUDA_VISIBLE_DEVICES");
        }
    }
    reservation.identify(&database, &job).map_err(service)?;
    let row = json!({"_key":job,"status":"starting","owner_instance":super::instance(),
        "database":database,"path":resolved,"source_git":source_git,"force":force,"allow_degraded_enrichment":allow_degraded_enrichment,"started_at":chrono::Utc::now().to_rfc3339(),
        "output_storage":"bounded_job_record","log_path":null,"stderr_path":null});
    command
        .arg("--db")
        .arg(&database)
        .arg("ingest")
        .arg(&resolved);
    if force {
        command.arg("--force");
    }
    if allow_degraded_enrichment {
        command.arg("--allow-degraded-enrichment");
    }
    let pool = pool.clone();
    let (reply, receive) = oneshot::channel();
    // Ownership transfers before the first job insertion await. A cancelled
    // caller cannot strand a starting row without its cleanup task.
    tokio::spawn(own_start(pool, row, command, reservation, reply));
    receive
        .await
        .map_err(|_| service("ingest startup owner stopped"))?
}

async fn persist(pool: &ArangoPool, job: &str, update: &Value) -> bool {
    for attempt in 0..3 {
        if matches!(
            timeout(
                WRITE_TIMEOUT,
                crud::update_document(pool, COLLECTION, job, update)
            )
            .await,
            Ok(Ok(_))
        ) {
            return true;
        }
        if attempt < 2 {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    // Do not dump database error bodies or captured child output into logs.
    tracing::error!(
        job,
        "ingest outcome persistence failed after bounded retries; reconciliation required"
    );
    false
}

fn failed(reason: &str) -> Value {
    json!({"status":"failed","finished_at":chrono::Utc::now().to_rfc3339(),"detail":reason,"result":null,"stderr_bytes":null,"stderr_truncated":null})
}

async fn own_start(
    pool: ArangoPool,
    row: Value,
    mut command: tokio::process::Command,
    reservation: Reservation,
    reply: oneshot::Sender<Result<Value, HandlerError>>,
) {
    let job = row["_key"]
        .as_str()
        .expect("internally constructed job key");
    let shutdown = reservation.shutdown();
    let _reservation = reservation;
    if !matches!(
        timeout(
            WRITE_TIMEOUT,
            crud::insert_document(&pool, COLLECTION, &row)
        )
        .await,
        Ok(Ok(_))
    ) {
        // Insertion may have committed even if the response was lost.
        persist(
            &pool,
            job,
            &failed("job insertion was not confirmed; child not started"),
        )
        .await;
        let _ = reply.send(Err(service(
            "failed to confirm ingest job record; child not started",
        )));
        return;
    }
    if shutdown.is_cancelled() || reply.is_closed() {
        persist(&pool, job, &failed("ingest cancelled before child startup")).await;
        let _ = reply.send(Err(service("ingest cancelled before child startup")));
        return;
    }
    let child = match Process::spawn(&mut command) {
        Ok(child) => child,
        Err(_) => {
            persist(&pool, job, &failed("failed to spawn ingest child")).await;
            let _ = reply.send(Err(service("failed to spawn ingest child")));
            return;
        }
    };
    let pid = child.id();
    let running =
        json!({"status":"running","pid":pid,"running_at":chrono::Utc::now().to_rfc3339()});
    let child = match confirm_child_running(
        child,
        crud::update_document(&pool, COLLECTION, job, &running),
        &shutdown,
    )
    .await
    {
        Ok(child) => child,
        Err(()) => {
            persist(
                &pool,
                job,
                &failed("ingest stopped because running state was not confirmed"),
            )
            .await;
            let _ = reply.send(Err(service(
                "ingest running state not confirmed; child stopped",
            )));
            return;
        }
    };
    let _ = reply.send(Ok(json!({"job_id":job,"status":"running","database":row["database"],"path":row["path"],"source_git":row["source_git"],"pid":pid,"poll":"ingest.status"})));
    let update = match child.finish(shutdown).await {
        Ok(output) => outcome(output),
        Err(error) => failed(&error.to_string()),
    };
    persist(&pool, job, &update).await;
}

async fn confirm_child_running<E>(
    child: Process,
    write: impl Future<Output = Result<Value, E>>,
    shutdown: &CancellationToken,
) -> Result<Process, ()> {
    // Keep the spawned process outside the database future's unwind boundary.
    // A panic must follow the same reap-before-release path as an uncertain write.
    let recorded = AssertUnwindSafe(async {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => false,
            result = timeout(WRITE_TIMEOUT, write) => matches!(result, Ok(Ok(_))),
        }
    })
    .catch_unwind()
    .await
    .unwrap_or(false);
    if recorded {
        return Ok(child);
    }
    let cancel = CancellationToken::new();
    cancel.cancel();
    let _ = child.finish(cancel).await;
    Err(())
}

fn outcome(output: super::process::Output) -> Value {
    let parsed = serde_json::from_slice::<Value>(&output.stdout);
    let mut diagnostic = String::from_utf8_lossy(&output.stderr_tail).into_owned();
    let mut start = diagnostic.len().saturating_sub(64 * 1024);
    while !diagnostic.is_char_boundary(start) {
        start += 1;
    }
    let truncated = start != 0 || output.stderr_bytes > output.stderr_tail.len() as u64;
    diagnostic.drain(..start);
    let success = output.status.success() && parsed.is_ok();
    let detail = if success {
        None
    } else {
        let reason = if output.status.success() {
            "ingest returned malformed JSON".to_owned()
        } else {
            format!("exit status {}", output.status)
        };
        Some(format!("{reason}\n{diagnostic}"))
    };
    json!({"status":if success {"completed"} else {"failed"},"finished_at":chrono::Utc::now().to_rfc3339(),"detail":detail,"result":parsed.ok(),"stderr_bytes":output.stderr_bytes,"stderr_truncated":truncated})
}

pub(crate) async fn status(pool: &ArangoPool, job: &str) -> Result<Value, HandlerError> {
    if !matches!(job.len(), 16 | 32) || !job.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(HandlerError::InvalidParameter {
            name: "job_id".into(),
            reason: "expected a 16- or 32-character hexadecimal job ID".into(),
        });
    }
    let mut row = read_job(pool, job).await?;
    let owned = row["owner_instance"] == super::instance() && super::owns(pool.database(), job);
    if matches!(row["status"].as_str(), Some("starting" | "running"))
        && row["owner_instance"] == super::instance()
        && !owned
    {
        // The owner persists its outcome before releasing admission. A read
        // begun before that write can finish after release. Refresh once after
        // observing the release rather than misclassifying a completed job.
        row = read_job(pool, job).await?;
    }
    if matches!(row["status"].as_str(), Some("starting" | "running")) {
        row["owned_by_service"] = json!(owned);
        if !owned {
            row["recorded_status"] = row["status"].clone();
            row["status"] = json!("recovery_required");
            row["note"] = json!(
                "No current owner can verify this unfinished record. Process liveness and final outcome are unknown; reconcile before retrying. This response does not modify the stored record."
            );
        }
    }
    Ok(row)
}

async fn read_job(pool: &ArangoPool, job: &str) -> Result<Value, HandlerError> {
    pool.reader()
        .clone()
        .with_response_limit(16 * 1024 * 1024)
        .map_err(service)?
        .get(&format!("document/{COLLECTION}/{job}"))
        .await
        .map_err(|error| {
            if error.is_not_found() {
                HandlerError::DocumentNotFound {
                    collection: COLLECTION.into(),
                    key: job.into(),
                }
            } else {
                service(error)
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor_mock::{Mock, Reply};
    use std::sync::Arc;
    use tokio::sync::Notify;

    #[tokio::test]
    async fn panicking_pid_confirmation_reaps_before_admission_release() {
        let admission = Arc::new(super::super::Admission::new(1));
        let path = std::path::Path::new("/private-confirmation-fixture");
        let reservation = admission.reserve(path).unwrap();
        let child = Process::spawn(&mut python("import time; time.sleep(10)")).unwrap();
        let pid = child.id();
        let write = async {
            tokio::task::yield_now().await;
            panic!("private PID persistence fault");
            #[allow(unreachable_code)]
            Ok::<Value, std::io::Error>(json!({}))
        };
        assert!(
            confirm_child_running(child, write, &reservation.shutdown())
                .await
                .is_err()
        );
        assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
        assert!(
            admission
                .reserve(std::path::Path::new("/another-private-fixture"))
                .is_err()
        );
        drop(reservation);
        assert!(admission.reserve(path).is_ok());
    }

    #[tokio::test]
    async fn source_git_is_preserved_in_job_start_and_status() {
        let _test = super::super::TEST_LOCK.lock().await;
        let root = tempfile::tempdir().unwrap();
        let config = HadesConfig::with_database("fixture");
        let git = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(root.path())
                .args(args)
                .output()
                .unwrap();
            assert!(out.status.success(), "{out:?}");
            out
        };
        let mut expected = Value::Null;
        for state in ["non_git", "clean", "dirty"] {
            if state == "clean" {
                git(&["init", "-q"]);
                git(&[
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "--allow-empty",
                    "-qm",
                    "fixture",
                ]);
                expected = json!({"commit": String::from_utf8(git(&["rev-parse", "HEAD"]).stdout).unwrap().trim(), "dirty":false});
            } else if state == "dirty" {
                std::fs::write(root.path().join("untracked"), "dirty").unwrap();
                expected["dirty"] = json!(true);
            }
            let mut replies = admission();
            replies.extend([ok(), ok(), ok()]);
            let mut mock = Mock::new(replies).await;
            let started = start_with(
                &mock.pool,
                &config,
                root.path().to_str().unwrap(),
                false,
                false,
                python("print('{}')"),
            )
            .await
            .unwrap();
            let mut row = inserted(&mut mock).await;
            assert_eq!(row.get("source_git"), Some(&expected));
            assert_eq!(started.get("source_git"), Some(&expected));
            let job = row["_key"].as_str().unwrap().to_owned();
            mock.event(&format!("PATCH document/{COLLECTION}/{job}"))
                .await;
            let finished = mock
                .event(&format!("PATCH document/{COLLECTION}/{job}"))
                .await;
            released(&job).await;
            row.as_object_mut()
                .unwrap()
                .extend(finished.as_object().unwrap().clone());
            let read = Mock::new(vec![Reply::page(row)]).await;
            let reported = status(&read.pool, &job).await.unwrap();
            assert_eq!(reported.get("source_git"), Some(&expected));
        }
    }

    #[tokio::test]
    async fn missing_git_is_named_at_job_admission() {
        const CHILD: &str = "HADES_MISSING_GIT_TEST_CHILD";
        let root = tempfile::tempdir().unwrap();
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "ingest_jobs::records::tests::missing_git_is_named_at_job_admission",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("PATH", root.path())
                .output()
                .unwrap();
            assert!(output.status.success(), "{output:?}");
            return;
        }
        let mut mock = Mock::new(vec![]).await;
        let error = start_with(
            &mock.pool,
            &HadesConfig::with_database("fixture"),
            root.path().to_str().unwrap(),
            false,
            false,
            tokio::process::Command::new("/bin/false"),
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("git executable not found on PATH"),
            "{error}"
        );
        assert!(
            mock.events.try_recv().is_err(),
            "missing git must fail before database admission"
        );
    }

    fn ok() -> Reply {
        Reply::page(json!({}))
    }
    fn unavailable() -> Reply {
        Reply {
            status: 503,
            body: json!({"error":true,"errorMessage":"private fixture failure"}),
            gate: None,
        }
    }
    fn admission() -> Vec<Reply> {
        vec![ok(), Reply::page(json!({"result":[],"hasMore":false}))]
    }
    fn python(script: &str) -> tokio::process::Command {
        let mut command = tokio::process::Command::new("/usr/bin/python3");
        command.args(["-c", script]);
        command
    }
    async fn inserted(mock: &mut Mock) -> Value {
        mock.event("POST collection").await;
        let query = mock.event("POST cursor").await;
        assert_eq!(query["memoryLimit"], 8 * 1024 * 1024);
        let row = mock.event("POST document/hades_ingest_jobs").await;
        assert_eq!(row["status"], "starting");
        assert_eq!(row["owner_instance"], super::super::instance());
        assert_eq!(row["log_path"], Value::Null);
        row
    }
    async fn released(job: &str) {
        timeout(Duration::from_secs(3), async {
            while super::super::owns("fixture", job) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn persisted_phases_cover_success_spawn_pid_output_and_retry_failures() {
        let _test = super::super::TEST_LOCK.lock().await;
        let root = tempfile::tempdir().unwrap();
        let path = root.path().to_str().unwrap();
        let config = HadesConfig::with_database("fixture");
        for (script, expected) in [
            ("print('{\"ok\":true}')", "completed"),
            ("print('not-json')", "failed"),
            (
                "import os;os.write(2,b'\\xff'*65533+b'END');os._exit(1)",
                "failed",
            ),
        ] {
            let mut replies = admission();
            replies.extend([ok(), ok(), ok()]);
            let mut mock = Mock::new(replies).await;
            let started = start_with(&mock.pool, &config, path, false, false, python(script))
                .await
                .unwrap();
            let row = inserted(&mut mock).await;
            assert_eq!(row["_key"], started["job_id"]);
            let job = row["_key"].as_str().unwrap();
            let running = mock
                .event(&format!("PATCH document/{COLLECTION}/{job}"))
                .await;
            assert_eq!(running["status"], "running");
            assert_eq!(running["pid"], started["pid"]);
            let finished = mock
                .event(&format!("PATCH document/{COLLECTION}/{job}"))
                .await;
            assert_eq!(finished["status"], expected);
            if script.contains("xff") {
                assert!(finished["detail"].as_str().unwrap().ends_with("END"));
                assert!(finished["detail"].as_str().unwrap().len() <= 64 * 1024 + 128);
                assert_eq!(finished["stderr_truncated"], true);
                assert_eq!(finished["stderr_bytes"], 65536);
            }
            released(job).await;
            assert!(mock.events.try_recv().is_err());
        }

        // Exercise the actual startup command's handoff, not only snapshot
        // serialization. Return booleans so no credential enters the job result.
        let mut effective = config.clone();
        effective.database.password = Some("fixture-snapshot-secret".into());
        effective.embedding.service.socket = "/private/selected-embed.sock".into();
        effective.gpu.enabled = false;
        effective.gpu.cuda_visible_devices = Some("fixture-mask".into());
        let mut command = python(
            "import json,os,sys\nfd=int(sys.argv[sys.argv.index('--resolved-config-fd')+1])\ns=json.loads(os.pread(fd,65536,0))\nprint(json.dumps({'preserved':s['password']=='fixture-snapshot-secret' and s['config']['database']['name']=='fixture' and s['config']['embedding']['service']['socket']=='/private/selected-embed.sock' and not s['config']['gpu']['enabled'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='fixture-mask'}))",
        );
        command
            .env("CUDA_VISIBLE_DEVICES", "wrong")
            .env("HADES_CONFIG", "/nonexistent/fixture");
        let mut replies = admission();
        replies.extend([ok(), ok(), ok()]);
        let mut mock = Mock::new(replies).await;
        start_with(&mock.pool, &effective, path, false, false, command)
            .await
            .unwrap();
        let row = inserted(&mut mock).await;
        let job = row["_key"].as_str().unwrap();
        mock.event(&format!("PATCH document/{COLLECTION}/{job}"))
            .await;
        let done = mock
            .event(&format!("PATCH document/{COLLECTION}/{job}"))
            .await;
        assert_eq!(done["status"], "completed");
        assert_eq!(done["result"], json!({"preserved":true}));
        assert!(!done.to_string().contains("fixture-snapshot-secret"));
        released(job).await;

        // A failed spawn receives a terminal record before returning its error.
        let mut replies = admission();
        replies.extend([ok(), ok()]);
        let mut mock = Mock::new(replies).await;
        let error = start_with(
            &mock.pool,
            &config,
            path,
            false,
            false,
            tokio::process::Command::new(root.path().join("missing-executable")),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("failed to spawn"));
        let row = inserted(&mut mock).await;
        let job = row["_key"].as_str().unwrap();
        assert_eq!(
            mock.event(&format!("PATCH document/{COLLECTION}/{job}"))
                .await["status"],
            "failed"
        );
        released(job).await;
        assert!(mock.events.try_recv().is_err());

        // Lost PID persistence stops/reaps the actual private child before the
        // failure record or caller error; it does not return a running job.
        let mut replies = admission();
        replies.extend([ok(), unavailable(), ok()]);
        let mut mock = Mock::new(replies).await;
        assert!(
            start_with(
                &mock.pool,
                &config,
                path,
                false,
                false,
                python("import time;time.sleep(10)")
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("running state not confirmed")
        );
        let row = inserted(&mut mock).await;
        let job = row["_key"].as_str().unwrap();
        let running = mock
            .event(&format!("PATCH document/{COLLECTION}/{job}"))
            .await;
        let pid = running["pid"].as_u64().unwrap() as i32;
        // SAFETY: signal zero only inspects the recorded private child.
        assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
        assert_eq!(
            mock.event(&format!("PATCH document/{COLLECTION}/{job}"))
                .await["status"],
            "failed"
        );
        released(job).await;

        // Completion retries are bounded; a stale row then reports unknown
        // ownership even when its numeric PID happens to be a live process.
        let mut replies = admission();
        replies.extend([ok(), ok(), unavailable(), unavailable(), unavailable()]);
        let mut mock = Mock::new(replies).await;
        let started = start_with(
            &mock.pool,
            &config,
            path,
            false,
            false,
            python("print('{}')"),
        )
        .await
        .unwrap();
        let row = inserted(&mut mock).await;
        let job = row["_key"].as_str().unwrap();
        mock.event(&format!("PATCH document/{COLLECTION}/{job}"))
            .await;
        for _ in 0..3 {
            assert_eq!(
                mock.event(&format!("PATCH document/{COLLECTION}/{job}"))
                    .await["status"],
                "completed"
            );
        }
        released(job).await;
        assert!(mock.events.try_recv().is_err());
        let mut stale = row.clone();
        stale["status"] = json!("running");
        stale["pid"] = json!(std::process::id());
        let mut reader =
            Mock::new(vec![Reply::page(stale.clone()), Reply::page(stale.clone())]).await;
        let result = status(&reader.pool, started["job_id"].as_str().unwrap())
            .await
            .unwrap();
        assert_eq!(result["status"], "recovery_required");
        assert_eq!(result["recorded_status"], "running");
        assert_eq!(result["owned_by_service"], false);
        reader
            .event(&format!("GET document/{COLLECTION}/{job}"))
            .await;
        reader
            .event(&format!("GET document/{COLLECTION}/{job}"))
            .await;
        assert!(reader.events.try_recv().is_err());

        // A delayed running snapshot must not override the terminal state
        // persisted before the owner released its reservation.
        let terminal = json!({"_key":job,"owner_instance":super::super::instance(),
            "status":"completed","result":{"success":true}});
        let mut reader = Mock::new(vec![Reply::page(stale), Reply::page(terminal.clone())]).await;
        assert_eq!(status(&reader.pool, job).await.unwrap(), terminal);
        for _ in 0..2 {
            reader
                .event(&format!("GET document/{COLLECTION}/{job}"))
                .await;
        }
        assert!(reader.events.try_recv().is_err());
    }

    #[tokio::test]
    async fn cancelled_startup_finishes_its_record_without_spawning() {
        let _test = super::super::TEST_LOCK.lock().await;
        let root = tempfile::tempdir().unwrap();
        let marker = root.path().join("must-not-exist");
        let gate = Arc::new(Notify::new());
        let mut replies = admission();
        replies.extend([Reply::blocked(json!({}), gate.clone()), ok()]);
        let mut mock = Mock::new(replies).await;
        let pool = mock.pool.clone();
        let path = root.path().to_owned();
        let command = python(&format!(
            "from pathlib import Path;Path({:?}).touch()",
            marker.to_str().unwrap()
        ));
        let task = tokio::spawn(async move {
            start_with(
                &pool,
                &HadesConfig::with_database("fixture"),
                path.to_str().unwrap(),
                false,
                false,
                command,
            )
            .await
        });
        let row = inserted(&mut mock).await;
        let job = row["_key"].as_str().unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(super::super::owns("fixture", job));
        gate.notify_one();
        let finished = mock
            .event(&format!("PATCH document/{COLLECTION}/{job}"))
            .await;
        assert_eq!(finished["status"], "failed");
        assert!(
            finished["detail"]
                .as_str()
                .unwrap()
                .contains("before child startup")
        );
        released(job).await;
        assert!(!marker.exists());
        assert!(mock.events.try_recv().is_err());
    }

    #[tokio::test]
    async fn unconfirmed_insert_and_unowned_records_do_not_spawn() {
        let _test = super::super::TEST_LOCK.lock().await;
        let root = tempfile::tempdir().unwrap();
        let path = root.path().to_str().unwrap();
        let marker = root.path().join("must-not-exist");
        let script = format!(
            "from pathlib import Path;Path({:?}).touch()",
            marker.to_str().unwrap()
        );
        let config = HadesConfig::with_database("fixture");
        let mut replies = admission();
        replies.extend([unavailable(), ok()]);
        let mut mock = Mock::new(replies).await;
        assert!(
            start_with(&mock.pool, &config, path, false, false, python(&script))
                .await
                .unwrap_err()
                .to_string()
                .contains("failed to confirm")
        );
        let row = inserted(&mut mock).await;
        let job = row["_key"].as_str().unwrap();
        let failure = mock
            .event(&format!("PATCH document/{COLLECTION}/{job}"))
            .await;
        assert_eq!(failure["status"], "failed");
        assert!(
            failure["detail"]
                .as_str()
                .unwrap()
                .contains("child not started")
        );
        released(job).await;
        assert!(!marker.exists());

        // An old owner and a current-instance row whose owner has finished
        // both block admission. PID presence is deliberately not consulted.
        for current_owner in [false, true] {
            let mut mock = Mock::new(vec![
                ok(),
                Reply::page(
                    json!({"result":[{"key":job,"current_owner":current_owner}],"hasMore":false}),
                ),
            ])
            .await;
            assert!(
                start_with(&mock.pool, &config, path, false, false, python(&script))
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("not owned")
            );
            mock.event("POST collection").await;
            mock.event("POST cursor").await;
            assert!(mock.events.try_recv().is_err());
            assert!(!marker.exists());
        }

        let mut mock = Mock::new(vec![]).await;
        for invalid in ["../other", "", "abc", "0123456789abcde/"] {
            assert!(matches!(
                status(&mock.pool, invalid).await,
                Err(HandlerError::InvalidParameter { .. })
            ));
        }
        // A UTF-8 symlink can resolve to a non-UTF-8 directory. Reject it
        // before serializing the job row or issuing any database request.
        use std::os::unix::ffi::OsStringExt;
        let native = root
            .path()
            .join(std::ffi::OsString::from_vec(vec![b'x', 0xff]));
        std::fs::create_dir(&native).unwrap();
        let alias = root.path().join("native-alias");
        std::os::unix::fs::symlink(&native, &alias).unwrap();
        assert!(matches!(
            start_with(
                &mock.pool,
                &config,
                alias.to_str().unwrap(),
                false,
                false,
                python(&script)
            )
            .await,
            Err(HandlerError::InvalidParameter { .. })
        ));
        assert!(mock.events.try_recv().is_err());
    }
}
