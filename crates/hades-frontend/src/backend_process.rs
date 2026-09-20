//! Bounded ownership of the viewer's CLI subprocesses (Linux).
use anyhow::{Context, Result, anyhow};
use std::process::Stdio;
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::{Semaphore, oneshot, watch};

static SHUTDOWN: LazyLock<watch::Sender<bool>> = LazyLock::new(|| watch::channel(false).0);

static SLOTS: LazyLock<Arc<Semaphore>> = LazyLock::new(|| Arc::new(Semaphore::new(4)));

/// Register both signals before discovery can launch a backend child.
pub fn install_shutdown_signals() -> Result<()> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut interrupt = signal(SignalKind::interrupt())?;
    let mut terminate = signal(SignalKind::terminate())?;
    tokio::spawn(async move {
        tokio::select! { _ = interrupt.recv() => {}, _ = terminate.recv() => {} }
        begin_shutdown();
    });
    Ok(())
}

fn begin_shutdown() {
    tracing::debug!(
        permits = SLOTS.available_permits(),
        "viewer shutdown requested"
    );
    SLOTS.close();
    SHUTDOWN.send_replace(true);
}

pub async fn shutdown_requested() {
    let mut receiver = SHUTDOWN.subscribe();
    let _ = receiver.wait_for(|requested| *requested).await;
}

pub async fn shutdown_and_wait() {
    begin_shutdown();
    // Slots are released only after direct children are reaped. Keep the
    // runtime alive for ownership cleanup before returning from main.
    while SLOTS.available_permits() != 4 {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

// Script-writing fixtures must not overlap a concurrent fork that can briefly
// inherit their writable descriptors before exec closes them (ETXTBSY).
#[cfg(test)]
pub(crate) static SCRIPT_FIXTURE: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

#[derive(Debug)]
pub struct Overloaded;
impl std::fmt::Display for Overloaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("viewer backend overloaded")
    }
}
impl std::error::Error for Overloaded {}

#[derive(Clone, Copy)]
struct Limits {
    stdout: usize,
    stderr: usize,
    runtime: Duration,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            stdout: 8 * 1024 * 1024,
            stderr: 16 * 1024,
            runtime: Duration::from_secs(30),
        }
    }
}

pub async fn run(bin: &str, args: &[&str]) -> Result<Vec<u8>> {
    run_with(bin, args, SLOTS.clone(), Limits::default()).await
}

async fn read_limited(
    mut input: impl AsyncRead + Unpin,
    cap: usize,
    name: &str,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let n = input.read(&mut buffer).await?;
        if n == 0 {
            return Ok(bytes);
        }
        if n > cap.saturating_sub(bytes.len()) {
            return Err(anyhow!("backend {name} exceeded its byte limit"));
        }
        bytes.extend_from_slice(&buffer[..n]);
    }
}

/// Keep the leader unreaped until its process group has been signalled. This
/// prevents PID reuse between observing exit and killing lingering descendants.
async fn observe_exit(pid: u32) -> Result<()> {
    loop {
        let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
        // SAFETY: valid zeroed siginfo output, an owned child PID, and read-only
        // observation of its exit status (WNOWAIT never reaps the child).
        let result = unsafe {
            libc::waitid(
                libc::P_PID,
                pid,
                info.as_mut_ptr(),
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if result != 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error.into());
        }
        // SAFETY: waitid succeeded and initialized the supplied siginfo.
        if unsafe { info.assume_init().si_pid() } != 0 {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

struct Group(u32);
impl Group {
    fn kill(&mut self) -> Result<()> {
        if self.0 == 0 {
            return Ok(());
        }
        let pid = self.0;
        // SAFETY: spawn created this child's fresh process group; its leader
        // has not been reaped, so this ID cannot identify a reused group.
        let result = unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
        let error = std::io::Error::last_os_error();
        if result != 0 && error.raw_os_error() != Some(libc::ESRCH) {
            return Err(error.into());
        }
        self.0 = 0;
        Ok(())
    }
}
impl Drop for Group {
    fn drop(&mut self) {
        let _ = self.kill();
    }
}

// Field drop order is deliberate: signal the group before Child can be
// dropped and handed to Tokio's orphan reaper during runtime/task teardown.
struct OwnedChild {
    group: Group,
    child: tokio::process::Child,
}

async fn run_with(
    bin: &str,
    args: &[&str],
    slots: Arc<Semaphore>,
    limits: Limits,
) -> Result<Vec<u8>> {
    run_with_shutdown(bin, args, slots, limits, SHUTDOWN.subscribe()).await
}

async fn run_with_shutdown(
    bin: &str,
    args: &[&str],
    slots: Arc<Semaphore>,
    limits: Limits,
    mut shutdown: watch::Receiver<bool>,
) -> Result<Vec<u8>> {
    if *shutdown.borrow() {
        return Err(anyhow!("viewer is shutting down"));
    }
    let permit = slots.try_acquire_owned().map_err(|_| anyhow!(Overloaded))?;
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .kill_on_drop(true)
        .spawn()
        .context("failed to spawn viewer backend")?;
    let group = Group(child.id().context("backend child has no PID")?);
    let stdout = child.stdout.take().context("missing backend stdout")?;
    let stderr = child.stderr.take().context("missing backend stderr")?;
    let mut owner = OwnedChild { group, child };
    let (mut send, receive) = oneshot::channel();
    // No await between spawn and ownership transfer. Client cancellation drops
    // only the receiver; this task retains the child and admission until reap.
    tokio::spawn(async move {
        let captured = {
            let work = async {
                let (out, err, ()) = tokio::try_join!(
                    read_limited(stdout, limits.stdout, "stdout"),
                    read_limited(stderr, limits.stderr, "stderr"),
                    observe_exit(owner.group.0),
                )?;
                Ok::<_, anyhow::Error>((out, err))
            };
            tokio::select! {
                _ = send.closed() => Err(anyhow!("viewer backend request cancelled")),
                _ = shutdown.wait_for(|requested| *requested) => Err(anyhow!("viewer is shutting down")),
                result = tokio::time::timeout(limits.runtime, work) =>
                    result.unwrap_or_else(|_| Err(anyhow!("viewer backend deadline exceeded"))),
            }
        };
        tracing::debug!(pid = owner.group.0, "cleaning backend process group");
        let killed = owner.group.kill();
        let status = owner
            .child
            .wait()
            .await
            .context("failed to reap viewer backend");
        let result = (|| {
            killed?;
            let status = status?;
            let (out, _diagnostic) = captured?;
            if !status.success() {
                // Diagnostics are untrusted and may contain corpus text or
                // credentials. Do not reflect them into an HTTP error/log.
                return Err(anyhow!("viewer backend failed ({status})"));
            }
            Ok(out)
        })();
        tracing::debug!("backend process reaped");
        drop(permit);
        let _ = send.send(result);
    });
    receive.await.context("viewer backend owner stopped")?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn script(root: &std::path::Path, text: &str) -> String {
        let path = root.join("child");
        std::fs::write(&path, format!("#!/usr/bin/python3\n{text}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        path.to_str().unwrap().to_string()
    }
    fn limits() -> Limits {
        Limits {
            stdout: 1024,
            stderr: 1024,
            runtime: Duration::from_millis(400),
        }
    }
    async fn started(root: &std::path::Path) -> Vec<u32> {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if let Ok(text) = std::fs::read_to_string(root.join("pids"))
                    && let Ok(pids) = serde_json::from_str(&text)
                {
                    return pids;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap()
    }
    async fn assert_stopped(pids: &[u32], slots: &Semaphore) {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let live = pids.iter().any(|pid| {
                    std::fs::read_to_string(format!("/proc/{pid}/stat"))
                        .ok()
                        .and_then(|s| s.rsplit_once(") ").map(|(_, rest)| !rest.starts_with('Z')))
                        .unwrap_or(false)
                });
                if !live && slots.available_permits() == 1 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("private child/group or admission leaked");
    }
    fn stalled(root: &std::path::Path) -> String {
        script(
            root,
            r#"import json, os, subprocess, sys, time
child = subprocess.Popen(['/usr/bin/python3', '-c', 'import time; time.sleep(10)'])
with open(sys.argv[1], 'w') as f: json.dump([os.getpid(), child.pid], f)
time.sleep(10)"#,
        )
    }

    #[tokio::test]
    async fn normal_exit_and_output_overflow_release_admission() {
        let _fixture = SCRIPT_FIXTURE.lock().await;
        let root = tempfile::tempdir().unwrap();
        let slots = Arc::new(Semaphore::new(1));
        let bin = script(root.path(), "print('ok')");
        assert_eq!(
            run_with(&bin, &[], slots.clone(), limits()).await.unwrap(),
            b"ok\n"
        );
        assert_eq!(slots.available_permits(), 1);
        let bin = script(root.path(), "print('X' * 2048)");
        let error = run_with(&bin, &[], slots.clone(), limits())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("stdout exceeded"));
        assert_eq!(slots.available_permits(), 1);
    }

    #[tokio::test]
    async fn deadline_kills_child_and_descendant_before_releasing_slot() {
        let _fixture = SCRIPT_FIXTURE.lock().await;
        let root = tempfile::tempdir().unwrap();
        let slots = Arc::new(Semaphore::new(1));
        let bin = stalled(root.path());
        let pid_path = root.path().join("pids");
        let error = run_with(&bin, &[pid_path.to_str().unwrap()], slots.clone(), limits())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("deadline"), "{error:#}");
        assert_stopped(&started(root.path()).await, &slots).await;
    }

    #[tokio::test]
    async fn cancelled_request_kills_group_and_releases_slot() {
        let _fixture = SCRIPT_FIXTURE.lock().await;
        let root = tempfile::tempdir().unwrap();
        let slots = Arc::new(Semaphore::new(1));
        let bin = stalled(root.path());
        let path = root.path().join("pids");
        let worker_slots = slots.clone();
        let request = tokio::spawn(async move {
            run_with(
                &bin,
                &[path.to_str().unwrap()],
                worker_slots,
                Limits {
                    runtime: Duration::from_secs(10),
                    ..limits()
                },
            )
            .await
        });
        let pids = started(root.path()).await;
        assert_eq!(slots.available_permits(), 0);
        // Admission fails before trying to spawn even an invalid executable.
        let error = run_with("/nonexistent", &[], slots.clone(), limits())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("overloaded"));
        request.abort();
        let _ = request.await;
        assert_stopped(&pids, &slots).await;
    }

    #[tokio::test]
    async fn spawn_failure_releases_slot() {
        let _fixture = SCRIPT_FIXTURE.lock().await;
        let slots = Arc::new(Semaphore::new(1));
        assert!(
            run_with("/nonexistent", &[], slots.clone(), limits())
                .await
                .is_err()
        );
        assert_eq!(slots.available_permits(), 1);
    }
    #[tokio::test]
    async fn shutdown_broadcast_cleans_group_and_refuses_new_children() {
        let _fixture = SCRIPT_FIXTURE.lock().await;
        let root = tempfile::tempdir().unwrap();
        let slots = Arc::new(Semaphore::new(1));
        let (send, receive) = watch::channel(false);
        let bin = stalled(root.path());
        let path = root.path().join("pids");
        let worker_slots = slots.clone();
        let request = tokio::spawn(async move {
            run_with_shutdown(
                &bin,
                &[path.to_str().unwrap()],
                worker_slots,
                Limits {
                    runtime: Duration::from_secs(10),
                    ..limits()
                },
                receive,
            )
            .await
        });
        let children = started(root.path()).await;
        send.send_replace(true);
        let error = request.await.unwrap().unwrap_err();
        assert!(error.to_string().contains("shutting down"));
        assert_stopped(&children, &slots).await;
        let error = run_with_shutdown(
            "/nonexistent",
            &[],
            slots.clone(),
            limits(),
            send.subscribe(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("shutting down"));
        assert_eq!(slots.available_permits(), 1);
    }
}
