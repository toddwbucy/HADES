//! Linux child-group ownership and bounded pipe capture for detached ingestion.
use anyhow::{Context, Result, anyhow};
use std::collections::VecDeque;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};

#[derive(Clone, Copy)]
struct Limits {
    stdout: usize,
    stderr_tail: usize,
    runtime: Duration,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            stdout: 8 * 1024 * 1024,
            stderr_tail: 64 * 1024,
            runtime: Duration::from_secs(6 * 60 * 60),
        }
    }
}

pub(crate) struct Output {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr_tail: Vec<u8>,
    pub stderr_bytes: u64,
}

struct Group(u32);
impl Group {
    fn kill(&mut self) -> Result<()> {
        if self.0 == 0 {
            return Ok(());
        }
        // SAFETY: this is the fresh process group of our unreaped child. The
        // unreaped leader prevents PID reuse until group signalling completes.
        let result = unsafe { libc::kill(-(self.0 as i32), libc::SIGKILL) };
        if result != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error.into());
            }
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

pub(crate) struct Process {
    // Drop order: signal the group before Tokio receives its direct orphan.
    group: Group,
    child: Child,
}
impl Process {
    pub fn spawn(command: &mut Command) -> Result<Self> {
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .context("failed to start ingest")?;
        let group = Group(child.id().context("ingest child has no PID")?);
        Ok(Self { group, child })
    }

    pub fn id(&self) -> u32 {
        self.group.0
    }

    pub async fn finish(self) -> Result<Output> {
        self.finish_with(Limits::default()).await
    }

    async fn finish_with(mut self, limits: Limits) -> Result<Output> {
        let stdout = self.child.stdout.take().context("missing ingest stdout")?;
        let stderr = self.child.stderr.take().context("missing ingest stderr")?;
        let pid = self.id();
        let captured = {
            let capture = async {
                tokio::try_join!(
                    read_stdout(stdout, limits.stdout),
                    read_tail(stderr, limits.stderr_tail)
                )
            };
            tokio::pin!(capture);
            let work = async {
                tokio::select! {
                    result = &mut capture => {
                        let result = result?;
                        observe_exit(pid).await?;
                        Ok::<_, anyhow::Error>(result)
                    }
                    exited = observe_exit(pid) => {
                        exited?;
                        // Descendants may keep pipes open after the leader
                        // exits. Kill them before waiting for pipe EOF.
                        self.group.kill()?;
                        capture.await
                    }
                }
            };
            tokio::time::timeout(limits.runtime, work)
                .await
                .unwrap_or_else(|_| Err(anyhow!("ingest runtime limit exceeded")))
        };
        let killed = self.group.kill();
        let status = self.child.wait().await.context("failed to reap ingest")?;
        killed?;
        let (stdout, (stderr_tail, stderr_bytes)) = captured?;
        Ok(Output {
            status,
            stdout,
            stderr_tail,
            stderr_bytes,
        })
    }
}

async fn observe_exit(pid: u32) -> Result<()> {
    loop {
        let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
        // SAFETY: owned child, valid output pointer, observation without reap.
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
        // SAFETY: successful waitid initializes the output structure.
        if unsafe { info.assume_init().si_pid() } != 0 {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn read_stdout(mut stream: impl AsyncRead + Unpin, cap: usize) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0; 8192];
    loop {
        let n = stream.read(&mut buffer).await?;
        if n == 0 {
            return Ok(output);
        }
        if n > cap.saturating_sub(output.len()) {
            return Err(anyhow!("ingest stdout byte limit exceeded"));
        }
        output.extend_from_slice(&buffer[..n]);
    }
}

async fn read_tail(mut stream: impl AsyncRead + Unpin, cap: usize) -> Result<(Vec<u8>, u64)> {
    let mut tail = VecDeque::with_capacity(cap);
    let mut total = 0_u64;
    let mut buffer = [0; 8192];
    loop {
        let n = stream.read(&mut buffer).await?;
        if n == 0 {
            return Ok((tail.into(), total));
        }
        total = total.saturating_add(n as u64);
        let keep = n.min(cap);
        let remove = (tail.len() + keep).saturating_sub(cap);
        tail.drain(..remove);
        tail.extend(&buffer[n - keep..n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn python(script: &str) -> Process {
        Process::spawn(Command::new("/usr/bin/python3").args(["-c", script])).unwrap()
    }
    fn limits() -> Limits {
        Limits {
            stdout: 1024,
            stderr_tail: 32,
            runtime: Duration::from_secs(2),
        }
    }
    fn absent(pid: u32) -> bool {
        // SAFETY: signal zero does not signal the private process.
        (unsafe { libc::kill(pid as i32, 0) }) == -1
            && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
    }

    #[tokio::test]
    async fn huge_single_line_diagnostics_keep_only_bounded_tail() {
        let process = python(
            "import os\nos.write(2, b'x' * 1048576 + b'END')\nos.write(1, b'{\"ok\":true}')",
        );
        let pid = process.id();
        let output = process.finish_with(limits()).await.unwrap();
        assert!(output.status.success());
        assert_eq!(output.stdout, br#"{"ok":true}"#);
        assert_eq!(output.stderr_bytes, 1048579);
        assert_eq!(output.stderr_tail.len(), 32);
        assert!(output.stderr_tail.ends_with(b"END"));
        assert!(absent(pid));
    }

    #[tokio::test]
    async fn stdout_overflow_and_runtime_expiry_reap_direct_children() {
        for (script, error) in [
            (
                "import os,time\nos.write(1,b'x'*4096)\ntime.sleep(10)",
                "stdout byte limit",
            ),
            ("import time\ntime.sleep(10)", "runtime limit"),
        ] {
            let process = python(script);
            let pid = process.id();
            let result = process
                .finish_with(Limits {
                    runtime: Duration::from_millis(400),
                    ..limits()
                })
                .await;
            assert!(result.err().unwrap().to_string().contains(error));
            assert!(absent(pid));
        }
    }

    #[tokio::test]
    async fn exited_leader_does_not_leave_descendant_holding_pipes() {
        let process = python(
            "import os,subprocess\np=subprocess.Popen(['/usr/bin/python3','-c','import time;time.sleep(10)'])\nos.write(1,str(p.pid).encode())\nos._exit(0)",
        );
        let pid = process.id();
        let output = process.finish_with(limits()).await.unwrap();
        assert!(output.status.success());
        assert!(absent(pid));
        let descendant: u32 = std::str::from_utf8(&output.stdout)
            .unwrap()
            .parse()
            .unwrap();
        // Production owns/reaps the direct child; the OS adopts its descendant.
        // Actual-server tests will use a private subreaper to assert disappearance.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            match std::fs::read_to_string(format!("/proc/{descendant}/stat")) {
                Ok(stat) if stat.rsplit_once(") ").unwrap().1.starts_with('Z') => break,
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
                Err(error) => panic!("cannot inspect private descendant: {error}"),
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "private descendant survived cleanup"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}
