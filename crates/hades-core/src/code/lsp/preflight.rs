//! Bounded synchronous probes, also usable from an owned blocking worker.
use std::io::{self, Read};
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use super::LspError;

const RUNTIME: Duration = Duration::from_secs(10);
const STREAM_CAP: usize = 64 * 1024;

struct OwnedChild {
    child: Child,
    group: Option<u32>,
}
impl OwnedChild {
    fn kill_group(&mut self) -> io::Result<()> {
        if let Some(pid) = self.group {
            // SAFETY: spawn created this group; its direct child is not reaped,
            // so the PID cannot be reused before this signal.
            if unsafe { libc::kill(-(pid as i32), libc::SIGKILL) } != 0 {
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ESRCH) {
                    return Err(error);
                }
            }
            self.group = None;
        }
        Ok(())
    }
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.kill_group();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn nonblocking(stream: &impl AsRawFd) -> io::Result<()> {
    let fd = stream.as_raw_fd();
    // SAFETY: the live owned pipe supplies a valid descriptor. Preserve flags.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags == -1 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn exited_without_reaping(pid: u32) -> io::Result<bool> {
    let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
    // SAFETY: observe only our owned child and keep its PID reserved for group cleanup.
    if unsafe {
        libc::waitid(
            libc::P_PID,
            pid,
            info.as_mut_ptr(),
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    } != 0
    {
        let error = io::Error::last_os_error();
        return if error.kind() == io::ErrorKind::Interrupted {
            Ok(false)
        } else {
            Err(error)
        };
    }
    // SAFETY: successful waitid initializes siginfo, including zero when not exited.
    Ok(unsafe { info.assume_init().si_pid() } != 0)
}

/// Read at most one bounded chunk so neither stream can starve cancellation.
fn capture(stream: &mut impl Read, bytes: &mut Vec<u8>, label: &str) -> Result<bool, LspError> {
    let mut buffer = [0; 8192];
    match stream.read(&mut buffer) {
        Ok(0) => Ok(true),
        Ok(n) => {
            if n > STREAM_CAP.saturating_sub(bytes.len()) {
                return Err(LspError::Process(format!(
                    "analyzer preflight {label} exceeds {STREAM_CAP} bytes"
                )));
            }
            bytes.extend_from_slice(&buffer[..n]);
            Ok(false)
        }
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ) =>
        {
            Ok(false)
        }
        Err(e) => Err(e.into()),
    }
}

pub(super) fn run(
    command: &str,
    args: &[&str],
    workspace: &Path,
    cancel: CancellationToken,
) -> Result<String, LspError> {
    run_until(command, args, workspace, cancel, RUNTIME)
}

fn run_until(
    command: &str,
    args: &[&str],
    workspace: &Path,
    cancel: CancellationToken,
    runtime: Duration,
) -> Result<String, LspError> {
    if cancel.is_cancelled() {
        return Err(LspError::Process("analyzer preflight cancelled".into()));
    }
    let started = Instant::now();
    let child = Command::new(command)
        .args(args)
        .current_dir(workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .map_err(|e| LspError::Process(format!("failed to spawn {command}: {e}")))?;
    let pid = child.id();
    let mut owner = OwnedChild {
        child,
        group: Some(pid),
    };
    let mut stdout = owner.child.stdout.take().expect("piped stdout");
    let mut stderr = owner.child.stderr.take().expect("piped stderr");
    nonblocking(&stdout)?;
    nonblocking(&stderr)?;
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let (mut out_done, mut err_done, mut exited) = (false, false, false);
    loop {
        if cancel.is_cancelled() {
            return Err(LspError::Process("analyzer preflight cancelled".into()));
        }
        if started.elapsed() >= runtime {
            return Err(LspError::Timeout(format!(
                "analyzer preflight exceeded {runtime:?}"
            )));
        }
        if !out_done {
            out_done = capture(&mut stdout, &mut out, "stdout")?;
        }
        if !err_done {
            err_done = capture(&mut stderr, &mut err, "stderr")?;
        }
        if !exited && exited_without_reaping(pid)? {
            exited = true;
            // Descendants can inherit the pipes; stop them before waiting for EOF.
            owner.kill_group()?;
        }
        if exited && out_done && err_done {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let status = owner.child.wait()?;
    if !status.success() {
        return Err(LspError::Process(format!(
            "{command} {} failed (run from {}): {}",
            args.join(" "),
            workspace.display(),
            String::from_utf8_lossy(&err).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out).trim().to_string())
}
