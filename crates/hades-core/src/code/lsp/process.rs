//! Ownership of a language server's fresh process group.
use std::io;
use std::time::Duration;
use tokio::process::Child;
use tokio_util::sync::CancellationToken;

use super::LspError;

pub(super) struct ServerProcess {
    pub child: Child,
    group: Option<u32>,
}
impl ServerProcess {
    /// The caller must spawn the child with process_group(0) and kill_on_drop.
    pub fn new(child: Child) -> Self {
        let group = Some(child.id().expect("newly spawned language server has a PID"));
        Self { child, group }
    }

    fn kill_group(&mut self) -> io::Result<()> {
        if let Some(pid) = self.group {
            // SAFETY: the unreaped owned leader reserves this fresh group ID.
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

    async fn observe_exit(&self) -> io::Result<()> {
        let pid = self.group.expect("leader is not yet reaped");
        loop {
            let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
            // SAFETY: observe our child without reaping, preserving its PID until
            // every termination path has signalled the owned process group.
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
                if error.kind() != io::ErrorKind::Interrupted {
                    return Err(error);
                }
            } else if unsafe { info.assume_init().si_pid() } != 0 {
                // SAFETY: successful waitid initialized the siginfo value.
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    pub async fn finish(mut self, stop: CancellationToken) -> Result<(), LspError> {
        let observed = tokio::select! {
            biased;
            _ = stop.cancelled() => Ok(()),
            result = self.observe_exit() => result,
        };
        let killed = self.kill_group();
        if killed.is_err() {
            let _ = self.child.start_kill();
        }
        let reaped = self.child.wait().await;
        observed?;
        killed?;
        reaped?;
        Ok(())
    }
}
impl Drop for ServerProcess {
    fn drop(&mut self) {
        // Signal before Child's kill_on_drop/orphan handling can release its PID.
        let _ = self.kill_group();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::process::Stdio;
    use tokio::process::Command;

    async fn read_pid(path: &Path) -> u32 {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Ok(text) = std::fs::read_to_string(path)
                    && let Ok(pid) = text.parse()
                {
                    break pid;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn unwinding_owner_signals_group_before_child_orphan_handling() {
        let directory = tempfile::tempdir().unwrap();
        let script = "import os,time\nopen('parent','w').write(str(os.getpid()))\nif os.fork()==0: open('descendant','w').write(str(os.getpid()))\ntime.sleep(30)";
        let child = Command::new("/usr/bin/python3")
            .args(["-c", script])
            .current_dir(directory.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let process = ServerProcess::new(child);
        let parent = read_pid(&directory.path().join("parent")).await;
        let descendant = read_pid(&directory.path().join("descendant")).await;
        let task: tokio::task::JoinHandle<()> = tokio::spawn(async move {
            let _owned = process;
            panic!("private owner unwind fixture");
        });
        assert!(task.await.unwrap_err().is_panic());
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let parent_reaped = !Path::new(&format!("/proc/{parent}")).exists();
                let descendant_running =
                    std::fs::read_to_string(format!("/proc/{descendant}/stat"))
                        .is_ok_and(|s| s.split(") ").nth(1).is_some_and(|s| !s.starts_with('Z')));
                if parent_reaped && !descendant_running {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("unwound owner left a process running or its leader unreaped");
    }
}
