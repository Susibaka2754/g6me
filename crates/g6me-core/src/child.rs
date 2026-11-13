//! Child-process plumbing for the ffmpeg stages.
//!
//! Both stages need one thing that is easy to get wrong: **stderr must be
//! drained continuously**, because a child that fills the 64 KiB stderr pipe
//! blocks, and a blocked ffmpeg looks like a hung render. Each [`Proc`]
//! therefore owns a stderr-reader thread and keeps only a bounded tail of it for
//! error messages.

use crate::error::{Error, Result};
use crate::probe::tail;
use std::ffi::OsStr;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

const TAIL_BYTES: usize = 6 * 1024;

/// A running external command with piped data streams plus drained stderr.
pub struct Proc {
    name: &'static str,
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Option<ChildStdout>,
    err: Arc<Mutex<Vec<u8>>>,
    err_reader: Option<JoinHandle<()>>,
    done: bool,
}

/// Drain a child's stderr on a helper thread into a bounded tail.
///
/// The handle is joined once the child has exited: the thread may still be copying the
/// last bytes, and those are the error message.
pub(crate) fn spawn_stderr_reader(
    name: &'static str,
    child: &mut Child,
    err: &Arc<Mutex<Vec<u8>>>,
) -> Option<JoinHandle<()>> {
    let pipe = child.stderr.take()?;
    let sink = err.clone();
    let what = name;
    std::thread::Builder::new()
        .name(format!("g6me-{what}-stderr"))
        .spawn(move || {
            let mut pipe = pipe;
            let mut buf = [0u8; 4096];
            loop {
                match pipe.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let mut s = sink.lock().unwrap_or_else(|e| e.into_inner());
                        s.extend_from_slice(&buf[..n]);
                        // keep a bounded tail
                        if s.len() > TAIL_BYTES * 2 {
                            let cut = s.len() - TAIL_BYTES;
                            s.drain(..cut);
                        }
                    }
                }
            }
        })
        .ok()
}

/// Keep a terminal's Ctrl-C away from the ffmpeg children. It reaches the whole
/// foreground process group, so the children used to die mid-write and the file failed
/// with a broken pipe, while g6me itself is the one that knows how to stop: between
/// blocks, killing its children and removing the partial output.
fn detach_from_console_signals(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP);
    }
}

impl Proc {
    /// Spawn with stdout piped (a producer we read PCM from).
    ///
    /// Generic over the argument type rather than fixed to `String`, because a path
    /// argument must be able to arrive as raw bytes: [`crate::util::ffmpeg_path_arg`]
    /// hands back an `OsString`, and a `String`-only signature would force the caller
    /// through a lossy conversion, which is the exact thing that function exists to
    /// avoid. `&[String]` still fits, so nothing else has to change.
    pub fn spawn_reader<S: AsRef<OsStr>>(
        name: &'static str,
        bin: &Path,
        args: &[S],
    ) -> Result<Self> {
        let mut cmd = Command::new(bin);
        cmd.args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        Self::start(name, cmd, true, false)
    }

    /// Spawn with stdin piped (a consumer we write PCM to).
    pub fn spawn_writer<S: AsRef<OsStr>>(
        name: &'static str,
        bin: &Path,
        args: &[S],
    ) -> Result<Self> {
        let mut cmd = Command::new(bin);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        Self::start(name, cmd, false, true)
    }

    /// Spawn with **both** data pipes open, for a child that sits in the middle
    /// of a chain (ffmpeg writing raw 24-bit PCM that we forward to `flac`).
    pub fn spawn_rw<S: AsRef<OsStr>>(name: &'static str, bin: &Path, args: &[S]) -> Result<Self> {
        let mut cmd = Command::new(bin);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        Self::start(name, cmd, true, true)
    }

    /// Hand the stdout pipe to a caller that wants to move it into its own
    /// forwarding thread. After this, [`Proc::read`] yields EOF.
    pub fn take_stdout(&mut self) -> Option<ChildStdout> {
        self.stdout.take()
    }

    fn start(name: &'static str, mut cmd: Command, out: bool, inp: bool) -> Result<Self> {
        // A stray FFREPORT in the environment would make ffmpeg write a report
        // file next to whatever it touches.
        cmd.env_remove("FFREPORT");
        detach_from_console_signals(&mut cmd);
        let mut child = cmd.spawn().map_err(|e| {
            Error::msg(format!(
                "could not start {} ({}) — install ffmpeg or set $G6ME_FFMPEG: {e}",
                name,
                cmd.get_program().to_string_lossy()
            ))
        })?;
        let err = Arc::new(Mutex::new(Vec::new()));
        let err_reader = spawn_stderr_reader(name, &mut child, &err);
        let stdout = if out { child.stdout.take() } else { None };
        let stdin = if inp { child.stdin.take() } else { None };
        Ok(Self {
            name,
            child,
            stdin,
            stdout,
            err,
            err_reader,
            done: false,
        })
    }

    /// Read up to `buf.len()` bytes of PCM. `Ok(0)` means EOF.
    pub fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self.stdout.as_mut() {
            Some(p) => p.read(buf),
            None => Ok(0),
        }
    }

    pub fn write(&mut self, buf: &[u8]) -> std::io::Result<()> {
        match self.stdin.as_mut() {
            Some(p) => p.write_all(buf),
            None => Ok(()),
        }
    }

    /// Close our end of the pipe so the child sees EOF and finalizes the file.
    pub fn close_input(&mut self) {
        if let Some(mut s) = self.stdin.take() {
            let _ = s.flush();
        }
    }

    /// Finish and check the exit status. Consumes the stderr tail into the error.
    pub fn finish(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        self.close_input();
        // Drop stdout too, or a producer never sees EOF.
        let _ = self.stdout.take();
        let status = self.child.wait().map_err(Error::Io)?;
        self.done = true;
        self.join_err_reader();
        if status.success() {
            return Ok(());
        }
        let msg = self.take_err();
        Err(Error::ChildFailed {
            stage: self.name,
            code: status.code().unwrap_or(-1),
            msg: if msg.trim().is_empty() {
                "(no message on stderr)".to_string()
            } else {
                msg
            },
        })
    }

    pub fn take_err(&self) -> String {
        let g = self.err.lock().unwrap_or_else(|e| e.into_inner());
        String::from_utf8_lossy(&g).into_owned()
    }

    pub fn err_tail(&self) -> String {
        let g = self.err.lock().unwrap_or_else(|e| e.into_inner());
        tail(&g, 800)
    }

    /// Non-blocking check whether the child already exited (used to notice a
    /// decode failure early instead of reading silence to the end).
    pub fn try_exit_code(&mut self) -> Option<i32> {
        match self.child.try_wait() {
            Ok(Some(st)) => Some(st.code().unwrap_or(-1)),
            _ => None,
        }
    }

    pub fn kill(&mut self) {
        if !self.done {
            let _ = self.child.kill();
            let _ = self.child.wait();
            self.done = true;
            self.join_err_reader();
        }
    }

    /// The child has exited, so its stderr reaches EOF and the reader returns.
    fn join_err_reader(&mut self) {
        if let Some(h) = self.err_reader.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        // Never leave an orphan behind: a child still writing into a pipe we
        // dropped would block on a full pipe forever.
        self.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::Tools;

    #[test]
    fn producer_reads_all_bytes() {
        let t = Tools::discover();
        if !t.is_usable() {
            eprintln!("skipping: no ffmpeg");
            return;
        }
        let args: Vec<String> = [
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.1",
            "-f",
            "f64le",
            "-c:a",
            "pcm_f64le",
            "-",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut p = Proc::spawn_reader("test", &t.ffmpeg, &args).unwrap();
        let mut got = Vec::new();
        let mut buf = [0u8; 9999];
        loop {
            let n = p.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            got.extend_from_slice(&buf[..n]);
        }
        p.finish().unwrap();
        // 0.1 s @ 48 kHz mono f64
        assert_eq!(got.len(), (0.1 * 48000.0) as usize * 8);
    }

    #[test]
    fn failure_carries_stderr() {
        let t = Tools::discover();
        if !t.is_usable() {
            return;
        }
        let args: Vec<String> = [
            "-v",
            "error",
            "-nostdin",
            "-i",
            "/nonexistent/nope.mka",
            "-f",
            "f64le",
            "-",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut p = Proc::spawn_reader("decode", &t.ffmpeg, &args).unwrap();
        let e = p.finish().unwrap_err();
        assert!(matches!(
            e,
            Error::ChildFailed {
                stage: "decode",
                ..
            }
        ));
        assert!(!e.to_string().is_empty());
    }
}
