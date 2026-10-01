//! A private X server for tests that exercise X11 protocol code.
use smol::{
    io::{AsyncBufReadExt as _, BufReader},
    process::{Child, Command, Stdio},
};

/// An `Xvfb` instance on a display number the server picked itself.
pub(crate) struct Xvfb {
    process: Child,
    pub(crate) display: String,
}

impl Xvfb {
    /// Starts a server, or returns `None` when `Xvfb` is not installed or cannot start.
    pub(crate) fn start() -> Option<Self> {
        let mut process = Command::new("Xvfb")
            .args([
                "-displayfd",
                "1",
                "-screen",
                "0",
                "64x64x24",
                "-nolisten",
                "tcp",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        // Xvfb writes the display number it chose once the server accepts connections.
        let mut number = String::new();
        let announced = process
            .stdout
            .take()
            .map(BufReader::new)
            .and_then(|mut stdout| smol::block_on(stdout.read_line(&mut number)).ok())
            .is_some_and(|read| read > 0);
        if !announced {
            let _ = process.kill();
            let _ = smol::block_on(process.status());
            return None;
        }
        Some(Self {
            process,
            display: format!(":{}", number.trim()),
        })
    }
}

impl Drop for Xvfb {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = smol::block_on(self.process.status());
    }
}
