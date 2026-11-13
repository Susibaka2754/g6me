//! ffmpeg / ffprobe discovery and capability probing.
//!
//! The app shells out to ffmpeg rather than linking libav*: TrueHD decoding
//! needs a GPL build, and a subprocess keeps the Windows and Arch builds
//! dependency-free (no ABI-matching libav, no license taint in our binary).
//!
//! Every capability is established **functionally**, by running the exact filter
//! string we would emit on a 50 ms synthetic clip, not by grepping `--help`. A
//! build that lists `soxr` but rejects our `dither_method` spelling fails the
//! probe, and the stage then degrades to passthrough with the reason recorded.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;

/// Path to the two external binaries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl Default for Tools {
    fn default() -> Self {
        Self {
            ffmpeg: PathBuf::from(exe("ffmpeg")),
            ffprobe: PathBuf::from(exe("ffprobe")),
        }
    }
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

impl Tools {
    /// `$G6ME_FFMPEG` / `$G6ME_FFPROBE` win; otherwise a PATH search, and
    /// otherwise bare names so the child spawn produces the OS's own error.
    pub fn discover() -> Self {
        let ffmpeg = std::env::var_os("G6ME_FFMPEG")
            .map(PathBuf::from)
            .or_else(|| which("ffmpeg"))
            .unwrap_or_else(|| PathBuf::from(exe("ffmpeg")));
        let ffprobe = std::env::var_os("G6ME_FFPROBE")
            .map(PathBuf::from)
            .or_else(|| which("ffprobe"))
            .unwrap_or_else(|| {
                // Derive from the ffmpeg directory when ffprobe is not on PATH
                // (common in bundled Windows builds that ship both together).
                if ffmpeg.parent().is_some() {
                    ffmpeg.with_file_name(exe("ffprobe"))
                } else {
                    PathBuf::from(exe("ffprobe"))
                }
            });
        Self { ffmpeg, ffprobe }
    }

    pub fn is_usable(&self) -> bool {
        self.ffmpeg.is_file()
    }

    /// Human-readable location for the GUI's status line.
    pub fn describe(&self) -> String {
        format!("{} / {}", self.ffmpeg.display(), self.ffprobe.display())
    }
}

/// Minimal PATH lookup (no `which` crate).
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.BAT;.CMD".into())
            .split(';')
            .map(|s| s.to_ascii_lowercase())
            .collect()
    } else {
        vec![String::new()]
    };
    for dir in std::env::split_paths(&path) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        for ext in &exts {
            let cand = dir.join(format!("{name}{ext}"));
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}

/// What this ffmpeg build can actually do for us.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Caps {
    pub ffmpeg_version: String,
    pub soxr: bool,
    pub afir: bool,
    pub flac: bool,
    /// True when lavfi is available, which is what the probes below need.
    pub lavfi: bool,
    /// This ffmpeg was measured writing a genuine, lossless 24-bit FLAC (see
    /// [`probe_flac24`]). It is what the default output depth rests on, so it is
    /// checked by round-tripping bytes rather than inferred from the version.
    pub flac24: bool,
    pub notes: Vec<String>,
}

impl Caps {
    /// Resampling is possible at all (soxr present; the rubato fallback is
    /// reported separately by [`crate::stages::resample_backend`]).
    pub fn can_resample(&self) -> bool {
        self.soxr
    }

    pub fn summary(&self) -> String {
        format!(
            "ffmpeg {} — soxr:{} afir:{} flac:{} flac24:{} lavfi:{}",
            self.ffmpeg_version,
            yes_no(self.soxr),
            yes_no(self.afir),
            yes_no(self.flac),
            yes_no(self.flac24),
            yes_no(self.lavfi)
        )
    }

    /// The fact behind the default output depth, without a leading label: callers
    /// print their own heading, and a label in here came out doubled in `--caps`.
    pub fn flac_summary(&self) -> String {
        if self.flac24 {
            "verified lossless (encoded bps=24, decoded bytes came back identical)".to_string()
        } else {
            "NOT verified on this ffmpeg — use 16-bit FLAC or WAV (see the notes)".to_string()
        }
    }
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

/// Run the functional capability probes. Independent children, so they run
/// concurrently — this is on the GUI's startup path.
pub fn probe_caps(tools: &Tools) -> Caps {
    let mut caps = Caps {
        ffmpeg_version: version(tools).unwrap_or_else(|e| format!("unknown ({e})")),
        ..Caps::default()
    };
    // lavfi first: three of the four probes need it. If it is missing (a
    // deliberately minimal build) the probes are meaningless, so report that.
    caps.lavfi = probe_lavfi(tools);
    if !caps.lavfi {
        caps.notes
            .push("lavfi unavailable; capabilities assumed absent".into());
        return caps;
    }

    let mut flac24_note = None;
    std::thread::scope(|s| {
        let soxr = s.spawn(|| probe_soxr(tools));
        let afir = s.spawn(|| probe_afir(tools));
        let flac = s.spawn(|| probe_flac(tools));
        let flac24 = s.spawn(|| probe_flac24(tools));
        caps.soxr = soxr.join().unwrap_or(false);
        caps.afir = afir.join().unwrap_or(false);
        caps.flac = flac.join().unwrap_or(false);
        match flac24.join() {
            Ok(None) => caps.flac24 = true,
            // The reason matters more than the flag: a build that fails this writes
            // a file labelled 24-bit while losing bits, so name what it did.
            Ok(Some(note)) => flac24_note = Some(note),
            Err(_) => flac24_note = Some("the 24-bit FLAC probe panicked".into()),
        }
    });
    if let Some(n) = flac24_note {
        caps.notes.push(format!("24-bit FLAC: {n}"));
    }

    if !caps.soxr {
        caps.notes.push(
            "no libsoxr (or dither rejected): resample stage disabled, passthrough only".into(),
        );
    }
    if !caps.afir {
        caps.notes
            .push("no afir: IR headphone calibration unavailable".into());
    }

    caps
}

/// Can this ffmpeg write a **genuine 24-bit FLAC**?
///
/// Established by doing it and checking the bytes came back, not by asking. The
/// encoder advertises only `s16 s32` as input sample formats and derives the
/// stream's bit depth from what it is fed, which is subtle enough to be worth
/// proving on the machine that will use it: fed a true 32-bit signal it writes a
/// `bps=24` stream and discards the low byte of every sample, while fed 24-bit
/// samples it is lossless. A `--version` or `-codecs` check cannot tell those
/// apart, and only the second one is what the default output depth relies on.
///
/// `None` means a synthetic 24-bit block went through the renderer's own encode
/// command line and decoded back byte-identical; `Some(note)` says why 24-bit
/// FLAC cannot be trusted here, and `--caps` shows it instead of a bare "no".
pub fn probe_flac24(tools: &Tools) -> Option<String> {
    let dir = match probe_dir("flac24") {
        Ok(d) => d,
        Err(e) => return Some(e),
    };
    let raw = dir.join("probe.s24");
    let enc = dir.join("probe.flac");
    let back = dir.join("back.s24");
    let result = (|| -> std::result::Result<(), String> {
        // 0.1 s of stereo 24-bit PCM. The stride is odd and coprime with 256, so
        // the low byte of most samples is non-zero: an encoder that throws the low
        // 8 bits away — or rounds toward zero instead of quantizing at 24-bit —
        // cannot survive the byte comparison at the end of this probe.
        const N: usize = 4800 * 2;
        let mut pcm = Vec::with_capacity(N * 3);
        for i in 0..N {
            let v: i32 = ((i as i32) * 149) % (1 << 23);
            pcm.extend_from_slice(&v.to_le_bytes()[..3]);
        }
        std::fs::write(&raw, &pcm).map_err(|e| format!("probe write: {e}"))?;
        let raw_s = raw.display().to_string();
        let enc_s = enc.display().to_string();
        let back_s = back.display().to_string();
        // Exactly the renderer's encode shape: depth asked for explicitly, level
        // from the default config.
        ffmpeg_ok(
            tools,
            &[
                "-f",
                "s24le",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-i",
                &raw_s,
                "-c:a",
                "flac",
                "-compression_level",
                "5",
                "-sample_fmt",
                "s32",
                "-y",
                &enc_s,
            ],
        )?;
        let info = flac_streaminfo(&enc).ok_or("ffmpeg wrote no readable FLAC stream info")?;
        if info.2 != 24 {
            return Err(format!("ffmpeg wrote {}-bit FLAC, not 24-bit", info.2));
        }
        ffmpeg_ok(tools, &["-i", &enc_s, "-f", "s24le", "-y", &back_s])?;
        let got = std::fs::read(&back).map_err(|e| format!("probe decode read: {e}"))?;
        if got != pcm {
            let diff = pcm.iter().zip(&got).filter(|(a, b)| a != b).count();
            return Err(format!(
                "24-bit FLAC is not lossless on this build: {diff} of {} bytes differ                  after a decode",
                pcm.len()
            ));
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result.err()
}

/// Run ffmpeg to completion, reporting its stderr as the reason on failure.
fn ffmpeg_ok(tools: &Tools, args: &[&str]) -> std::result::Result<(), String> {
    let out = Command::new(&tools.ffmpeg)
        .arg("-v")
        .arg("error")
        .arg("-nostdin")
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("could not run ffmpeg: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "ffmpeg failed: {}",
            tail(&out.stderr, 300).replace('\n', " ")
        ))
    }
}

/// Minimal FLAC STREAMINFO reader: sample rate, channels, bits per sample.
///
/// Just enough to prove an encode really is 24-bit without depending on ffprobe.
/// Layout (RFC 9639): `fLaC`, then a metadata block header, then a 34-byte
/// STREAMINFO whose bytes 10..18 pack rate(20) | channels-1(3) | bps-1(5) |
/// total-samples(36).
pub fn flac_streaminfo(path: &Path) -> Option<(u32, u16, u16)> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 4 + 4 + 34 || &b[0..4] != b"fLaC" {
        return None;
    }
    let block_type = b[4] & 0x7f;
    let len = ((b[5] as usize) << 16) | ((b[6] as usize) << 8) | b[7] as usize;
    if block_type != 0 || len < 34 {
        return None;
    }
    let body = &b[8..8 + 34];
    let mut v = 0u64;
    for byte in &body[10..18] {
        v = (v << 8) | *byte as u64;
    }
    let rate = (v >> 44) as u32;
    let channels = ((v >> 41) & 0x7) as u16 + 1;
    let bps = ((v >> 36) & 0x1f) as u16 + 1;
    if rate == 0 {
        return None;
    }
    Some((rate, channels, bps))
}

fn version(tools: &Tools) -> Result<String> {
    let out = Command::new(&tools.ffmpeg)
        .args(["-version"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map_err(|e| Error::msg(format!("ffmpeg -version: {e}")))?;
    let txt = String::from_utf8_lossy(&out.stdout);
    Ok(txt
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(2))
        .unwrap_or("unknown")
        .to_string())
}

/// Run a tiny graph to /dev/null-equivalent and report success + byte count.
fn run_probe(tools: &Tools, args: &[&str], expect_min_bytes: u64) -> bool {
    let mut cmd = Command::new(&tools.ffmpeg);
    cmd.arg("-v").arg("error").arg("-nostdin").args(args);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => return false,
    };
    // Drain stdout while the child runs so a full pipe cannot deadlock it.
    let mut pipe = child.stdout.take();
    let mut total: u64 = 0;
    if let Some(p) = pipe.as_mut() {
        let mut buf = vec![0u8; 1 << 16];
        loop {
            match std::io::Read::read(p, &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => total += n as u64,
            }
        }
    }
    let status = child.wait_with_output();
    match status {
        Ok(o) => o.status.success() && total >= expect_min_bytes,
        Err(_) => false,
    }
}

const SINE48: &str = "sine=frequency=997:sample_rate=48000:duration=0.05";
const SINE_LOW: &str = "sine=frequency=997:sample_rate=44100:duration=0.05";

fn probe_lavfi(tools: &Tools) -> bool {
    run_probe(
        tools,
        &[
            "-f",
            "lavfi",
            "-i",
            SINE48,
            "-f",
            "f64le",
            "-c:a",
            "pcm_f64le",
            "-",
        ],
        1000,
    )
}

fn probe_soxr(tools: &Tools) -> bool {
    // The exact filter string the app emits, including the dither spelling.
    run_probe(
        tools,
        &[
            "-f",
            "lavfi",
            "-i",
            SINE48,
            "-af",
            "aresample=44100:resampler=soxr:precision=33:dither_method=low_shibata",
            "-f",
            "f64le",
            "-c:a",
            "pcm_f64le",
            "-",
        ],
        800,
    )
}

fn probe_afir(tools: &Tools) -> bool {
    run_probe(
        tools,
        &[
            "-f",
            "lavfi",
            "-i",
            SINE48,
            "-f",
            "lavfi",
            "-i",
            SINE_LOW,
            "-filter_complex",
            "[0:a]aformat=sample_fmts=dbl[am];[1:a]aformat=sample_fmts=dbl:channel_layouts=mono[ir];\
             [am][ir]afir=irfmt=mono:irnorm=-1:wet=1:minp=4096:maxp=4096:precision=double[o]",
            "-map",
            "[o]",
            "-f",
            "f64le",
            "-c:a",
            "pcm_f64le",
            "-",
        ],
        1000,
    )
}

fn probe_flac(tools: &Tools) -> bool {
    run_probe(
        tools,
        &[
            "-f", "lavfi", "-i", SINE48, "-c:a", "flac", "-f", "flac", "-",
        ],
        100,
    )
}

/// What ffprobe reported about one audio stream.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AudioInfo {
    pub stream_index: usize,
    pub codec_name: String,
    pub profile: Option<String>,
    pub sample_rate: u32,
    pub channels: usize,
    pub layout: Option<String>,
    /// Bits per raw sample when the demuxer/decoder knows it (24 for TrueHD).
    pub bits: Option<u32>,
    pub duration: f64,
}

impl AudioInfo {
    pub fn describe(&self) -> String {
        format!(
            "{} {} ch {} Hz {}",
            self.codec_name,
            self.channels,
            self.sample_rate,
            self.layout
                .clone()
                .unwrap_or_else(|| "layout-unknown".into())
        )
    }
}

#[derive(Deserialize)]
struct ProbeJson {
    streams: Vec<RawStream>,
    #[serde(default)]
    format: Option<RawFormat>,
}

#[derive(Deserialize)]
struct RawStream {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    codec_name: Option<String>,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    sample_rate: Option<String>,
    #[serde(default)]
    channels: Option<usize>,
    #[serde(default)]
    channel_layout: Option<String>,
    #[serde(default)]
    bits_per_raw_sample: Option<String>,
    #[serde(default)]
    duration: Option<String>,
}

#[derive(Deserialize, Default)]
struct RawFormat {
    #[serde(default)]
    duration: Option<String>,
}

/// Probe the `pick`-th audio stream (0 = first).
pub fn probe_audio(tools: &Tools, path: &Path, pick: usize) -> Result<AudioInfo> {
    let out = Command::new(&tools.ffprobe)
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
        ])
        .arg("-select_streams")
        .arg(format!("a:{pick}"))
        // The user's own file name, so ffprobe obeys the same rule as ffmpeg: it parses
        // `[protocol:]target` before it looks at the filesystem, and answers "Protocol
        // not found" for a relative name holding a colon. Measured: `ffprobe a:b.wav`
        // fails that way while `ffprobe ./a:b.wav` reads the file.
        .arg(crate::util::ffmpeg_path_arg(path))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| Error::InputUnreadable {
            path: path.to_path_buf(),
            msg: format!(
                "could not run {} ({}) — ffprobe is required to read channel layout",
                tools.ffprobe.display(),
                e
            ),
        })?;
    if !out.status.success() {
        return Err(Error::ProbeFailed {
            path: path.display().to_string(),
            msg: tail(&out.stderr, 400),
        });
    }
    let parsed: ProbeJson =
        serde_json::from_slice(&out.stdout).map_err(|e| Error::ProbeFailed {
            path: path.display().to_string(),
            msg: format!("unparsable ffprobe json: {e}"),
        })?;
    let s = parsed
        .streams
        .into_iter()
        .next()
        .ok_or_else(|| Error::NoAudio(path.to_path_buf()))?;
    Ok(AudioInfo {
        stream_index: s.index.unwrap_or(0),
        codec_name: s.codec_name.unwrap_or_else(|| "unknown".into()),
        profile: s.profile,
        sample_rate: s.sample_rate.and_then(|v| v.parse().ok()).unwrap_or(48000),
        channels: s.channels.unwrap_or(0),
        layout: s.channel_layout.filter(|l| l != "unknown" && !l.is_empty()),
        bits: s
            .bits_per_raw_sample
            .as_deref()
            .and_then(|v| v.parse().ok()),
        duration: s
            .duration
            .as_deref()
            .or(parsed.format.as_ref().and_then(|f| f.duration.as_deref()))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0),
    })
}

/// A unique scratch directory for a functional probe; the caller removes it.
pub(crate) fn probe_dir(tag: &str) -> std::result::Result<PathBuf, String> {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("g6me-probe-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("probe dir {}: {e}", dir.display()))?;
    Ok(dir)
}

pub(crate) fn tail(bytes: &[u8], max: usize) -> String {
    let s = String::from_utf8_lossy(bytes);
    if s.len() <= max {
        return s.into_owned();
    }
    // A byte offset can land inside a multi-byte character (any non-ASCII path in an
    // ffmpeg message), and slicing there panics.
    let mut cut = s.len() - max;
    while !s.is_char_boundary(cut) {
        cut += 1;
    }
    s[cut..].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_default_to_bare_names_when_nothing_is_installed() {
        let t = Tools::default();
        assert!(t.ffmpeg.to_string_lossy().contains("ffmpeg"));
    }

    /// The default output depth rests on this, so it is proven by encoding and
    /// decoding rather than assumed. Skips only when there is no ffmpeg at all.
    #[test]
    fn flac24_probe_passes_on_a_working_ffmpeg() {
        let tools = Tools::default();
        if version(&tools).is_err() {
            eprintln!("skipping: no ffmpeg on PATH");
            return;
        }
        if let Some(note) = probe_flac24(&tools) {
            panic!("this ffmpeg cannot round-trip a 24-bit FLAC: {note}");
        }
    }

    /// A file that is not FLAC at all, and a truncated header, must not parse.
    #[test]
    fn streaminfo_rejects_rubbish() {
        let dir = probe_dir("flaclike").unwrap();
        let bogus = dir.join("notflac.flac");
        std::fs::write(&bogus, b"RIFF____data").unwrap();
        assert_eq!(flac_streaminfo(&bogus), None, "wrong magic");
        std::fs::write(&bogus, b"fLaC").unwrap();
        assert_eq!(flac_streaminfo(&bogus), None, "truncated");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tail_bounds() {
        assert_eq!(tail(b"hello", 10), "hello");
        assert_eq!(tail(b"0123456789abcdef", 4), "cdef");
        // Every cut position through two-byte characters: none may panic, and the result
        // never exceeds the budget.
        let s = "Музыка/трек.flac: Invalid data".as_bytes();
        for max in 1..s.len() {
            let t = tail(s, max);
            assert!(t.len() <= max, "{max}: {t}");
            assert!(s.ends_with(t.as_bytes()));
        }
    }

    /// Real probe against the local ffmpeg, skipped when it is absent.
    #[test]
    fn probe_local_ffmpeg() {
        let t = Tools::discover();
        if !t.is_usable() {
            eprintln!("skipping: no ffmpeg");
            return;
        }
        let c = probe_caps(&t);
        eprintln!("{}", c.summary());
        assert!(!c.ffmpeg_version.is_empty());
        assert!(c.lavfi, "lavfi probe failed on a stock build");
    }
}
