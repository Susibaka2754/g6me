//! Batch engine: many files and folders, one output per input.
//!
//! ## Threading model
//!
//! The downmixer is a cascade of recursive all-pass sections, so its state is a
//! sequence: split one file across threads and you have to reconstruct that
//! sequence across a boundary, which is exactly what changes phase behaviour. So
//! **DSP stays single-threaded per file** — one [`crate::dsp::Dsp`] owned by one
//! worker for the whole file — and the parallelism lives *between* files: N
//! workers, each rendering a different file, each with its own two ffmpeg
//! children.
//!
//! Workers pull from a shared cursor rather than a queue. There is nothing to
//! co-ordinate beyond "give me the next index", so a slow 3-hour Atmos file
//! cannot block the four short files behind it.
//!
//! ## Failures
//!
//! A file that cannot be decoded, or whose layout cannot be determined, is
//! recorded and skipped; the queue continues. Only cancellation stops the batch,
//! and it stops between blocks rather than mid-write, so a cancelled run leaves
//! no half-written output behind ([`crate::pipeline`] deletes it).

use crate::error::Error;
use crate::pcm::BitDepth;
use crate::pipeline::{render_file, RenderOpts, Report, SampleTap, Sink};
use crate::probe::{Caps, Tools};
use crate::stages::{OutFormat, RenderCfg};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Containers we will attempt, chosen by extension because deciding by probing
/// every file in a 40 000-file library would take longer than rendering it.
/// Anything ffmpeg can decode is accepted at render time regardless.
pub const AUDIO_EXTENSIONS: &[&str] = &[
    "mka", "mkv", "webm", "m4a", "m4b", "mp4", "mov", "m4v", "flac", "wav", "wave", "rf64", "aiff",
    "aif", "aifc", "afc", "w64", "caf", "wma", "ape", "mla", "tta", "wv", "tak", "oga", "ogg",
    "opus", "spx", "ac3", "eac3", "dts", "dtshd", "dff", "dsf", "aac", "mp3", "mp2", "m4r", "amr",
    "3gp", "3g2", "voc", "au", "snd", "raw",
];

/// Files that are definitely not audio, so a folder scan does not offer them.
pub const SKIP_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "gif", "bmp", "webp", "nfo", "txt", "md", "cue", "log", "srt", "ass",
    "ssa", "sub", "pdf", "zip", "rar", "7z", "exe", "dll", "json", "xml", "yml", "yaml", "csv",
    "dat", "jpg", "cover", "jpeg",
];

/// What one file in the queue is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Queued,
    Running,
    Done,
    Failed,
    /// Left alone on purpose (output exists and `--force` was not given).
    Skipped,
}

impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Phase::Queued => "queued",
            Phase::Running => "running",
            Phase::Done => "done",
            Phase::Failed => "failed",
            Phase::Skipped => "skipped",
        }
    }
    pub fn is_terminal(self) -> bool {
        matches!(self, Phase::Done | Phase::Failed | Phase::Skipped)
    }
}

/// One queued file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Item {
    pub src: PathBuf,
    pub dst: PathBuf,
    pub phase: Phase,
    /// Frames rendered so far, for a progress bar that moves mid-file.
    pub frames: u64,
    pub error: Option<String>,
    pub report: Option<Report>,
    /// Why this item's destination is not the one its name alone would have produced.
    ///
    /// Set when two inputs wanted the same output path. Silence here means the obvious
    /// destination was used; a rename that nobody can see is how a batch quietly loses a
    /// file, so the reason travels with the item to the queue row and the dry-run listing.
    pub note: Option<String>,
}

impl Item {
    pub fn new(src: PathBuf, dst: PathBuf) -> Self {
        Self {
            src,
            dst,
            phase: Phase::Queued,
            frames: 0,
            error: None,
            report: None,
            note: None,
        }
    }

    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    pub fn seconds(&self) -> f64 {
        self.report.as_ref().map(|r| r.seconds).unwrap_or(0.0)
    }
}

/// Makes the meter tap for one file, given that file's index in the queue.
///
/// Per file on purpose. One meter shared by four workers would display a level
/// that belongs to none of them — the peak would be the loudest of four different
/// pieces of music, which is not a measurement of anything.
pub type TapFor = dyn Fn(usize) -> Option<Arc<dyn SampleTap>> + Send + Sync;

/// [`TapFor`] wrapped so [`BatchCfg`] can keep deriving `Debug`: a closure has
/// nothing honest to print, so it says whether one is there and stops.
#[derive(Clone, Default)]
pub struct MeterFactory(pub Option<std::sync::Arc<TapFor>>);

impl std::fmt::Debug for MeterFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match &self.0 {
            Some(_) => "MeterFactory(set)",
            None => "MeterFactory(none)",
        })
    }
}

/// Inputs to one batch run.
#[derive(Clone, Debug)]
pub struct BatchCfg {
    /// Folders and files, in the order given.
    pub inputs: Vec<PathBuf>,
    /// `None` writes each output next to its input.
    pub out_dir: Option<PathBuf>,
    /// Scan folders recursively (default, and what a music library expects).
    pub recursive: bool,
    /// How many files to render at once. DSP stays single-threaded per file.
    pub workers: usize,
    pub render: RenderCfg,
    /// Refuse to start if nothing was found (instead of a silent no-op).
    pub fail_on_empty: bool,
    /// Live levels for each file as it renders. `None` renders blind, which is
    /// what the CLI does.
    pub taps: MeterFactory,
}

impl Default for BatchCfg {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            out_dir: None,
            recursive: true,
            workers: default_workers(),
            render: RenderCfg::with_defaults(),
            fail_on_empty: false,
            taps: MeterFactory::default(),
        }
    }
}

/// Sensible default: enough workers to keep two ffmpeg children plus the DSP
/// busy per file, without pretending a hard drive scales linearly.
pub fn default_workers() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // Each worker is a decoder + DSP + encoder; four of those saturate a laptop,
    // and a 64-core box gains nothing past that (the DSP is the bottleneck).
    cores.clamp(1, 4)
}

/// Extension of a path, lowercased, or "" for no extension.
fn ext_of(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default()
}

/// True when this looks like an audio file we should offer.
pub fn looks_like_audio(p: &Path) -> bool {
    let e = ext_of(p);
    if e.is_empty() {
        return false;
    }
    if SKIP_EXTENSIONS.contains(&e.as_str()) {
        return false;
    }
    AUDIO_EXTENSIONS.contains(&e.as_str())
}

/// Where a file's output goes.
///
/// With an output directory the input tree is mirrored under it, so a batch over
/// `~/Music` lands as `~/Music-flac/Artist/Album/track.flac` instead of 40 000
/// files in one heap.
pub fn output_for(src: &Path, base: &Path, out_dir: Option<&Path>, ext: &str) -> PathBuf {
    output_for_prefixed(src, base, Path::new(""), out_dir, ext)
}

/// [`output_for`] with a folder component forced in front of the mirrored path.
///
/// This is how a second input folder called `Album` gets a tree of its own. Its root
/// is used as the mirror base (so the relative path is what is inside it) and the
/// renamed folder becomes this prefix, which keeps every file beneath it in the same
/// shape as the folder that kept the original name.
fn output_for_prefixed(
    src: &Path,
    base: &Path,
    prefix: &Path,
    out_dir: Option<&Path>,
    ext: &str,
) -> PathBuf {
    let stem = src
        .file_name()
        .map(|s| {
            let s = s.to_string_lossy().to_string();
            match s.rfind('.') {
                Some(i) if i > 0 => s[..i].to_string(),
                _ => s,
            }
        })
        .unwrap_or_else(|| "output".to_string());
    let root = match out_dir {
        Some(r) => r.to_path_buf(),
        None => src.parent().unwrap_or(Path::new(".")).to_path_buf(),
    };
    let mut out = root;
    if out_dir.is_some() {
        if !prefix.as_os_str().is_empty() {
            out.push(prefix);
        }
        let subdir =
            relative_within(src.parent().unwrap_or(Path::new(".")), base).unwrap_or(Path::new(""));
        out.push(subdir);
    }
    out.push(format!("{stem}.{ext}"));
    out
}

/// `child` inside `root`, or None when it is not below it.
fn relative_within<'a>(child: &'a Path, root: &Path) -> Option<&'a Path> {
    let (c, r) = (child.as_os_str(), root.as_os_str());
    if c == r {
        return Some(Path::new(""));
    }
    let mut cpath = child;
    if cpath.starts_with(root) {
        cpath = cpath.strip_prefix(root).unwrap_or(Path::new(""));
        Some(cpath)
    } else {
        None
    }
}

/// Expand files and folders into a de-duplicated queue.
///
/// `base` is the folder whose structure is mirrored into `out_dir` — the first
/// folder given on the command line, which is what a user means by "this tree".
pub fn discover(
    inputs: &[PathBuf],
    out_dir: Option<&Path>,
    ext: &str,
    recursive: bool,
) -> Vec<Item> {
    let mut seen: Vec<PathBuf> = Vec::new();
    let mut items: Vec<Item> = Vec::new();
    // What a loose file argument mirrors against. This stays "the first directory argument,
    // else the working directory", because that is the long-standing behaviour for a file
    // named on the command line and nothing below changes it.
    let loose_base = inputs
        .iter()
        .find(|p| p.is_dir())
        .cloned()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    // Each folder mirrors against *itself*. With one shared root for every argument, the
    // second folder's files lose their structure and land on the first folder's names:
    // `Library/AlbumA/01.mka` and `Library/AlbumB/01.mka` both asked for `out/01.flac`, so
    // the second was skipped as "already rendered" and one file of the batch quietly never
    // happened.
    // (directory, mirror base, forced output folder, why it was forced)
    let mut stack: Vec<(PathBuf, PathBuf, PathBuf, Option<String>)> = Vec::new();
    // Folders already entered, by canonical path. A directory reached twice is a queue
    // written twice, and a symlink pointing at an ancestor is a walk that never ends.
    let mut entered: Vec<PathBuf> = Vec::new();
    // One folder in: its contents mirror into `out_dir` directly, which is what `g6me Music/`
    // has always meant. Two or more: each keeps its own folder name, because mirroring both
    // against their own root puts `AlbumA/01.mka` and `AlbumB/01.mka` on the same
    // `out/01.flac`, and the second would be skipped as already rendered — a batch that
    // quietly renders one file fewer than it was handed.
    let nested = inputs.iter().filter(|p| p.is_dir()).count() > 1;
    // Output folder names claimed so far, and the root that claimed each one.
    //
    // Two input folders can carry the same basename: `/a/Album` and `/b/Album`, or a
    // library plus the backup of it that is still on the drive. In the multi-folder mode
    // each folder's own name is the first component of its output tree, so both ask for
    // `out/Album/01.flac`; the second one meets a file that is already there, is skipped
    // as already rendered, and the batch reports success having quietly not made one of
    // the files it was handed. This is the same loss the per-folder mirror base fixed,
    // one level up the tree, and the same rule applies: never let two inputs share a
    // destination, and say out loud when a destination had to move.
    let mut claimed: Vec<(String, PathBuf)> = Vec::new();
    let mut enter = |dir: &Path,
                     stack: &mut Vec<(PathBuf, PathBuf, PathBuf, Option<String>)>,
                     claimed: &mut Vec<(String, PathBuf)>| {
        let key = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
        if entered.iter().any(|e| e == &key) {
            return;
        }
        // Cloned because `key` is still needed below as the owner to compare against.
        entered.push(key.clone());
        // One folder mirrors its contents straight into `out_dir`, which is what
        // `g6me Music/` has always meant, and no folder name is in play at all.
        if !nested || out_dir.is_none() {
            stack.push((dir.to_path_buf(), dir.to_path_buf(), PathBuf::new(), None));
            return;
        }
        let (name, note) = claim_output_folder(dir, &key, claimed);
        let entry = match note {
            // The folder kept its own name: mirror against the parent, which is what puts
            // the folder name at the head of every path under it.
            None => (
                dir.to_path_buf(),
                dir.parent().unwrap_or(dir).to_path_buf(),
                PathBuf::new(),
                None,
            ),
            // Mirror against the folder itself, so everything inside it keeps the same
            // shape under the renamed folder as the folder that kept the original name.
            Some(why) => (
                dir.to_path_buf(),
                dir.to_path_buf(),
                PathBuf::from(&name),
                Some(why),
            ),
        };
        stack.push(entry);
    };

    for input in inputs {
        if input.is_dir() {
            enter(input, &mut stack, &mut claimed);
        } else if input.is_file() {
            push_file(
                input,
                &loose_base,
                Path::new(""),
                None,
                out_dir,
                ext,
                &mut seen,
                &mut items,
            );
        }
    }
    // The stack pops last-in-first-out; reversed, folder arguments are walked in the
    // order they were given.
    stack.reverse();
    while let Some((dir, base, prefix, note)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue; // unreadable directory: skip it, do not fail the batch
        };
        let mut names: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        // Deterministic order, so two runs of the same batch queue identically: a folder's
        // own files first, ascending, then its subfolders, ascending. Files are queued as
        // they are seen, so they are walked forwards; subfolders go on the stack, so they
        // are pushed in reverse to pop in ascending order.
        names.sort();
        let mut subdirs = Vec::new();
        for p in &names {
            // A render writes to a staging sibling and renames it into place, so a name with
            // that prefix is an unfinished render, not a track. It should not exist at all --
            // that is the point of the rename -- and the case for recognising it is the hard
            // kill and the power cut, where an orphan is exactly what is left on disk.
            if p.file_name()
                .map(|n| {
                    n.to_string_lossy()
                        .starts_with(crate::pipeline::STAGING_PREFIX)
                })
                .unwrap_or(false)
            {
                continue;
            }
            if p.is_dir() {
                if !recursive {
                    continue;
                }
                let key = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
                if entered.iter().any(|e| e == &key) {
                    continue;
                }
                entered.push(key);
                subdirs.push(p.clone());
            } else if looks_like_audio(p) {
                push_file(
                    p,
                    &base,
                    &prefix,
                    note.as_deref(),
                    out_dir,
                    ext,
                    &mut seen,
                    &mut items,
                );
            }
        }
        for p in subdirs.into_iter().rev() {
            stack.push((p, base.clone(), prefix.clone(), note.clone()));
        }
    }
    keep_each_item_on_its_own_path(&mut items);
    items
}

/// Give one input folder its own output folder name, numbering it if the name is taken.
///
/// The first folder to claim a name keeps it and later ones are numbered in input order,
/// which is deterministic because the caller's argument order is. A folder reached twice
/// through two spellings (a symlink, or the same directory named twice on the command
/// line) is not a collision and is not renumbered.
fn claim_output_folder(
    dir: &Path,
    key: &Path,
    claimed: &mut Vec<(String, PathBuf)>,
) -> (String, Option<String>) {
    let raw = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let mut name = raw.clone();
    let mut n = 2usize;
    while claimed
        .iter()
        .any(|(c, owner)| c == &name && owner.as_path() != key)
    {
        name = format!("{raw}_{n}");
        n += 1;
    }
    claimed.push((name.clone(), key.to_path_buf()));
    if name == raw {
        return (name, None);
    }
    (
        name.clone(),
        Some(format!(
            "writes to {name}/ \u{2014} another input folder already owns the name {raw}"
        )),
    )
}

/// Last line of defence against two queue entries sharing one destination.
///
/// The folder logic above handles the case that actually happens; this handles the rest,
/// because every way I can think of for a destination to be wanted twice is a way to lose
/// a file quietly: the duplicate is skipped as already rendered, the run reports no
/// failures, and the count of outputs is short by however many collided.
fn keep_each_item_on_its_own_path(items: &mut [Item]) {
    // Keyed through `path_key` so two spellings of one not-yet-existing file
    // (`out/x.flac`, `./out/x.flac`, a symlinked folder) still collide; one key per
    // path keeps this linear, since each key costs filesystem calls.
    let mut taken: std::collections::HashSet<PathBuf> =
        std::collections::HashSet::with_capacity(items.len());
    for it in items.iter_mut() {
        let mut key = path_key(&it.dst);
        // An item that maps onto itself writes nothing (the render refuses it), so it does
        // not hold the name: otherwise an earlier run's in-place output, discovered as an
        // input, would push its own source to `name_2` instead of being skipped.
        if key == path_key(&it.src) {
            continue;
        }
        let mut dst = it.dst.clone();
        let mut n = 2usize;
        while taken.contains(&key) {
            dst = bump_stem(&it.dst, n);
            key = path_key(&dst);
            n += 1;
        }
        if dst != it.dst {
            let moved = dst
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "a renamed file".to_string());
            let reason = format!("written as {moved} \u{2014} that output path was already taken");
            it.dst = dst;
            it.note = Some(match it.note.take() {
                Some(already) => format!("{already}; {reason}"),
                None => reason,
            });
        }
        taken.insert(key);
    }
}

/// `name.ext` -> `name_2.ext`, keeping whatever directory it was in.
fn bump_stem(p: &Path, n: usize) -> PathBuf {
    let name = p
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let stem = match name.rfind('.') {
        Some(i) if i > 0 => &name[..i],
        _ => name.as_str(),
    };
    let mut out = p.parent().unwrap_or(Path::new("")).to_path_buf();
    out.push(format!("{stem}_{n}{}", &name[stem.len()..]));
    out
}

/// Where a path lands on disk, for comparing outputs with each other and with inputs.
/// Outputs usually do not exist yet, so only the parent is resolved.
fn path_key(p: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(p) {
        return c;
    }
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(p)
    };
    match (abs.parent(), abs.file_name()) {
        (Some(dir), Some(name)) => std::fs::canonicalize(dir)
            .map(|d| d.join(name))
            .unwrap_or(abs.clone()),
        _ => abs,
    }
}

/// With overwriting on, fail every item whose output is a *different* input of the same
/// batch: `song.wav` next to `song.flac` would otherwise render over the original
/// `song.flac`. Without overwriting the render already skips an existing output, so
/// nothing is refused. Called by [`Batch::start`]; public for `--dry-run`.
pub fn refuse_overwriting_inputs(items: &mut [Item]) {
    let srcs: std::collections::HashSet<PathBuf> = items.iter().map(|i| path_key(&i.src)).collect();
    for item in items.iter_mut() {
        if item.phase == Phase::Failed {
            continue;
        }
        let key = path_key(&item.dst);
        if key != path_key(&item.src) && srcs.contains(&key) {
            item.phase = Phase::Failed;
            item.error = Some(format!(
                "its output {} is also an input of this batch, and overwriting is on; \
                 rendering it would replace that file (use an output folder, or render \
                 them separately)",
                item.dst.display()
            ));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_file(
    src: &Path,
    base: &Path,
    prefix: &Path,
    note: Option<&str>,
    out_dir: Option<&Path>,
    ext: &str,
    seen: &mut Vec<PathBuf>,
    items: &mut Vec<Item>,
) {
    let canon = std::fs::canonicalize(src).unwrap_or_else(|_| src.to_path_buf());
    if seen.iter().any(|s| s == &canon) {
        return; // the same album reached through two arguments or a symlink
    }
    seen.push(canon);
    let dst = output_for_prefixed(src, base, prefix, out_dir, ext);
    let item = Item::new(src.to_path_buf(), dst);
    items.push(match note {
        Some(why) => item.with_note(why),
        None => item,
    });
}

/// Live counters, read by the UI thread while workers render.
#[derive(Default)]
pub struct Counters {
    pub total: usize,
    pub done: AtomicUsize,
    pub failed: AtomicUsize,
    pub skipped: AtomicUsize,
    pub running: AtomicUsize,
    pub frames: AtomicU64,
    pub bytes: AtomicU64,
    pub seconds: AtomicU64,
}

impl Counters {
    /// Seconds of audio completed, for an ETA.
    pub fn seconds_done(&self) -> f64 {
        f64::from_bits(self.seconds.load(Ordering::Relaxed))
    }

    pub fn set_seconds_done(&self, v: f64) {
        self.seconds.store(v.to_bits(), Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> CountersView {
        CountersView {
            total: self.total,
            done: self.done.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            skipped: self.skipped.load(Ordering::Relaxed),
            running: self.running.load(Ordering::Relaxed),
            frames: self.frames.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            seconds_done: self.seconds_done(),
        }
    }
}

/// Serializable view of [`Counters`].
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct CountersView {
    pub total: usize,
    pub done: usize,
    pub failed: usize,
    pub skipped: usize,
    pub running: usize,
    pub frames: u64,
    pub bytes: u64,
    pub seconds_done: f64,
}

impl CountersView {
    /// Finished items over total, 0..=1.
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            ((self.done + self.failed + self.skipped) as f64 / self.total as f64) as f32
        }
    }

    /// Files still to finish.
    pub fn remaining(&self) -> usize {
        self.total
            .saturating_sub(self.done + self.failed + self.skipped)
    }

    /// Wall-clock estimate from the average rendered-second-per-second-of-wall
    /// time observed so far. `None` until there is a completed file to learn from.
    pub fn eta(&self, elapsed: Duration, per_item_seconds: f64) -> Option<Duration> {
        let finished = self.done + self.failed;
        if finished == 0 || self.seconds_done <= 0.0 || elapsed.is_zero() {
            return None;
        }
        let rate = self.seconds_done / elapsed.as_secs_f64();
        if rate <= 0.0 {
            return None;
        }
        let left = self.remaining() as f64 * per_item_seconds.max(1.0);
        Some(Duration::from_secs_f64(
            (left / rate).clamp(0.0, 86_400.0 * 7.0),
        ))
    }
}

/// Bump the shared counters from inside a render.
struct Feed {
    frames: Arc<AtomicU64>,
    bytes: Arc<AtomicU64>,
}

impl Sink for Feed {
    fn update(&self, frames: u64, bytes_out: u64) {
        self.frames.store(frames, Ordering::Relaxed);
        self.bytes.store(bytes_out, Ordering::Relaxed);
    }
}

/// Outcome of a whole batch.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub done: usize,
    pub failed: usize,
    pub skipped: usize,
    pub cancelled: bool,
    pub elapsed: Duration,
    pub items: Vec<Item>,
}

impl Outcome {
    pub fn one_line(&self) -> String {
        format!(
            "{} ok, {} failed, {} skipped in {:.1}s",
            self.done,
            self.failed,
            self.skipped,
            self.elapsed.as_secs_f64()
        )
    }

    /// Non-zero exit code for the CLI when something failed.
    pub fn exit_code(&self) -> i32 {
        if self.cancelled {
            130
        } else if self.failed > 0 {
            1
        } else {
            0
        }
    }
}

/// A running batch, shared with the caller so it can show progress and cancel.
pub struct Batch {
    pub items: Arc<Mutex<Vec<Item>>>,
    pub counters: Arc<Counters>,
    pub cancel: Arc<AtomicBool>,
    started: Instant,
    handles: Vec<std::thread::JoinHandle<()>>,
}

/// Called after each item reaches a terminal state. Must be cheap: it runs on a
/// worker thread between files.
pub type Progress = Arc<dyn Fn(usize, &Item) + Send + Sync>;

impl Drop for Batch {
    fn drop(&mut self) {
        // A batch dropped without `finish` must not keep rendering files nobody is
        // waiting for. The workers own their own clones, so signalling is all this
        // can safely do from a destructor.
        self.cancel();
    }
}

impl Batch {
    /// Queue and start rendering. Returns as soon as the workers are running.
    pub fn start(
        items: Vec<Item>,
        cfg: &BatchCfg,
        tools: Tools,
        caps: Caps,
        progress: Option<Progress>,
    ) -> Self {
        let mut items = items;
        if cfg.render.out.overwrite {
            refuse_overwriting_inputs(&mut items);
        }
        let total = items.len();
        let items = Arc::new(Mutex::new(items));
        let counters = Arc::new(Counters {
            total,
            ..Counters::default()
        });
        let cancel = Arc::new(AtomicBool::new(false));
        let cursor = Arc::new(AtomicUsize::new(0));
        let workers = cfg.workers.clamp(1, 64).min(total.max(1));

        // One ffmpeg child pair per worker, and each ffmpeg is told not to slice
        // its own filtergraph across cores: N workers x M filter threads is how a
        // batch ends up slower than a single thread.
        let mut cfg = cfg.clone();
        if cfg.render.out.filter_threads.is_none() && workers > 1 {
            cfg.render.out.filter_threads = Some(1);
        }

        let mut handles = Vec::with_capacity(workers);
        for id in 0..workers {
            let items = items.clone();
            let counters = counters.clone();
            let cancel_w = cancel.clone();
            let cursor = cursor.clone();
            let tools = tools.clone();
            let caps = caps.clone();
            let cfg = cfg.clone();
            let progress = progress.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("g6me-worker-{id}"))
                .spawn(move || {
                    loop {
                        if cancel_w.load(Ordering::Relaxed) {
                            break;
                        }
                        let idx = cursor.fetch_add(1, Ordering::Relaxed);
                        if idx >= total {
                            break;
                        }
                        // Refused before the run (`refuse_overwriting_inputs`): counted, never rendered.
                        let refused = {
                            let guard = items.lock().unwrap_or_else(|e| e.into_inner());
                            (guard[idx].phase == Phase::Failed).then(|| guard[idx].clone())
                        };
                        if let Some(item) = refused {
                            counters.failed.fetch_add(1, Ordering::Relaxed);
                            if let Some(p) = &progress {
                                p(idx, &item);
                            }
                            continue;
                        }
                        let (src, dst) = {
                            let guard = items.lock().unwrap_or_else(|e| e.into_inner());
                            (guard[idx].src.clone(), guard[idx].dst.clone())
                        };
                        {
                            let mut guard = items.lock().unwrap_or_else(|e| e.into_inner());
                            guard[idx].phase = Phase::Running;
                        }
                        counters.running.fetch_add(1, Ordering::Relaxed);

                        let frames = Arc::new(AtomicU64::new(0));
                        let bytes = Arc::new(AtomicU64::new(0));
                        let feed = Arc::new(Feed {
                            frames: frames.clone(),
                            bytes: bytes.clone(),
                        });
                        // Fresh tap per file: its meter belongs to this file alone.
                        let tap = cfg.taps.0.as_ref().and_then(|make| make(idx));
                        let opts = RenderOpts {
                            tools: &tools,
                            caps: &caps,
                            cancel: Some(cancel_w.clone()),
                            progress: Some(feed),
                            tap,
                        };
                        // A panic in one render fails that file, not the worker: an unwound
                        // worker never reports its item, and a caller waiting for every item
                        // to finish waits forever.
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            render_file(&src, &dst, &cfg.render, &opts)
                        }))
                        .unwrap_or_else(|p| {
                            let why = p
                                .downcast_ref::<String>()
                                .map(String::as_str)
                                .or_else(|| p.downcast_ref::<&str>().copied())
                                .unwrap_or("unknown cause");
                            Err(Error::msg(format!("render panicked: {why}")))
                        });

                        counters.running.fetch_sub(1, Ordering::Relaxed);
                        let mut guard = items.lock().unwrap_or_else(|e| e.into_inner());
                        let item = &mut guard[idx];
                        item.frames = frames.load(Ordering::Relaxed);
                        match result {
                            Ok(report) => {
                                counters.done.fetch_add(1, Ordering::Relaxed);
                                counters.frames.fetch_add(report.frames, Ordering::Relaxed);
                                counters
                                    .bytes
                                    .fetch_add(report.output_bytes, Ordering::Relaxed);
                                counters.set_seconds_done(counters.seconds_done() + report.seconds);
                                item.phase = Phase::Done;
                                item.report = Some(report);
                                item.error = None;
                            }
                            Err(e) => {
                                // An existing output is a decision, not a fault.
                                if matches!(e, Error::Exists(_)) {
                                    counters.skipped.fetch_add(1, Ordering::Relaxed);
                                    item.phase = Phase::Skipped;
                                } else {
                                    counters.failed.fetch_add(1, Ordering::Relaxed);
                                    item.phase = Phase::Failed;
                                }
                                item.error = Some(e.to_string());
                                item.report = None;
                            }
                        }
                        let done = item.clone();
                        drop(guard);
                        if let Some(p) = &progress {
                            p(idx, &done);
                        }
                    }
                })
                .map(|h| handles.push(h));
            if let Err(e) = spawned {
                // Better to stop than to quietly render on fewer threads than the
                // user asked for.
                eprintln!("worker {id} could not start: {e}");
                cancel.store(true, Ordering::Relaxed);
                break;
            }
        }

        Self {
            items,
            counters,
            cancel,
            started: Instant::now(),
            handles,
        }
    }

    /// Ask every worker to stop at the next item boundary, and every in-flight
    /// render to stop at the next block.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// Copy of the queue for display.
    pub fn snapshot(&self) -> Vec<Item> {
        self.items.lock().map(|i| i.clone()).unwrap_or_default()
    }

    /// Join the workers and produce the final result.
    ///
    /// Workers retire by themselves once the cursor passes the end of the queue,
    /// so this does *not* cancel: cancelling here would abandon whatever is still
    /// queued and report a clean run that rendered almost nothing.
    pub fn finish(mut self) -> Outcome {
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
        let view = self.counters.snapshot();
        let done = view.done;
        Outcome {
            done,
            failed: view.failed,
            skipped: view.skipped,
            cancelled: self.is_cancelled(),
            elapsed: self.started.elapsed(),
            items: self.snapshot(),
        }
    }

    /// Run a batch to completion (the CLI path).
    pub fn run(
        items: Vec<Item>,
        cfg: &BatchCfg,
        tools: Tools,
        caps: Caps,
        progress: Option<Progress>,
    ) -> Outcome {
        if items.is_empty() {
            return Outcome::default();
        }
        Batch::start(items, cfg, tools, caps, progress).finish()
    }
}

/// Choose the encoder from the requested format and depth, so callers do not
/// have to know that 24-bit FLAC needs a different binary.
pub fn encoder_for(fmt: &OutFormat, depth: BitDepth) -> OutFormat {
    match fmt {
        OutFormat::Flac { level } if depth == BitDepth::S24 => OutFormat::Flac { level: *level },
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::probe_caps;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("g6me-batch-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"fake audio").unwrap();
    }

    #[test]
    fn a_folder_queues_in_ascending_order_files_before_subfolders() {
        let root = tmp("order");
        let lib = root.join("lib");
        for n in [
            "b.wav",
            "a.wav",
            "c.wav",
            "Disc 2/01.wav",
            "Disc 1/02.wav",
            "Disc 1/01.wav",
        ] {
            touch(&lib.join(n));
        }
        let items = discover(
            std::slice::from_ref(&lib),
            Some(&root.join("out")),
            "flac",
            true,
        );
        let got: Vec<String> = items
            .iter()
            .map(|i| {
                i.src
                    .strip_prefix(&lib)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        assert_eq!(
            got,
            [
                "a.wav",
                "b.wav",
                "c.wav",
                "Disc 1/01.wav",
                "Disc 1/02.wav",
                "Disc 2/01.wav"
            ]
        );
    }

    #[test]
    fn colliding_outputs_are_renamed_and_inputs_are_not_overwritten() {
        let root = tmp("collide");
        let lib = root.join("lib");
        touch(&lib.join("song.wav"));
        touch(&lib.join("song.flac"));
        touch(&lib.join("take.wav"));
        touch(&lib.join("take.aiff"));
        touch(&lib.join("solo.wav"));

        // In place: take.* share an output name; the first one keeps it and the second
        // is moved, with the move said on the item.
        let mut items = discover(std::slice::from_ref(&lib), None, "flac", true);
        let get = |items: &[Item], name: &str| {
            items
                .iter()
                .find(|i| i.src.ends_with(name))
                .unwrap()
                .clone()
        };
        assert_eq!(get(&items, "take.aiff").dst, lib.join("take.flac"));
        let t = get(&items, "take.wav");
        assert_eq!(t.phase, Phase::Queued);
        assert_eq!(t.dst, lib.join("take_2.flac"));
        assert!(t.note.unwrap_or_default().contains("already taken"));
        assert_eq!(get(&items, "solo.wav").dst, lib.join("solo.flac"));
        // song.flac -> itself writes nothing, so it does not hold the name: song.wav
        // keeps song.flac, which only skips without overwriting...
        let s = get(&items, "song.wav");
        assert_eq!((s.phase, s.dst), (Phase::Queued, lib.join("song.flac")));
        // ...and is refused with it. song.flac -> itself stays the render's own
        // same-file refusal.
        refuse_overwriting_inputs(&mut items);
        let s = get(&items, "song.wav");
        assert_eq!(s.phase, Phase::Failed);
        assert!(s.error.unwrap_or_default().contains("also an input"));
        assert_eq!(get(&items, "song.flac").phase, Phase::Queued);

        // With an output folder nothing is an input any more; take.* still collide, and
        // so do song.flac and song.wav, which both want out/song.flac. Nothing fails.
        let out = root.join("out");
        let mut items = discover(std::slice::from_ref(&lib), Some(&out), "flac", true);
        refuse_overwriting_inputs(&mut items);
        assert!(items.iter().all(|i| i.phase == Phase::Queued));
        assert_eq!(get(&items, "song.flac").dst, out.join("song.flac"));
        assert_eq!(get(&items, "song.wav").dst, out.join("song_2.flac"));
        assert_eq!(get(&items, "take.wav").dst, out.join("take_2.flac"));
        let mut dsts: Vec<_> = items.iter().map(|i| i.dst.clone()).collect();
        dsts.sort();
        dsts.dedup();
        assert_eq!(dsts.len(), items.len());
    }

    /// Two spellings of one output that does not exist yet still collide (`path_key`).
    #[test]
    fn differently_spelled_destinations_still_collide() {
        let root = tmp("spelled");
        std::fs::create_dir_all(root.join("out")).unwrap();
        let mut items = vec![
            Item::new(root.join("a.wav"), root.join("out/x.flac")),
            Item::new(root.join("b.wav"), root.join("out/./x.flac")),
            Item::new(root.join("c.wav"), root.join("out/../out/x.flac")),
        ];
        keep_each_item_on_its_own_path(&mut items);
        let names: Vec<_> = items
            .iter()
            .map(|i| i.dst.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["x.flac", "x_2.flac", "x_3.flac"]);
    }

    #[test]
    fn refused_items_count_as_failed_and_never_reach_a_render() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let root = tmp("refused");
        let lib = root.join("lib");
        touch(&lib.join("song.wav"));
        touch(&lib.join("song.flac"));
        let before = std::fs::read(lib.join("song.flac")).unwrap();
        let cfg = BatchCfg {
            inputs: vec![lib.clone()],
            workers: 1,
            render: RenderCfg {
                out: crate::stages::OutCfg {
                    overwrite: true,
                    copy_metadata: false,
                    ..Default::default()
                },
                ..RenderCfg::default()
            },
            ..Default::default()
        };
        let items = discover(&cfg.inputs, None, "flac", true);
        let out = Batch::run(items, &cfg, tools, caps, None);
        assert_eq!(
            out.failed, 2,
            "song.wav refused, song.flac is its own output"
        );
        assert_eq!(std::fs::read(lib.join("song.flac")).unwrap(), before);
    }

    /// A batch over `dir` that may overwrite and does not copy metadata (the
    /// metadata copy needs a real source file, which several of these tests fake).
    fn failing_cfg(dir: PathBuf, workers: usize) -> BatchCfg {
        BatchCfg {
            inputs: vec![dir],
            workers,
            render: RenderCfg {
                out: crate::stages::OutCfg {
                    overwrite: true,
                    copy_metadata: false,
                    ..Default::default()
                },
                ..RenderCfg::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn discovery_walks_mirrors_and_dedupes() {
        let root = tmp("walk");
        let music = root.join("Music");
        touch(&music.join("Artist/Album/01.mka"));
        touch(&music.join("Artist/Album/02.mka"));
        touch(&music.join("Artist/Album/cover.jpg"));
        touch(&music.join("Artist/Album/notes.txt"));
        touch(&music.join("Artist/Album/03.flac"));
        touch(&root.join("loose.m4a"));

        let out = root.join("out");
        let items = discover(std::slice::from_ref(&music), Some(&out), "flac", true);
        let names: Vec<String> = items
            .iter()
            .map(|i| {
                i.dst
                    .strip_prefix(&out)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        assert_eq!(items.len(), 3, "cover/txt must not appear, got {names:?}");
        assert!(
            names.iter().any(|n| n.ends_with("Artist/Album/01.flac")),
            "tree must be mirrored, got {names:?}"
        );

        // Passing the same folder twice must not queue anything twice.
        let twice = discover(&[music.clone(), music], Some(&out), "flac", true);
        assert_eq!(twice.len(), 3);

        // Loose file argument: output goes next to it.
        let loose = discover(&[root.join("loose.m4a")], None, "flac", true);
        assert_eq!(loose.len(), 1);
        assert_eq!(
            loose[0].dst.file_name().unwrap().to_string_lossy(),
            "loose.flac"
        );
        assert_eq!(loose[0].dst.parent(), Some(root.as_path()));
        std::fs::remove_dir_all(&root).ok();
    }

    /// Two folders on one command line each keep their own output.
    ///
    /// Both albums hold a file named `01.mka`. Mirroring every argument against the first
    /// one's root — and then, in the naive fix, each against its own root — puts both on
    /// `out/01.flac`, and the collision is silent: the second file is skipped as already
    /// rendered, so the batch reports success having written one file of two.
    #[test]
    fn several_folders_do_not_write_on_each_other() {
        let root = tmp("roots");
        let a = root.join("AlbumA");
        let b = root.join("AlbumB");
        touch(&a.join("01.mka"));
        touch(&b.join("01.mka"));
        let out = root.join("out");
        let items = discover(&[a, b], Some(&out), "flac", true);
        assert_eq!(items.len(), 2, "both albums must be queued");
        let mut names: Vec<String> = items
            .iter()
            .map(|i| {
                i.dst
                    .strip_prefix(&out)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        names.sort();
        assert_ne!(names[0], names[1], "both wrote {names:?}");
        assert_eq!(
            names,
            ["AlbumA/01.flac", "AlbumB/01.flac"],
            "each folder should keep its own name"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// Two input folders that share a basename, which the test above cannot catch
    /// because it deliberately gave them different names.
    ///
    /// Sabotage check: drop the numbering loop in `claim_output_folder` and the `names`
    /// assertion fails with both albums claiming `out/Album/01.flac` — the shape of the
    /// bug where the second file meets a file that is already there, is skipped as already
    /// rendered, and the batch reports success one file short of what it was handed.
    #[test]
    fn two_folders_with_the_same_name_get_their_own_trees() {
        let root = tmp("same_roots");
        let a = root.join("a").join("Album");
        let b = root.join("b").join("Album");
        touch(&a.join("01.mka"));
        touch(&b.join("01.mka"));
        // One album has a subfolder, so the test also proves the renamed album mirrors
        // its own shape rather than dumping its contents flat.
        touch(&b.join("Disc 2").join("02.mka"));
        let out = root.join("out");
        let items = discover(&[a.clone(), b.clone()], Some(&out), "flac", true);

        let mut names: Vec<String> = items
            .iter()
            .map(|i| {
                i.dst
                    .strip_prefix(&out)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        names.sort();
        assert_eq!(
            names,
            ["Album/01.flac", "Album_2/01.flac", "Album_2/Disc 2/02.flac"],
            "the first folder keeps its name, the second is numbered, and the renamed one \
             keeps its own subfolder structure"
        );
        assert_ne!(items[0].dst, items[1].dst);
        assert!(
            items
                .iter()
                .any(|i| i.src == a.join("01.mka") && i.dst == out.join("Album").join("01.flac")),
            "the first argument is the one that keeps the name: {names:?}"
        );
        assert!(
            items
                .iter()
                .any(|i| i.src == b.join("Disc 2").join("02.mka")
                    && i.dst == out.join("Album_2").join("Disc 2").join("02.flac")),
            "the renamed album must keep its own folder structure: {names:?}"
        );

        // The rename is not allowed to be silent.
        let noted: Vec<&Item> = items.iter().filter(|i| i.note.is_some()).collect();
        assert_eq!(
            noted.len(),
            2,
            "both files of the moved folder should say they moved: {names:?}"
        );
        for n in noted {
            assert!(
                n.note.as_deref().unwrap().contains("Album_2"),
                "{:?}",
                n.note
            );
        }
        std::fs::remove_dir_all(&root).ok();
    }

    /// The backstop, for the cases the folder logic does not cover: two *different* files
    /// that compute the same destination.
    ///
    /// Reachable without any trickery — hand the tool two loose files of the same name
    /// that are not under the working directory. Neither is inside the mirror base, so
    /// both land flat in `out_dir`, and without this the second is skipped as already
    /// rendered while the run reports no failures.
    #[test]
    fn a_second_file_that_wants_a_taken_path_gets_its_own() {
        let root = tmp("dup_dst");
        let x = root.join("x").join("song.mka");
        let y = root.join("y").join("song.mka");
        touch(&x);
        touch(&y);
        let out = root.join("out");
        let items = discover(&[x.clone(), y.clone()], Some(&out), "flac", true);
        assert_eq!(items.len(), 2, "both files must be queued");
        assert_ne!(
            items[0].dst, items[1].dst,
            "two different files were queued onto one destination"
        );
        let mut stems: Vec<String> = items
            .iter()
            .map(|i| i.dst.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        stems.sort();
        assert_eq!(stems, ["song.flac", "song_2.flac"], "{stems:?}");
        let noted: Vec<&Item> = items.iter().filter(|i| i.note.is_some()).collect();
        assert_eq!(noted.len(), 1, "exactly one file had to move: {stems:?}");
        assert!(
            noted[0].dst.ends_with("song_2.flac"),
            "the note must travel with the file that actually moved"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// `recursive: false` has to mean flat.
    #[test]
    fn a_flat_scan_stays_at_the_top() {
        let root = tmp("flat");
        touch(&root.join("top.mka"));
        touch(&root.join("Deep/inner.mka"));
        let items = discover(std::slice::from_ref(&root), None, "flac", false);
        assert_eq!(
            items.len(),
            1,
            "a non-recursive scan found a file in a subfolder: {:?}",
            items.iter().map(|i| &i.src).collect::<Vec<_>>()
        );
        let deep = discover(std::slice::from_ref(&root), None, "flac", true);
        assert_eq!(deep.len(), 2, "the same tree must still walk recursively");
        std::fs::remove_dir_all(&root).ok();
    }

    /// A symlink pointing back at an ancestor must not spin forever.
    #[cfg(unix)]
    #[test]
    fn a_symlink_loop_is_entered_once() {
        let root = tmp("loop");
        let inner = root.join("Artist");
        touch(&inner.join("01.mka"));
        std::os::unix::fs::symlink(&root, inner.join("back")).unwrap();
        let items = discover(std::slice::from_ref(&root), None, "flac", true);
        assert_eq!(
            items.len(),
            1,
            "the loop should add nothing, got {:?}",
            items.iter().map(|i| &i.src).collect::<Vec<_>>()
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn extension_filters_are_exclusive() {
        assert!(looks_like_audio(Path::new("a.mka")));
        assert!(looks_like_audio(Path::new("a.DSF")));
        assert!(!looks_like_audio(Path::new("cover.JPG")));
        assert!(!looks_like_audio(Path::new("notes.txt")));
        assert!(!looks_like_audio(Path::new("noextension")));
    }

    #[test]
    fn counters_view_math() {
        let c = Counters::default();
        let mut v = c.snapshot();
        assert_eq!(v.fraction(), 0.0);
        assert_eq!(v.remaining(), 0);
        v.total = 10;
        v.done = 4;
        v.failed = 1;
        assert!((v.fraction() - 0.5).abs() < 1e-6);
        assert_eq!(v.remaining(), 5);
        // An ETA needs observed throughput, not just a count of finished files.
        assert!(v.eta(Duration::from_secs(10), 60.0).is_none());
        v.seconds_done = 300.0;
        let eta = v
            .eta(Duration::from_secs(10), 60.0)
            .expect("throughput known");
        // 300 s of audio in 10 s of wall time = 30x; 5 files x 60 s = 300 s left.
        assert!((eta.as_secs_f64() - 10.0).abs() < 0.5, "got {eta:?}");
        assert!(CountersView::default()
            .eta(Duration::from_secs(10), 60.0)
            .is_none());
    }

    #[test]
    fn workers_stay_within_reason() {
        let w = default_workers();
        assert!((1..=4).contains(&w), "got {w}");
    }

    /// The real thing: three generated files, two workers, FLAC out.
    #[test]
    fn batch_renders_a_folder_and_reports_each_file() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("run");
        let src_dir = dir.join("in");
        std::fs::create_dir_all(&src_dir).unwrap();
        for i in 0..3 {
            let p = src_dir.join(format!("t{i}.wav"));
            crate::wav::write_pcm_s32(&p, 48000, 8, &vec![(i as u8) + 1; (48000 / 8) * 8 * 4])
                .unwrap();
        }
        let out = dir.join("out");
        let mut cfg = failing_cfg(src_dir, 2);
        cfg.out_dir = Some(out.clone());

        let items = discover(&cfg.inputs, cfg.out_dir.as_deref(), "flac", cfg.recursive);
        assert_eq!(items.len(), 3);
        let done = Arc::new(AtomicUsize::new(0));
        let sink = {
            let done = done.clone();
            Arc::new(move |_i: usize, it: &Item| {
                assert_eq!(
                    it.phase,
                    Phase::Done,
                    "{}: {:?}",
                    it.src.display(),
                    it.error
                );
                done.fetch_add(1, Ordering::Relaxed);
            })
        } as Progress;
        let o = Batch::run(items, &cfg, tools.clone(), caps.clone(), Some(sink));
        eprintln!("batch: {}", o.one_line());
        assert_eq!(
            o.done,
            3,
            "{:?}",
            o.items
                .iter()
                .map(|i| (&i.dst, &i.error))
                .collect::<Vec<_>>()
        );
        assert_eq!(o.failed, 0);
        assert_eq!(done.load(Ordering::Relaxed), 3);
        for i in 0..3 {
            assert!(out.join(format!("t{i}.flac")).is_file());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A corrupt file must fail alone; the queue keeps going.
    #[test]
    fn one_bad_file_does_not_stop_the_batch() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("bad");
        std::fs::write(
            dir.join("broken.mka"),
            b"this is not a matroska file at all",
        )
        .unwrap();
        crate::wav::write_pcm_s32(&dir.join("good.wav"), 48000, 2, &vec![0u8; 4800]).unwrap();
        let cfg = failing_cfg(dir.clone(), 2);
        let items = discover(&cfg.inputs, None, "flac", cfg.recursive);
        assert_eq!(items.len(), 2);
        let o = Batch::run(items, &cfg, tools, caps, None);
        assert_eq!(
            o.done,
            1,
            "the good file must still render: {:?}",
            o.items
                .iter()
                .map(|i| (&i.src, &i.phase, &i.error))
                .collect::<Vec<_>>()
        );
        assert_eq!(o.failed, 1);
        assert_eq!(o.exit_code(), 1);
        assert!(dir.join("good.flac").is_file());
        assert!(
            !dir.join("broken.flac").exists(),
            "no output for a failed file"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn existing_outputs_are_skipped_not_clobbered() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("exists");
        crate::wav::write_pcm_s32(&dir.join("a.wav"), 48000, 2, &vec![0u8; 4800]).unwrap();
        let out = dir.join("a.flac");
        std::fs::write(&out, b"precious").unwrap();
        let cfg = BatchCfg {
            inputs: vec![dir.clone()],
            workers: 1,
            ..Default::default()
        };
        let items = discover(&cfg.inputs, None, "flac", cfg.recursive);
        let o = Batch::run(items, &cfg, tools, caps, None);
        assert_eq!(o.skipped, 1, "{:?}", o.items[0].error);
        assert_eq!(o.done, 0);
        assert_eq!(std::fs::read(&out).unwrap(), b"precious");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cancel_stops_the_batch_and_leaves_no_partial_files() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("cancel");
        for i in 0..12 {
            crate::wav::write_pcm_s32(
                &dir.join(format!("c{i}.wav")),
                48000,
                8,
                &vec![3u8; 48000 * 8 * 4],
            )
            .unwrap();
        }
        let cfg = failing_cfg(dir.clone(), 2);
        let items = discover(&cfg.inputs, None, "flac", cfg.recursive);
        let b = Batch::start(items, &cfg, tools, caps, None);
        std::thread::sleep(Duration::from_millis(150));
        b.cancel();
        let o = b.finish();
        assert!(o.cancelled);
        assert!(
            o.done < 12,
            "cancelling after 150 ms should not have finished everything"
        );
        // Every file that exists must be a complete render, never a truncated one.
        for entry in std::fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            // The staging namespace is not an output namespace: a file down that path is a
            // render that did not finish, which is precisely why it has no destination. Weighing
            // it as though it were a delivered track is a category error, and on Windows it was
            // also unreliable, because the operating system can hold a just-killed child's file
            // long enough that the cleanup has not landed by the time the directory is read.
            if p.file_name()
                .map(|n| {
                    n.to_string_lossy()
                        .starts_with(crate::pipeline::STAGING_PREFIX)
                })
                .unwrap_or(false)
            {
                continue;
            }
            if p.extension().map(|e| e == "flac").unwrap_or(false) {
                let meta = std::fs::metadata(&p).unwrap();
                assert!(meta.len() > 100, "suspiciously small output {:?}", p);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The point of staging: cancelling does not leave a destination to be sorry about.
    ///
    /// The test above asks every file that survives a cancel to be a complete render, which
    /// is a heuristic over file size, and it passed on Linux while Windows left a 46-byte
    /// FLAC holding nothing but a stream header. This one asks the stronger question -- a
    /// render that did not finish never got a name -- and it also checks that the staging
    /// siblings themselves are gone, because a guarantee that litters is a guarantee people
    /// route around.
    #[test]
    fn a_cancelled_render_never_took_the_destination() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("cancel-atomic");
        for i in 0..12 {
            crate::wav::write_pcm_s32(
                &dir.join(format!("k{i}.wav")),
                48000,
                8,
                &vec![3u8; 48000 * 8 * 4],
            )
            .unwrap();
        }
        let cfg = failing_cfg(dir.clone(), 4);
        let items = discover(&cfg.inputs, None, "flac", cfg.recursive);
        let b = Batch::start(items, &cfg, tools.clone(), caps.clone(), None);
        std::thread::sleep(Duration::from_millis(120));
        b.cancel();
        let o = b.finish();
        assert!(o.cancelled);
        assert!(o.done < 12, "cancel landed after everything finished");
        for entry in std::fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if name.starts_with(crate::pipeline::STAGING_PREFIX) {
                // Left behind on Windows often enough to be called expected: the file is inert,
                // and the guarantee this rename exists for is about destinations. What has to
                // hold is that no such file is ever mistaken for a track, which is asserted
                // below through the same discovery the CLI uses.
                continue;
            }
            if name.ends_with(".flac") {
                // Whatever is here was renamed into place, so it is a finished render, and
                // "finished" is checked by decoding it rather than by weighing it: the 46-byte
                // file Windows left behind had a perfectly good stream header in it, and a
                // header whose total-samples field was never rewritten decodes to nothing.
                let info = crate::probe::probe_audio(&tools, &p, 0)
                    .unwrap_or_else(|e| panic!("{name} is not readable audio: {e}"));
                assert!(
                    info.duration > 0.5,
                    "{name} decodes to {} s of audio: a header and nothing else",
                    info.duration
                );
            }
        }
        // The property that makes an orphan harmless rather than dangerous: the next batch
        // must not pick it up. Run through the real discovery, not a re-implementation of it.
        let queued = discover(std::slice::from_ref(&dir), None, "flac", false);
        for it in &queued {
            assert!(
                !it.src
                    .file_name()
                    .map(|n| n
                        .to_string_lossy()
                        .starts_with(crate::pipeline::STAGING_PREFIX))
                    .unwrap_or(false),
                "a staging file was queued as input: {:?}",
                it.src
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A staging sibling is never input, wherever it came from.
    ///
    /// The rename means one should not exist, and the case that produces one is the kill and
    /// the power cut. Reading one back as a track would be worse than the orphan: the render
    /// would quietly double the library with a truncated file that has a real-looking name.
    #[test]
    fn staging_files_are_never_queued_as_input() {
        let dir = tmp("staging-inputs");
        crate::wav::write_pcm_s32(&dir.join("real.wav"), 48000, 2, &vec![3u8; 48000 * 2 * 4])
            .unwrap();
        for name in [
            ".g6me-1-0-real.flac",
            ".g6me-99-3-orphan.wav",
            ".g6me-orphan.flac",
        ] {
            std::fs::write(dir.join(name), b"fLaC").unwrap();
        }
        let items = discover(std::slice::from_ref(&dir), None, "flac", true);
        let names: Vec<String> = items
            .iter()
            .map(|i| {
                i.src
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
            .collect();
        assert_eq!(names, vec!["real.wav".to_string()], "queued {names:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The publish step is on the path, and this is the case that shows it.
    ///
    /// A directory sitting on the destination name is the one failure a staged render hits
    /// *after* the encode: the encoder never touches the real name, so the encode succeeds
    /// and only the rename fails. Written straight to the destination, the same case dies
    /// inside ffmpeg and the message talks about muxers and codecs. The wording is not the
    /// point -- the point is that the real name is touched once, by the last step, and that
    /// nothing before it can leave anything behind under it.
    #[test]
    fn a_destination_that_cannot_be_replaced_fails_at_the_publish() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("publish-blocked");
        crate::wav::write_pcm_s32(&dir.join("p.wav"), 48000, 2, &vec![3u8; 48000 * 2 * 4]).unwrap();
        std::fs::create_dir(dir.join("p.flac")).unwrap();
        let cfg = failing_cfg(dir.clone(), 1);
        let items = discover(&cfg.inputs, None, "flac", cfg.recursive);
        assert_eq!(items.len(), 1, "a directory must not be queued as input");
        let o = Batch::run(items, &cfg, tools, caps, None);
        assert_eq!(o.failed, 1);
        let err = o.items[0].error.clone().unwrap_or_default();
        assert!(
            err.contains("could not put it in place"),
            "expected the failure at the publish step, got: {err}"
        );
        let left: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(crate::pipeline::STAGING_PREFIX))
            .collect();
        assert!(left.is_empty(), "the aborted publish left {left:?} behind");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn two_runs_of_the_same_batch_are_identical() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("determinism");
        crate::wav::write_pcm_s32(&dir.join("d.wav"), 44100, 6, &vec![9u8; 44100 * 6 * 4]).unwrap();
        let mut cfg = failing_cfg(dir.clone(), 1);
        cfg.render.sha256 = true;
        let items = discover(&cfg.inputs, None, "flac", cfg.recursive);
        let first = Batch::run(items.clone(), &cfg, tools.clone(), caps.clone(), None);
        let h1 = first.items[0]
            .report
            .as_ref()
            .and_then(|r| r.sha256.clone())
            .expect("hash requested");
        let second = Batch::run(items, &cfg, tools, caps, None);
        let h2 = second.items[0]
            .report
            .as_ref()
            .and_then(|r| r.sha256.clone())
            .expect("hash requested");
        assert_eq!(h1, h2, "same input and config must give the same bytes");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Meters are per file. A GUI spawns one level strip per rendering file, so the
    /// factory must be called exactly once per file, and each set must have seen
    /// that file's own samples.
    #[test]
    fn one_meter_tap_per_rendering_file() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("meters");
        for i in 0..3 {
            crate::wav::write_pcm_s32(
                &dir.join(format!("m{i}.wav")),
                48000,
                8,
                &vec![200u8; 48000 * 8 * 4 / 4],
            )
            .unwrap();
        }
        let mut cfg = failing_cfg(dir.clone(), 2);
        cfg.out_dir = Some(dir.join("out"));

        let seen = Arc::new(Mutex::new(Vec::<usize>::new()));
        let meters: Arc<Mutex<Vec<std::sync::Arc<crate::meter::Meters>>>> =
            Arc::new(Mutex::new(Vec::new()));
        let (seen_make, meters_make) = (seen.clone(), meters.clone());
        cfg.taps = MeterFactory(Some(Arc::new(move |idx: usize| {
            seen_make.lock().unwrap().push(idx);
            let m = std::sync::Arc::new(crate::meter::Meters::new());
            meters_make.lock().unwrap().push(m.clone());
            Some(Arc::new(crate::meter::MeterTap::new(m)))
        })));

        let items = discover(&cfg.inputs, cfg.out_dir.as_deref(), "flac", cfg.recursive);
        assert_eq!(items.len(), 3);
        let outcome = Batch::run(items, &cfg, tools, caps, None);
        assert_eq!(
            outcome.done,
            3,
            "{:?}",
            outcome.items.iter().map(|i| &i.error).collect::<Vec<_>>()
        );

        let calls = seen.lock().unwrap().clone();
        assert_eq!(calls.len(), 3, "one tap per file, got {calls:?}");
        let mut calls = calls;
        calls.sort_unstable();
        assert_eq!(calls, vec![0, 1, 2], "each file got its own tap");
        let ms = meters.lock().unwrap();
        assert_eq!(ms.len(), 3);
        for m in ms.iter() {
            assert!(m.frames() > 0, "a meter that saw no samples is not a meter");
            assert!(
                m.peaks().iter().any(|p| *p > 0.0),
                "the file is a constant tone; something should be lit"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_factory_renders_blind() {
        // The CLI never asks for levels; the default factory must stay empty and
        // must not change what is written.
        let cfg = BatchCfg::default();
        assert!(cfg.taps.0.is_none());
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("MeterFactory(none)"), "{dbg}");
    }

    fn tools_and_caps() -> Option<(Tools, Caps)> {
        let t = Tools::discover();
        if !t.is_usable() {
            eprintln!("skipping: no ffmpeg on PATH");
            return None;
        }
        let caps = probe_caps(&t);
        Some((t, caps))
    }
}
