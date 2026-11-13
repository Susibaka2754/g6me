//! The two widgets that are not just controls: the per-file level cards and the
//! file browser.
//!
//! A card shows one file's ten levels — the same eight input slots plus the two
//! outputs the Windows VST insert exposed, peak-hold with a 300 ms fall — because a
//! person calibrating a headphone curve needs the reference they had in the plugin,
//! not a different picture of the same numbers. Levels arrive already in the
//! plugin's parameter space via [`Meters::param`], so nothing drawn here can invent
//! its own curve.
//!
//! There is one card per file that is rendering, not one per application, because
//! a shared meter would show the loudest of several different pieces of music at
//! once — a level that belongs to none of them.
//!
//! The file browser exists because a native dialog on Linux means GTK or an XDG
//! portal, and this project promises to run on a plain Wayland or dwm session.
//! A pure-egui browser has no such dependency.

use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Ui, Vec2};
use g6me_core::batch;
use g6me_core::meter::{Meters, IN_CHANNELS, OUT_CHANNELS, TOTAL};
use std::path::{Path, PathBuf};

/// dB grid lines, drawn where the eye expects them. -60 is the floor the parameter
/// space maps to 0.0, 0 dBFS maps to 1.0.
const DB_LINES: [f32; 7] = [0.0, -6.0, -12.0, -24.0, -36.0, -48.0, -60.0];

/// Labels under the bars. A 20 px bar has no room for "In3 FC", so the bars carry
/// the short form; the long names are the queue table's job, and [`IN_LABELS`] and
/// [`OUT_LABELS`] stay exported for anything that has the space.
const SHORT_LABELS: [&str; TOTAL] = ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR", "L", "R"];

/// Pick the dark or the light variant of a hand-picked colour.
///
/// Semantic colours (green/amber/red for level, red for an error) are chosen per
/// palette: the amber that reads well on `#000` is nearly invisible on paper white,
/// and the fix is a second value, not a blend.
pub fn tone(ui: &Ui, dark: Color32, light: Color32) -> Color32 {
    if ui.visuals().dark_mode {
        dark
    } else {
        light
    }
}

/// Colour for a level: the hot band is the information, so it is coloured by level
/// rather than by index.
fn level_color(ui: &Ui, p: f32) -> Color32 {
    if p > 0.9 {
        tone(
            ui,
            Color32::from_rgb(222, 84, 66),
            Color32::from_rgb(188, 42, 28),
        )
    } else if p > 0.7 {
        tone(
            ui,
            Color32::from_rgb(216, 178, 58),
            Color32::from_rgb(168, 128, 14),
        )
    } else {
        tone(
            ui,
            Color32::from_rgb(96, 176, 92),
            Color32::from_rgb(30, 122, 60),
        )
    }
}

/// Colour for the state dot in the finished-file history.
///
/// Deliberately the same semantic ramp as [`level_color`]: a red dot in the list and a
/// red bar on a card have to mean the same thing, or the two halves of the meter say
/// contradictory things about the same file.
pub fn status_dot(ui: &Ui, hot: bool, ok: bool) -> Color32 {
    if !ok {
        tone(
            ui,
            Color32::from_rgb(240, 110, 96),
            Color32::from_rgb(176, 26, 16),
        )
    } else if hot {
        tone(
            ui,
            Color32::from_rgb(222, 84, 66),
            Color32::from_rgb(188, 42, 28),
        )
    } else {
        tone(
            ui,
            Color32::from_rgb(96, 176, 92),
            Color32::from_rgb(30, 122, 60),
        )
    }
}

/// Parameter (0..=1) for a dBFS value, mirroring the plugin's mapping.
pub fn db_to_param(db: f32) -> f32 {
    // The inverse of what `param_to_db` prints; -60 dBFS is the floor.
    ((db + 60.0) / 60.0).clamp(0.0, 1.0)
}

/// How wide a bar is allowed to get. Ten 20 px bars read as a row of meters; ten
/// 60 px ones are colour swatches, and the gaps between them get wide enough to
/// park a fader in.
pub const BAR_MAX: f32 = 20.0;
/// Gap between neighbouring bars in the same group when there is room to spare.
pub const GAP_MAX: f32 = 4.0;
/// Air between the input group and the output group. Enough to read as a split,
/// little enough that the pair is not left sitting on the far edge of the window.
pub const SPLIT: f32 = 22.0;
/// Room for the dB numbers, immediately left of the row rather than of the card —
/// the numbers belong to the meters, not to the border.
pub const SCALE_W: f32 = 30.0;

/// Where everything in a card goes. Separate from the drawing so the layout — the
/// part that can actually be wrong — is testable without a window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    /// The dB numbers' column.
    pub scale_w: f32,
    pub bar_w: f32,
    pub gap: f32,
    /// First input bar's left edge.
    pub in_x: f32,
    /// First output bar's left edge.
    pub out_x: f32,
    /// The line drawn between the two groups.
    pub split_x: f32,
    /// Where the bars start and end vertically.
    pub top: f32,
    pub bottom: f32,
    /// Below the bars.
    pub label_y: f32,
    /// Above the bars.
    pub group_y: f32,
    /// The row's own extent: the dB column plus both groups. The card is as wide as
    /// the window; the row is not, and everything drawn for the meters stays inside
    /// this instead of being stretched across the card.
    pub row_left: f32,
    pub row_right: f32,
}

impl Geometry {
    /// Lay `IN_CHANNELS` inputs and `OUT_CHANNELS` outputs out inside `r`: dB column,
    /// then the eight inputs, then a bounded split, then the two outputs — one row,
    /// centred in the card, with the leftovers kept outside it rather than threaded
    /// between the bars.
    pub fn layout(r: Rect) -> Geometry {
        let pad = 6.0f32;
        let title_h = 17.0f32;
        let group_h = 13.0f32;
        let label_h = 13.0f32;
        let inner = Rect::from_min_max(
            Pos2::new(r.left() + pad, r.top() + title_h),
            Pos2::new(r.right() - pad, r.bottom() - pad),
        );

        // What the row costs when there is room to spare.
        let roomy = SCALE_W
            + pad
            + IN_CHANNELS as f32 * BAR_MAX
            + (IN_CHANNELS as f32 - 1.0) * GAP_MAX
            + SPLIT
            + OUT_CHANNELS as f32 * BAR_MAX
            + (OUT_CHANNELS as f32 - 1.0) * GAP_MAX;
        let (bar_w, gap) = if inner.width() >= roomy {
            (BAR_MAX, GAP_MAX)
        } else {
            // Squeeze the bars rather than the split: a gap that closes up turns two
            // groups into eleven anonymous bars, which is the one thing a 7.1 meter
            // must not become.
            let avail = (inner.width() - SCALE_W - pad - SPLIT).max(0.0);
            let pitch = avail / TOTAL as f32;
            ((pitch * 0.82).max(3.0), (pitch * 0.18).max(1.0))
        };
        let top = inner.top() + group_h;
        let bottom = inner.bottom() - label_h;

        let in_w = IN_CHANNELS as f32 * bar_w + (IN_CHANNELS as f32 - 1.0) * gap;
        let out_w = OUT_CHANNELS as f32 * bar_w + (OUT_CHANNELS as f32 - 1.0) * gap;
        let row_w = SCALE_W + pad + in_w + SPLIT + out_w;
        let row_left = inner.left() + ((inner.width() - row_w) / 2.0).max(0.0);
        let in_x = row_left + SCALE_W + pad;
        Geometry {
            scale_w: SCALE_W,
            bar_w,
            gap,
            in_x,
            out_x: in_x + in_w + SPLIT,
            split_x: in_x + in_w + SPLIT / 2.0,
            top,
            bottom,
            label_y: inner.bottom() - label_h + 3.0,
            group_y: inner.top() + 2.0,
            row_left,
            row_right: row_left + row_w,
        }
    }

    /// Left edge of bar `i` (0..8 inputs, 8..10 outputs).
    pub fn bar_x(&self, i: usize) -> f32 {
        let (x0, j) = if i < IN_CHANNELS {
            (self.in_x, i)
        } else {
            (self.out_x, i - IN_CHANNELS)
        };
        x0 + j as f32 * (self.bar_w + self.gap)
    }

    pub fn bar(&self, i: usize) -> Rect {
        Rect::from_min_size(
            Pos2::new(self.bar_x(i), self.top),
            Vec2::new(self.bar_w, self.bar_height()),
        )
    }

    pub fn bar_height(&self) -> f32 {
        (self.bottom - self.top).max(1.0)
    }
}

/// One file's levels, in a bordered card with its name across the top.
pub struct Card<'a> {
    /// File name. The card is per file, so this is the only thing telling ten
    /// identical meters apart.
    pub title: &'a str,
    /// Right-aligned: rate, phase, or a result.
    pub status: &'a str,
    /// A finished file's card stays up, dimmed, so a peak can be read after the
    /// file is gone rather than being missed by a second.
    pub finished: bool,
}

/// Draw one card across the space available. Returns true when clicked, which the
/// caller treats as "clear this file's peak-hold".
pub fn meter_card(ui: &mut Ui, meters: &Meters, card: &Card, height: f32) -> bool {
    let v = ui.visuals().clone();
    let want = Vec2::new(ui.available_width(), height);
    let (resp, painter) = ui.allocate_painter(want, Sense::click());
    let r = resp.rect;
    let g = Geometry::layout(r);

    let border = if card.finished {
        tone(ui, Color32::from_gray(58), Color32::from_gray(190))
    } else {
        v.window_stroke.color
    };
    // The border is what separates a card from the panel now that surfaces are flat
    // — in the pure-black palette the backdrop is identical behind every card.
    painter.rect_stroke(r, 6.0, Stroke::new(1.0_f32, border), StrokeKind::Inside);

    let title_col = if card.finished {
        v.weak_text_color()
    } else {
        v.strong_text_color()
    };
    painter.text(
        Pos2::new(r.left() + 8.0, r.top() + 3.0),
        Align2::LEFT_TOP,
        card.title,
        FontId::proportional(if card.finished { 11.0 } else { 12.5 }),
        title_col,
    );
    if !card.status.is_empty() {
        painter.text(
            Pos2::new(r.right() - 8.0, r.top() + 4.0),
            Align2::RIGHT_TOP,
            card.status,
            FontId::proportional(10.5),
            v.weak_text_color(),
        );
    }

    // Group headings: the split the meters are read by.
    painter.text(
        Pos2::new(g.in_x, g.group_y),
        Align2::LEFT_TOP,
        "input",
        FontId::proportional(9.5),
        v.weak_text_color(),
    );
    painter.text(
        Pos2::new(g.out_x, g.group_y),
        Align2::LEFT_TOP,
        "output",
        FontId::proportional(9.5),
        v.weak_text_color(),
    );
    painter.line_segment(
        [
            Pos2::new(g.split_x, g.group_y),
            Pos2::new(g.split_x, r.bottom() - 5.0),
        ],
        Stroke::new(1.0_f32, v.widgets.noninteractive.bg_stroke.color),
    );

    let scale = Color32::from_gray(if v.dark_mode { 120 } else { 110 });

    // One grid across the whole row, split line and all: the two groups share a
    // scale, and drawing the grid in two pieces makes them look like two meters.
    let grid = Stroke::new(1.0_f32, v.widgets.noninteractive.bg_stroke.color);
    let grid_x0 = g.in_x - 4.0;
    let grid_x1 = g.row_right.max(grid_x0 + 1.0);
    for db in DB_LINES {
        let p = db_to_param(db);
        let y = g.bottom - p * g.bar_height();
        painter.line_segment([Pos2::new(grid_x0, y), Pos2::new(grid_x1, y)], grid);
        // 0 and -60 sit on the ends of the track, so their numbers are nudged in
        // rather than left half outside the card.
        let ty = y.clamp(g.top + 4.0, g.bottom - 4.0);
        painter.text(
            Pos2::new(g.row_left + g.scale_w - 4.0, ty),
            Align2::RIGHT_CENTER,
            format!("{db:.0}"),
            FontId::proportional(9.0),
            scale,
        );
    }

    let peaks = meters.peaks();
    let mut capped = [false; TOTAL];
    for i in 0..TOTAL {
        let bar = g.bar(i);
        // The meter body takes the palette's deepest fill, which in the pure-black
        // mode is exactly #000: an unlit meter should cost nothing.
        painter.rect_filled(bar, 2.0, v.extreme_bg_color);

        let p = meters.param(i).clamp(0.0, 1.0);
        let h = p * bar.height();
        if h > 0.5 {
            painter.rect_filled(
                Rect::from_min_max(Pos2::new(bar.left(), bar.bottom() - h), bar.right_bottom()),
                2.0,
                level_color(ui, p),
            );
        }

        // Peak hold in the same parameter space, so the two are comparable.
        let hold = g6me_core::meter::level_to_param(peaks[i].clamp(0.0, 1.0)).clamp(0.0, 1.0);
        if hold > 0.0 {
            let y = bar.bottom() - hold * bar.height();
            painter.line_segment(
                [Pos2::new(bar.left(), y), Pos2::new(bar.right(), y)],
                Stroke::new(
                    2.0_f32,
                    tone(
                        ui,
                        Color32::from_rgb(226, 226, 226),
                        Color32::from_rgb(40, 40, 40),
                    ),
                ),
            );
        }

        // Always the short set: at 20 px a bar has no room for "In3 FC", and the
        // long names are in the queue table and in the group headings anyway.
        painter.text(
            Pos2::new(bar.left() + bar.width() / 2.0, g.label_y),
            Align2::CENTER_TOP,
            SHORT_LABELS[i],
            FontId::proportional(9.0),
            if card.finished {
                v.weak_text_color()
            } else {
                v.text_color()
            },
        );

        if peaks[i] >= 1.0 {
            // Clipping is worth stating in words, but not in a label: "CLIP" is wider
            // than the bar it belongs to and would land on its neighbour. So the bar
            // itself wears a red cap, and the group heading says that it happened.
            capped[i] = true;
            let cap = bar.height().min(4.0);
            painter.rect_filled(
                Rect::from_min_max(
                    Pos2::new(bar.left(), bar.top()),
                    Pos2::new(bar.right(), bar.top() + cap),
                ),
                2.0,
                tone(
                    ui,
                    Color32::from_rgb(240, 110, 96),
                    Color32::from_rgb(176, 26, 16),
                ),
            );
        }
    }

    // One warning per group, where there is room to read it.
    for (x, hit) in [
        (g.in_x, capped[..IN_CHANNELS].iter().any(|c| *c)),
        (g.out_x, capped[IN_CHANNELS..].iter().any(|c| *c)),
    ] {
        if hit {
            painter.text(
                Pos2::new(x + 30.0, g.group_y),
                Align2::LEFT_TOP,
                "clip",
                FontId::proportional(9.5),
                tone(
                    ui,
                    Color32::from_rgb(240, 110, 96),
                    Color32::from_rgb(176, 26, 16),
                ),
            );
        }
    }

    // The release curve lives in the meter, not here: a redraw is not a sample, so
    // decay is driven by the render thread. Clicking only clears the holds.
    resp.clicked()
}

/// The meters' caption, which is also where the unit is named: nothing in the card
/// needs to spill outside its own border to explain itself.
pub fn caption(running: usize) -> String {
    format!(
        "dBFS, peak hold with the plugin's 300 ms fall · {} card(s) · click one to clear \
         its hold",
        running
    )
}

/// What the browser is being used to pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Any file, including non-audio ones.
    AnyFile,
    /// Audio files only, by extension.
    AudioFile,
    /// A directory to write into, or a folder to queue.
    Dir,
}

/// A drive or mount point worth jumping to from the browser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mount {
    pub path: PathBuf,
    /// What a person would call it: a volume name, a drive letter, `root`.
    pub label: String,
    /// Filesystem type, for the hover text. Empty where the OS does not say.
    pub fstype: String,
}

#[cfg(any(target_os = "linux", test))]
/// Filesystems that are plumbing rather than a place someone keeps music: kernel
/// interfaces, container and package overhead, and the loop mounts a desktop
/// accumulates by the hundred.
const SKIP_FS: [&str; 26] = [
    "proc",
    "sysfs",
    "devtmpfs",
    "devpts",
    "tmpfs",
    "cgroup",
    "cgroup2",
    "pstore",
    "securityfs",
    "debugfs",
    "tracefs",
    "configfs",
    "fusectl",
    "mqueue",
    "hugetlbfs",
    "bpf",
    "autofs",
    "binfmt_misc",
    "nsfs",
    "efivarfs",
    "ramfs",
    "selinuxfs",
    "rpc_pipefs",
    "nfsd",
    "overlay",
    "squashfs",
];

#[cfg(any(target_os = "linux", test))]
/// Prefixes of filesystem names that are plumbing. Checked separately because
/// `fuse.sshfs` is a real mount while `fuse.portal` is not, and both start `fuse.`.
const SKIP_FS_PREFIX: [&str; 3] = ["fuse.portal", "fuse.snapfuse", "fuse.gvfs"];

#[cfg(any(target_os = "linux", test))]
/// Mount points that are never interesting, whatever is under them.
const SKIP_UNDER: [&str; 9] = [
    "/snap/",
    "/var/lib/docker/",
    "/var/lib/snapd/",
    "/var/lib/flatpak/",
    "/proc/",
    "/sys/",
    "/boot/",
    "/tmp/",
    "/var/tmp/",
];

#[cfg(any(target_os = "linux", test))]
/// The same idea as [`SKIP_UNDER`], for the mount point itself rather than what hangs
/// under it. These are skipped by name and not by filesystem type because a machine is
/// free to format `/boot` as vfat and `/tmp` as ext4 — which is exactly when they stop
/// looking like plumbing and start looking like drives worth a button. A boot partition
/// holds kernels and an initramfs and a scratch filesystem is emptied before the next
/// login; neither holds a library, and typing the path still works if anyone disagrees.
const SKIP_POINTS: [&str; 3] = ["/boot", "/tmp", "/var/tmp"];

#[cfg(any(target_os = "linux", test))]
/// Where removable and network media turns up.
const MEDIA_PREFIX: [&str; 4] = ["/run/media/", "/media/", "/mnt/", "/Volumes/"];

#[cfg(any(target_os = "linux", test))]
/// Is this mount worth a button?
fn show_mount(fstype: &str, point: &str) -> bool {
    // Media first: a disc formatted as squashfs, or a stick mounted under a media
    // prefix, still deserves to be reachable. `squashfs` is only on the skip list
    // because of snap, and snap never mounts there.
    if MEDIA_PREFIX.iter().any(|p| point.starts_with(p)) {
        return true;
    }
    // A root shortcut is always useful — that is where a library on a single
    // partition lives.
    if point == "/" {
        return true;
    }
    if SKIP_POINTS.contains(&point) || SKIP_UNDER.iter().any(|p| point.starts_with(p)) {
        return false;
    }
    if SKIP_FS.contains(&fstype) {
        return false;
    }
    !SKIP_FS_PREFIX.iter().any(|p| fstype.starts_with(p))
}

#[cfg(any(target_os = "linux", test))]
/// `/proc/self/mountinfo` escapes a space as `\040`, a tab as `\011`, a newline as
/// `\012` and a backslash as `\134` — and a folder called `My Music` is common
/// enough in a media library that ignoring it would drop the mount that matters.
fn unescape_mount(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        let mut code = String::new();
        for _ in 0..3 {
            match it.peek() {
                Some(d) if d.is_ascii_digit() => code.push(it.next().unwrap()),
                _ => break,
            }
        }
        match code.as_str() {
            "040" => out.push(' '),
            "011" => out.push('\t'),
            "012" => out.push('\n'),
            "134" => out.push('\\'),
            other => {
                out.push('\\');
                out.push_str(other);
            }
        }
    }
    out
}

#[cfg(any(target_os = "linux", test))]
/// What a person would call this mount: the volume name for media, `root` for `/`,
/// otherwise the last path component.
fn mount_label(point: &str) -> String {
    if point == "/" {
        return "root".to_string();
    }
    match point.rsplit('/').find(|s| !s.is_empty()) {
        Some(t) => t.to_string(),
        None => point.to_string(),
    }
}

#[cfg(any(target_os = "linux", test))]
/// Pick the interesting mounts out of `/proc/self/mountinfo` text. `exists` says
/// whether a mount point is currently reachable — a stale entry (an unmounted stick
/// the kernel still lists) would open an empty folder.
///
/// Split from [`mounts`] so the parsing is testable without a machine to mount
/// things on.
pub fn parse_mountinfo_with(text: &str, exists: impl Fn(&Path) -> bool) -> Vec<Mount> {
    let mut out: Vec<Mount> = Vec::new();
    for line in text.lines() {
        // id parent maj:min root mount-point opts [tag:…]* - fstype source superopts
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 7 {
            continue;
        }
        let point = unescape_mount(f[4]);
        // The optional fields end at a bare "-"; the two after it identify the
        // filesystem.
        let Some(sep) = f[6..].iter().position(|x| *x == "-") else {
            continue;
        };
        let rest = &f[6 + sep + 1..];
        if rest.len() < 2 {
            continue;
        }
        let (fstype, source) = (rest[0], rest[1]);
        if !show_mount(fstype, &point) || !exists(Path::new(&point)) {
            continue;
        }
        let label = if fstype == "cifs" || fstype == "smb3" || fstype.ends_with("sshfs") {
            // A network share is identified by where it comes from, not by where it
            // happened to land.
            format!(
                "{} · {}",
                mount_label(&point),
                source.rsplit('/').next().unwrap_or(source)
            )
        } else {
            mount_label(&point)
        };
        let path = PathBuf::from(&point);
        if !out.iter().any(|m| m.path == path) {
            out.push(Mount {
                path,
                label,
                fstype: fstype.to_string(),
            });
        }
    }
    // Volume names sort better than device paths, and a stable order matters more
    // than a clever one: buttons that move between clicks are unreadable.
    out.sort_by_key(|m| m.label.to_lowercase());
    out
}

/// The machine's own mounts. Empty on a system that reports none, which the browser
/// treats as "no buttons" rather than as an error.
#[cfg(target_os = "linux")]
pub fn mounts() -> Vec<Mount> {
    match std::fs::read_to_string("/proc/self/mountinfo") {
        Ok(t) => parse_mountinfo_with(&t, |p| p.is_dir()),
        Err(_) => vec![Mount {
            path: PathBuf::from("/"),
            label: "root".to_string(),
            fstype: String::new(),
        }],
    }
}

#[cfg(target_os = "windows")]
pub fn mounts() -> Vec<Mount> {
    // Every letter the shell answers for. The real volume label needs
    // GetVolumeInformationW, which is not worth a Windows-only dependency for a
    // button caption — the letter is what a Windows user types anyway.
    ('A'..='Z')
        .filter_map(|c| {
            let p = PathBuf::from(format!("{c}:\\"));
            p.exists().then(|| Mount {
                path: p,
                label: format!("{c}:"),
                fstype: String::new(),
            })
        })
        .collect()
}

#[cfg(target_os = "macos")]
pub fn mounts() -> Vec<Mount> {
    let mut out = vec![Mount {
        path: PathBuf::from("/"),
        label: "root".to_string(),
        fstype: String::new(),
    }];
    if let Ok(rd) = std::fs::read_dir("/Volumes") {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.push(Mount {
                    label: p
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| "/Volumes".to_string()),
                    path: p,
                    fstype: String::new(),
                });
            }
        }
    }
    out
}

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
pub fn mounts() -> Vec<Mount> {
    Vec::new()
}

/// What one click on the entry list means, decided with the modifier state and
/// applied by the caller after the list is drawn.
#[derive(Default)]
struct Click {
    /// A folder to open.
    enter: Option<PathBuf>,
    /// Paths to hand back immediately (a double-click on a file).
    confirm: Option<Vec<PathBuf>>,
    /// Index to add to, or take out of, the selection.
    toggle: Option<usize>,
    /// Index to extend the selection to, from the anchor.
    range: Option<usize>,
}

/// A small in-app browser: current path, entry list, selection.
/// What the picker orders a folder by.
///
/// Dates, because a library grows at the end: someone looking for the file they ripped
/// yesterday should not have to scroll past a decade of names to find it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sort {
    #[default]
    Name,
    Modified,
    Created,
}

impl Sort {
    pub const ALL: [Sort; 3] = [Sort::Name, Sort::Modified, Sort::Created];

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Sort::Name => "name",
            Sort::Modified => "modified",
            Sort::Created => "created",
        }
    }
}

/// One folder entry, with the dates the sort needs kept beside it.
///
/// Cached because the picker redraws every frame and a folder of a thousand files should
/// not be stat'd a thousand times a second to answer "which of these is newest".
#[derive(Clone, Debug)]
struct Entry {
    path: PathBuf,
    /// Lowercased file name, for the tie-break. Precomputed because a comparator runs it
    /// on every compare and `to_lowercase()` allocates each time.
    fold: String,
    is_dir: bool,
    modified: Option<std::time::SystemTime>,
    created: Option<std::time::SystemTime>,
}

#[derive(Clone, Debug)]
pub struct Browser {
    pub path: PathBuf,
    /// The path as edited. egui's text widget binds to `String`, and a `PathBuf`
    /// cannot be handed to it directly, so the two are synced on edit and on "go".
    path_text: String,
    /// Everything selected right now. More than one is the normal case when adding to
    /// a queue, which is why this is a list rather than an `Option`.
    pub picked: Vec<PathBuf>,
    /// The entry a shift-click extends from.
    anchor: Option<usize>,
    /// Drives and mount points, read once and refreshed on request.
    mounts: Vec<Mount>,
    pub mode: Mode,
    pub filter_audio: bool,
    /// Set when a chosen path is wrong, shown under the list.
    pub message: Option<String>,
    entries: Vec<PathBuf>,
    entries_for: PathBuf,
    /// What the list is ordered by. Session state, like the path and "audio only": the
    /// panel keeps no settings file, so there is nothing to persist this into.
    pub sort: Sort,
    listing: Vec<Entry>,
    /// The order `entries` is actually in, so a sort change reorders without a re-read.
    sorted_with: Option<Sort>,
}

impl Default for Browser {
    fn default() -> Self {
        Self {
            // Home is where a music library usually starts.
            path: dirs_home(),
            path_text: String::new(),
            picked: Vec::new(),
            anchor: None,
            mounts: Vec::new(),
            mode: Mode::AudioFile,
            filter_audio: true,
            message: None,
            entries: Vec::new(),
            entries_for: PathBuf::new(),
            sort: Sort::default(),
            listing: Vec::new(),
            sorted_with: None,
        }
    }
}

fn dirs_home() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

impl Browser {
    fn refresh(&mut self) {
        self.path_text = self.path.display().to_string();
        if self.entries_for == self.path && !self.entries.is_empty() {
            return;
        }
        self.entries.clear();
        self.entries_for = self.path.clone();
        let read = std::fs::read_dir(&self.path);
        let Ok(rd) = read else {
            self.message = Some(format!("cannot read {}", self.path.display()));
            return;
        };
        let mut dirs: Vec<Entry> = Vec::new();
        let mut files: Vec<Entry> = Vec::new();
        for e in rd.flatten() {
            let p = e.path();
            // One stat per entry, reused for the folder test and for both dates. Sorting
            // by name alone never needed the clock; this does, so the metadata is taken
            // once instead of once per question.
            let md = e.metadata().ok();
            let is_dir = md.as_ref().is_some_and(|m| m.is_dir());
            let Some(name) = p.file_name().map(|n| n.to_string_lossy().to_string()) else {
                continue;
            };
            if is_dir {
                // Skip the noisy hidden and special entries; a library tree has
                // enough of them, and `..` has its own button. Files are not skipped
                // for starting with a dot, which is how they have always behaved here.
                if name == ".." || name == "." || name.starts_with('.') {
                    continue;
                }
            }
            let entry = Entry {
                fold: name.to_lowercase(),
                modified: md.as_ref().and_then(|m| m.modified().ok()),

                // Birth time is the one the filesystem may refuse: it is absent on
                // anything without a crtime, and on those the picker says so instead of
                // showing a column of identical values.
                created: md.as_ref().and_then(|m| m.created().ok()),
                path: p,
                is_dir,
            };
            if is_dir {
                dirs.push(entry);
            } else if self.filter_audio && self.mode != Mode::Dir {
                if batch::looks_like_audio(&entry.path) {
                    files.push(entry);
                }
            } else {
                files.push(entry);
            }
        }
        dirs.extend(files);
        self.listing = dirs;
        self.resort();
    }

    /// Reorder the cached listing into `entries`, the order the list actually draws.
    fn resort(&mut self) {
        let sort = self.sort;
        let mut list = std::mem::take(&mut self.listing);
        list.sort_by(|a, b| {
            // Folders first, whatever the key. This list is also how somebody walks a
            // tree, and a folder that drifts into the middle of a date order is a folder
            // they walk past on the way to the file they came for.
            b.is_dir
                .cmp(&a.is_dir)
                // Newest first: comparing the *other* entry's key is what reverses it.
                // A missing date compares below every real one, so an entry the
                // filesystem would not date lands at the end rather than posing as
                // ancient — which matters because "no date" is a fact about the
                // filesystem, not about the file.
                .then_with(|| match sort {
                    Sort::Name => std::cmp::Ordering::Equal,
                    Sort::Modified => b.modified.cmp(&a.modified),
                    Sort::Created => b.created.or(b.modified).cmp(&a.created.or(a.modified)),
                })
                // And a name to tie-break, so the order of two files written in the same
                // second is the same on two renders of the same folder.
                .then_with(|| a.fold.cmp(&b.fold))
        });
        self.entries = list.iter().map(|e| e.path.clone()).collect();
        self.listing = list;
        self.sorted_with = Some(sort);
    }

    /// Draw the browser. `on_pick` receives the confirmed paths — one on a
    /// double-click, the whole selection when the button is pressed — and the caller
    /// decides what they mean.
    pub fn ui(&mut self, ui: &mut egui::Ui, mut on_pick: impl FnMut(Vec<PathBuf>)) {
        self.refresh();
        if self.sorted_with != Some(self.sort) {
            self.resort();
        }
        if self.mounts.is_empty() {
            self.mounts = mounts();
        }
        self.drive_bar(ui);
        if self.sort == Sort::Created && !self.listing.iter().any(|e| e.created.is_some()) {
            // Better than a column of the same date: this filesystem has no birth time
            // to show, and "created" is quietly answering with "modified".
            ui.weak("nothing in this folder reports a creation time — showing the modified date");
        }
        if self.path_bar(ui) {
            let want = PathBuf::from(self.path_text.trim());
            if want.is_dir() {
                self.goto(want);
            } else {
                self.message = Some(format!("{} is not a folder", self.path_text.trim()));
            }
        }

        // Modifiers are read once, outside the list: reading them per row could see
        // them change halfway through one repaint, which is how a ctrl-drag ends up
        // half-applied.
        let mods = ui.input(|i| i.modifiers);
        let click = self.entry_list(ui, mods.command || mods.ctrl, mods.shift);

        if let Some(idx) = click.toggle {
            let p = self.entries[idx].clone();
            self.anchor = Some(idx);
            match self.picked.iter().position(|s| *s == p) {
                // Ctrl-clicking something already selected takes it out, which is what
                // makes a mis-click recoverable.
                Some(i) => {
                    self.picked.remove(i);
                }
                None => self.picked.push(p),
            }
        }
        if let Some(idx) = click.range {
            let from = self.anchor.unwrap_or(idx);
            let (a, b) = if from <= idx {
                (from, idx)
            } else {
                (idx, from)
            };
            let mut got: Vec<PathBuf> = Vec::new();
            for p in &self.entries[a..=b] {
                // Folders are skipped: they cannot go in the queue, and a range that
                // appeared to select one would look like it worked and then drop it.
                if !p.is_dir() {
                    got.push(p.clone());
                }
            }
            if !got.is_empty() {
                self.picked = got;
            }
        }

        // A button press wins over a double-click recorded in the same frame only when
        // both happened, which cannot really occur; either way the list is the answer.
        let confirm = self.confirm_bar(ui).or(click.confirm);
        ui.weak(
            "click selects · ctrl adds or removes · shift takes the range · double-click \
             confirms · a folder opens on click",
        );

        if let Some(v) = confirm {
            self.message = None;
            on_pick(v);
        }
        if let Some(p) = click.enter {
            self.goto(p);
        }
        if let Some(m) = &self.message {
            ui.colored_label(Color32::from_rgb(226, 140, 90), m.clone());
        }
    }

    /// Drives and volumes, plus the two shortcuts that always make sense.
    ///
    /// This sits above everything else because reaching a USB stick should not require
    /// knowing that it mounted under `/run/media/<user>/`, let alone typing that.
    fn drive_bar(&mut self, ui: &mut egui::Ui) {
        let mut again = false;
        ui.horizontal_wrapped(|ui| {
            if ui
                .small_button("⌂ home")
                .on_hover_text("home folder")
                .clicked()
            {
                let h = dirs_home();
                self.goto(h);
            }
            if ui
                .small_button("⬆ up")
                .on_hover_text("one folder up")
                .clicked()
            {
                if let Some(p) = self.path.parent() {
                    let p = p.to_path_buf();
                    self.goto(p);
                }
            }
            if ui
                .small_button("↻ drives")
                .on_hover_text("re-read the mounted drives and volumes")
                .clicked()
            {
                again = true;
            }
            ui.separator();
            // Cloned: the loop writes `self.path` while reading the list.
            for m in self.mounts.clone() {
                let here = m.path == self.path;
                let hover = if m.fstype.is_empty() {
                    m.path.display().to_string()
                } else {
                    format!("{} · {}", m.path.display(), m.fstype)
                };
                if ui
                    .add(egui::SelectableLabel::new(here, format!("⛁ {}", m.label)))
                    .on_hover_text(hover)
                    .clicked()
                {
                    self.goto(m.path);
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let before = self.sort;
                let cb = egui::ComboBox::from_id_salt("browser-sort")
                    .selected_text(format!("sort: {}", self.sort.label()))
                    .show_ui(ui, |ui| {
                        for s in Sort::ALL {
                            ui.selectable_value(&mut self.sort, s, s.label());
                        }
                    });
                cb.response.on_hover_text(
                    "Newest first, folders first, ties broken by name. \"created\" is the \
                     birth time where the filesystem records it; where it does not, the \
                     modified date is used and the picker says so above the list.",
                );
                if before != self.sort {
                    self.resort();
                }
                ui.checkbox(&mut self.filter_audio, "audio only");
            });
        });
        if again {
            self.mounts = mounts();
        }
    }

    /// The editable path. Returns true when "go" was pressed.
    fn path_bar(&mut self, ui: &mut egui::Ui) -> bool {
        let mut go = false;
        ui.horizontal(|ui| {
            ui.add_sized(
                [ui.available_width() - 60.0, 0.0],
                egui::TextEdit::singleline(&mut self.path_text).font(FontId::monospace(11.0)),
            );
            if ui.button("go").clicked() {
                go = true;
            }
        });
        go
    }

    /// The entry list. Returns what the click on it means; the selection itself is
    /// applied by the caller, because the list borrows `self.entries`.
    fn entry_list(&mut self, ui: &mut egui::Ui, adding: bool, ranging: bool) -> Click {
        let mut click = Click::default();
        egui::ScrollArea::vertical()
            .auto_shrink([false, true])
            .max_height(280.0)
            .show(ui, |ui| {
                for (idx, p) in self.entries.iter().enumerate() {
                    let is_dir = p.is_dir();
                    let name = p
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| p.display().to_string());
                    let is_sel = self.picked.iter().any(|s| s == p);
                    let icon = if is_dir { "🗀" } else { "·" };
                    let resp = ui
                        .selectable_label(is_sel, format!("{icon} {name}"))
                        .on_hover_text(p.display().to_string());
                    if resp.clicked() {
                        if is_dir {
                            // A folder is opened, not selected: the queue takes files,
                            // and a folder is added with the main window's own button.
                            click.enter = Some(p.clone());
                        } else if ranging {
                            click.range = Some(idx);
                        } else if adding {
                            click.toggle = Some(idx);
                        } else {
                            self.picked = vec![p.clone()];
                            self.anchor = Some(idx);
                        }
                    }
                    if resp.double_clicked() {
                        if is_dir {
                            click.enter = Some(p.clone());
                        } else {
                            self.picked = vec![p.clone()];
                            self.anchor = Some(idx);
                            click.confirm = Some(self.picked.clone());
                        }
                    }
                }
                if self.entries.is_empty() {
                    ui.weak("(empty)");
                }
            });
        click
    }

    /// The button that commits the selection. `Mode::Dir` commits the folder being
    /// looked at, which is what "write outputs to" is asking for.
    fn confirm_bar(&mut self, ui: &mut egui::Ui) -> Option<Vec<PathBuf>> {
        let title = match self.mode {
            Mode::AnyFile => "file",
            Mode::AudioFile => "audio file",
            Mode::Dir => "folder",
        };
        let mut out = None;
        ui.horizontal(|ui| {
            let n = self.picked.len();
            let can = match self.mode {
                Mode::Dir => self.path.is_dir(),
                _ => n > 0,
            };
            let label = match self.mode {
                Mode::Dir => format!("choose {title}"),
                _ if n > 1 => format!("choose {n} files"),
                _ => format!("choose {title}"),
            };
            if ui.add_enabled(can, egui::Button::new(label)).clicked() {
                out = Some(match self.mode {
                    Mode::Dir => vec![self.path.clone()],
                    _ => self.picked.clone(),
                });
            }
            if n > 0 {
                // Said out loud, so a shift-drag can be checked before committing.
                ui.weak(format!("{n} selected"));
            }
        });
        out
    }

    /// Move to another folder, dropping the entries cache and the selection anchor:
    /// a range from the previous folder has no meaning here.
    fn goto(&mut self, p: PathBuf) {
        self.path = p;
        self.entries.clear();
        self.entries_for = PathBuf::new();
        self.anchor = None;
        self.refresh();
    }

    /// Show a path as already selected, e.g. the IR currently configured.
    pub fn set_single(&mut self, p: Option<PathBuf>) {
        self.picked = p.into_iter().collect();
        self.anchor = None;
    }
}

/// A short path with the middle elided, so a deep library tree does not push the
/// whole layout sideways.
pub fn short_path(p: &Path, max: usize) -> String {
    let s = p.display().to_string();
    // Chars, not bytes: a path with a non-ASCII name is routine in a music library,
    // and byte slicing would panic on a codepoint boundary.
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max || max < 4 {
        return s;
    }
    // Elide the middle. The tail gets the larger share because the tail is what
    // identifies a file — its name and its extension.
    let budget = max - 1;
    let tail_n = (budget * 2).div_ceil(3);
    let head_n = budget - tail_n;
    let head: String = chars[..head_n].iter().collect();
    let tail: String = chars[chars.len() - tail_n..].iter().collect();
    format!("{head}…{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use g6me_core::meter::{IN_LABELS, OUT_LABELS, TOTAL};

    #[test]
    fn ten_meters_as_the_plugin_had_them() {
        assert_eq!(IN_LABELS.len() + OUT_LABELS.len(), TOTAL);
        assert_eq!(
            &IN_LABELS[..],
            ["In1 FL", "In2 FR", "In3 FC", "In4 LFE", "In5 BL", "In6 BR", "In7 SL", "In8 SR"]
        );
        assert_eq!(&OUT_LABELS[..], ["OutL", "OutR"]);
    }

    #[test]
    fn db_grid_maps_to_the_meter_parameter_space() {
        // The strip must agree with the meters about where 0 and -60 sit, or the
        // bars and their own scale disagree.
        assert_eq!(db_to_param(0.0), 1.0);
        assert_eq!(db_to_param(-60.0), 0.0);
        assert_eq!(db_to_param(-30.0), 0.5);
        assert_eq!(db_to_param(-200.0), 0.0, "clamped at the floor");
        assert_eq!(db_to_param(6.0), 1.0, "and at full scale");
        for db in DB_LINES {
            let p = db_to_param(db);
            assert!((0.0..=1.0).contains(&p), "{db} dB -> {p}");
        }
    }

    fn rect(w: f32, h: f32) -> Rect {
        Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(w, h))
    }

    #[test]
    fn outputs_follow_the_inputs_across_a_bounded_split() {
        // The pair belongs on the right of the row but next to the inputs, not flung
        // to the far edge of a wide window where the two halves stop reading as one
        // meter.
        let g = Geometry::layout(rect(1100.0, 220.0));
        let last_in = g.bar(IN_CHANNELS - 1);
        let first_out = g.bar(IN_CHANNELS);
        assert!(first_out.left() > last_in.right());
        assert!(
            first_out.left() - last_in.right() <= SPLIT,
            "the split grew to {} px",
            first_out.left() - last_in.right()
        );
        assert!(g.split_x > last_in.right() && g.split_x < first_out.left());
        assert_eq!(g.bar_x(0), g.in_x, "inputs start after the dB column");
        // The row is much narrower than the card, and centred in it.
        let row_w = g.row_right - g.row_left;
        assert!(
            row_w < 400.0,
            "the row should stay compact, it is {row_w} px"
        );
        let slack_l = g.row_left - 6.0;
        let slack_r = 1100.0 - 6.0 - g.row_right;
        assert!(
            (slack_l - slack_r).abs() < 1.0,
            "the leftovers should sit evenly outside the row: {slack_l} vs {slack_r}"
        );
    }

    #[test]
    fn a_small_window_gets_a_card_not_an_overlap() {
        // 640x120 is a plausible small laptop window. Ten meters plus a split still
        // have to fit without bars landing on each other.
        let g = Geometry::layout(rect(640.0, 120.0));
        for i in 1..TOTAL {
            assert!(
                g.bar(i).left() >= g.bar(i - 1).right(),
                "bar {i} overlaps bar {}: {:?} {:?}",
                i - 1,
                g.bar(i - 1),
                g.bar(i)
            );
        }
        for i in 0..TOTAL {
            let b = g.bar(i);
            assert!(
                b.left() >= 0.0 && b.right() <= 640.0,
                "bar {i} leaves the card: {b:?}"
            );
            assert!(
                b.top() >= 0.0 && b.bottom() <= 120.0,
                "bar {i} leaves the card: {b:?}"
            );
            assert!(
                b.width() > 0.0 && b.height() > 8.0,
                "bar {i} is unusable: {b:?}"
            );
        }
        assert!(g.bar(IN_CHANNELS).left() > g.bar(IN_CHANNELS - 1).right());
    }

    #[test]
    fn bars_stay_thin_and_packed_however_big_the_window() {
        // The complaint this answers: bars that grew with the window until ten of
        // them were swatches with a fader's worth of air between them.
        let big = Geometry::layout(rect(2400.0, 260.0));
        assert!(big.bar_w <= BAR_MAX, "{} px bar", big.bar_w);
        assert!(big.gap <= GAP_MAX, "{} px gap", big.gap);
        assert!(big.row_right - big.row_left < 400.0);
        // Height, not width, is what a taller window buys — that is the axis a level
        // is read on.
        let short = Geometry::layout(rect(2400.0, 120.0));
        assert!(big.bar_height() > short.bar_height());
        // 360 px is already enough for the full-size row, so it stays at the cap.
        let mid = Geometry::layout(rect(360.0, 140.0));
        assert_eq!(
            mid.bar_w, BAR_MAX,
            "{} px should fit the row at full size",
            360
        );
        // A genuinely tight window squeezes the bars, not the split between groups.
        let tight = Geometry::layout(rect(240.0, 140.0));
        assert!(
            tight.bar_w < BAR_MAX,
            "tight bars should shrink: {}",
            tight.bar_w
        );
        let last_in = tight.bar(IN_CHANNELS - 1);
        assert!(
            tight.bar(IN_CHANNELS).left() - last_in.right() >= 8.0,
            "the split must survive a squeeze"
        );
        for i in 1..TOTAL {
            assert!(tight.bar(i).left() >= tight.bar(i - 1).right());
        }
    }

    #[test]
    fn unescape_mount_decodes_the_kernels_escapes() {
        // The kernel writes a space in a mount point as \040. A USB stick called
        // "My Music" is the ordinary case, not a curiosity.
        assert_eq!(unescape_mount("/mnt/My\\040Music"), "/mnt/My Music");
        assert_eq!(
            unescape_mount("/run/media/u/Disk\\0111"),
            "/run/media/u/Disk\t1"
        );
        assert_eq!(unescape_mount("/mnt/back\\134slash"), "/mnt/back\\slash");
        assert_eq!(unescape_mount("/plain"), "/plain");
        // An unknown sequence must not eat characters.
        assert_eq!(unescape_mount("/x\\999y"), "/x\\999y");
    }

    /// The boot partition and the scratch filesystems are not places a library lives.
    ///
    /// Written as mountinfo rather than as calls to `show_mount` because the point is the
    /// list a person sees: `/boot` as vfat and `/tmp` as ext4 pass every filesystem test
    /// there is, which is how they ended up as buttons on a real machine.
    #[test]
    fn the_scratch_mounts_are_not_shortcuts() {
        let text = "\
1 1 0:1 / / rw,relatime - f2fs /dev/sda1
2 1 0:2 / /boot rw,relatime - vfat /dev/nvme0n1p1
3 1 0:3 / /boot/efi rw,relatime - vfat /dev/nvme0n1p12
4 1 0:4 / /tmp rw,relatime - ext4 /dev/sda3
5 1 0:5 / /tmp/build rw,relatime - ext4 /dev/sda3
6 1 0:6 / /var/tmp rw,relatime - ext4 /dev/sda4
7 1 0:7 / /run/media/sanjxz/data rw,relatime - f2fs /dev/sdb1
8 1 0:8 / /mnt/aosp rw,relatime - ext4 /dev/sdc1
";
        let mut got: Vec<String> = parse_mountinfo_with(text, |_| true)
            .into_iter()
            .map(|m| m.path.display().to_string())
            .collect();
        got.sort();
        assert_eq!(
            got,
            ["/", "/mnt/aosp", "/run/media/sanjxz/data"],
            "a scratch mount is still taking a button"
        );
    }

    /// The order the picker draws, without a filesystem to write timestamps onto.
    #[test]
    fn the_picker_sorts_by_date_and_keeps_folders_first() {
        use std::time::Duration;
        let older = std::time::SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let newer = std::time::SystemTime::UNIX_EPOCH + Duration::from_secs(9_000);
        let entry = |n: &str,
                     is_dir: bool,
                     modified: Option<std::time::SystemTime>,
                     created: Option<std::time::SystemTime>| Entry {
            path: PathBuf::from(n),
            fold: n.to_lowercase(),
            is_dir,
            modified,
            created,
        };
        let mut b = Browser {
            sort: Sort::Modified,
            listing: vec![
                entry("old.mp3", false, Some(older), Some(older)),
                entry("new.mp3", false, Some(newer), Some(newer)),
                entry("zoo", true, Some(newer), Some(newer)),
                entry("nodate.mp3", false, None, None),
            ],
            ..Browser::default()
        };
        let names = |b: &Browser| -> Vec<String> {
            b.entries
                .iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
                .collect()
        };

        // Folders first, then newest to oldest, and an entry the filesystem would not
        // date last rather than first: "no date" is a fact about the filesystem.
        b.sort = Sort::Modified;
        b.resort();
        assert_eq!(names(&b), ["zoo", "new.mp3", "old.mp3", "nodate.mp3"]);

        // Birth time where there is one; the modified date is the honest answer where
        // there is not, and the order must not shuffle when that happens.
        for e in b.listing.iter_mut() {
            e.created = None;
        }
        b.sort = Sort::Created;
        b.resort();
        assert_eq!(names(&b), ["zoo", "new.mp3", "old.mp3", "nodate.mp3"]);

        // And the name order is the one the list had before dates existed: folders first,
        // then A to Z.
        b.sort = Sort::Name;
        b.resort();
        assert_eq!(
            names(&b),
            ["zoo", "new.mp3", "nodate.mp3", "old.mp3"],
            "the default order changed"
        );
    }

    #[test]
    fn mount_labels_name_volumes_not_paths() {
        assert_eq!(mount_label("/run/media/sanjxz/SANDISK"), "SANDISK");
        assert_eq!(mount_label("/mnt/nas"), "nas");
        assert_eq!(mount_label("/"), "root");
        assert_eq!(mount_label("/data/music"), "music");
    }

    #[test]
    fn mountinfo_keeps_media_and_drives_and_drops_plumbing() {
        // A trimmed real mountinfo: the interesting four, plus the noise a desktop
        // actually has hundreds of.
        let text = "\
18 25 8:1 / / rw,relatime - ext4 /dev/sda1 rw
19 18 8:2 / /boot rw,relatime - vfat /dev/sda2 rw
20 18 0:45 / /sys rw,nosuid - sysfs sysfs rw
21 18 0:46 / /run rw,nosuid,nodev - tmpfs tmpfs rw
22 18 0:61 / /snap/bare/5 ro - squashfs /dev/loop6 ro
23 18 0:70 / /var/lib/docker/overlay2/x/merged rw - overlay overlay rw
24 18 0:80 / /run/user/1000/doc rw - fuse.portal portal rw
25 18 8:33 / /run/media/sanjxz/SANDISK rw,nosuid - vfat /dev/sdb1 rw,uid=1000
26 18 8:49 / /mnt/My\\040Music rw - ext4 /dev/sdc1 rw
27 18 0:99 / /mnt/gone rw - ext4 /dev/sdd1 rw
28 18 0:33 / /run/user/1000/gvfs rw - fuse.gvfsd-fuse gvfsd-fuse rw
7 18 0:40 / /mnt/nas rw - cifs //server/music rw
";
        // "gone" is listed but no longer reachable — an unmounted stick the kernel
        // still remembers would open as an empty folder.
        let got = parse_mountinfo_with(text, |p| p.to_string_lossy() != "/mnt/gone");
        let labels: Vec<&str> = got.iter().map(|m| m.label.as_str()).collect();
        // Alphabetical and case-insensitive, because buttons that reorder themselves
        // between clicks are unreadable. `/boot` used to be in this list on the reasoning
        // that a real partition on a real disk deserves a shortcut; it does not. It holds
        // kernels and an initramfs, it is the first button a person who is looking for
        // music has to ignore, and the path is still typeable.
        assert_eq!(
            labels,
            vec!["My Music", "nas · music", "root", "SANDISK"],
            "got {got:?}"
        );
        // Root is named usefully, the escaped path survived, and the network share
        // says where it comes from.
        let by_label = |l: &str| got.iter().find(|m| m.label == l).unwrap();
        assert_eq!(by_label("My Music").path, PathBuf::from("/mnt/My Music"));
        assert_eq!(by_label("SANDISK").fstype, "vfat");
        assert_eq!(by_label("root").path, PathBuf::from("/"));
        assert_eq!(by_label("nas · music").fstype, "cifs");
        // Exact paths, because `/run/media/…` must survive while `/run` and
        // `/run/user/…` must not — a prefix test on "/run" would fail for the stick.
        for junk in [
            "/snap/bare/5",
            "/var/lib/docker/overlay2/x/merged",
            "/sys",
            "/run",
            "/run/user/1000/doc",
            "/run/user/1000/gvfs",
            "/mnt/gone",
        ] {
            assert!(
                !got.iter().any(|m| m.path == Path::new(junk)),
                "{junk} should be filtered out"
            );
        }
    }

    #[test]
    fn a_machine_with_no_mounts_is_not_an_error() {
        // An unreadable or empty mountinfo means no buttons, not a broken browser.
        assert!(parse_mountinfo_with("", |_| true).is_empty());
        assert!(parse_mountinfo_with("garbage without fields\n", |_| true).is_empty());
        // A line missing the "-" separator is skipped rather than mis-parsed.
        assert!(parse_mountinfo_with("1 2 3:4 / /x rw -", |_| true).is_empty());
    }

    #[test]
    fn bars_are_labelled_with_the_short_names() {
        // 20 px of bar cannot hold "In3 FC", and the full slot names are in the queue
        // table and the group headings.
        assert_eq!(SHORT_LABELS.len(), TOTAL);
        assert_eq!(
            SHORT_LABELS[..8],
            ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"]
        );
        assert_eq!(SHORT_LABELS[IN_CHANNELS..], ["L", "R"]);
    }

    #[test]
    fn short_path_elides_the_middle_not_the_end() {
        let p = Path::new("/music/Artists/Album/Very Long Track Name.flac");
        let s = short_path(p, 20);
        assert!(s.contains('…'), "{s}");
        assert!(
            s.ends_with("Name.flac"),
            "the tail is what identifies it: {s}"
        );
        assert!(s.chars().count() <= 21);
        assert_eq!(short_path(Path::new("/a.flac"), 40), "/a.flac");
    }

    #[test]
    fn browser_lists_a_real_directory() {
        let mut b = Browser {
            mode: Mode::Dir,
            filter_audio: false,
            path: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
            ..Default::default()
        };
        b.entries.clear();
        b.refresh();
        assert!(
            b.entries
                .iter()
                .any(|p| p.file_name() == Some(std::ffi::OsStr::new("src"))),
            "the crate's own src dir should be listed"
        );
        // Audio filtering must actually filter: this folder holds .rs files only.
        // Folders stay listed whatever the mode — without them there is no way to
        // navigate to the folder that does hold audio.
        b.filter_audio = true;
        b.mode = Mode::AudioFile;
        b.entries.clear();
        b.entries_for = PathBuf::new();
        b.refresh();
        let stray: Vec<_> = b
            .entries
            .iter()
            .filter(|p| !p.is_dir() && !batch::looks_like_audio(p))
            .collect();
        assert!(stray.is_empty(), "audio-only mode listed {stray:?}");
        assert!(
            b.entries.iter().any(|p| p.is_dir()),
            "a picker that cannot navigate is not a picker"
        );
    }

    #[test]
    fn unreadable_folder_reports_instead_of_panicking() {
        let mut b = Browser {
            path: PathBuf::from("/definitely/not/here"),
            ..Default::default()
        };
        b.entries.clear();
        b.entries_for = PathBuf::new();
        b.refresh();
        assert!(b.entries.is_empty());
        assert!(
            b.message.is_some(),
            "the user needs to know why it is empty"
        );
    }
}
