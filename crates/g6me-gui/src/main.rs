//! G6 Media Encoder — desktop front end.
//!
//! Everything here is presentation. The queue, the render, the filters, the
//! meters' release curve all live in `g6me-core`, which is the same library the CLI
//! drives, so a setting that behaves one way in the CLI behaves that way here. Two
//! things in particular are worth stating because they are the reasons this window
//! exists:
//!
//! * **the levels are the render's own.** Each file that starts rendering gets a
//!   meter, and the render feeds it the block it handed the encoder. There is no
//!   second "just measure it" path to drift out of agreement with the real one.
//! * **this window never makes a sound.** It writes files and it measures them. No
//!   audio device is opened anywhere in this crate, so nothing here can put a
//!   downmix into someone's headphones by accident, and the test suite is silent by
//!   construction rather than by convention.
//!
//! The layout is deliberately fixed — queue left, controls centre, meters and
//! progress along the bottom — because a mixer's muscle memory is the point.

// A released Windows GUI must not drag a console window along with it. Debug builds
// keep the console, because a panic that prints nowhere is a worse trade than one
// extra window while developing.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod runner;
mod settings;
mod theme;
mod widgets;

use crate::runner::{Event, Level, Runner, Summary};
use crate::settings::{describe, Format, Settings};
use crate::theme::Palette;
use crate::widgets::{Browser, Mode};
use eframe::CreationContext;
use g6me_core::batch::{self, Phase};
use g6me_core::mask;
use g6me_core::meter::{MeterTap, Meters};
use g6me_core::pcm::{BitDepth, Dither};
use g6me_core::pipeline;
use g6me_core::pipeline::Report;
use g6me_core::probe::{probe_caps, Caps, Tools};
use g6me_core::stages::Gtype;
use g6me_core::util;
use std::path::PathBuf;
use std::sync::Mutex;

/// Which settings page is showing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Output,
    TruePeak,
    Resample,
    Ir,
    Advanced,
}

impl Tab {
    const ALL: [Tab; 5] = [
        Tab::Output,
        Tab::TruePeak,
        Tab::Resample,
        Tab::Ir,
        Tab::Advanced,
    ];

    fn label(self) -> &'static str {
        match self {
            Tab::Output => "Output",
            Tab::TruePeak => "True peak",
            Tab::Resample => "Resample",
            Tab::Ir => "IR calibration",
            Tab::Advanced => "Advanced",
        }
    }
}

/// One meter per file that is rendering, in the order the workers picked them up.
///
/// Per file rather than one for the app: four files sharing a meter would show the
/// loudest of four different pieces of music, which is a level that belongs to none
/// of them.
#[derive(Clone, Default)]
struct Cards(std::sync::Arc<std::sync::Mutex<Vec<CardState>>>);

struct CardState {
    index: usize,
    meters: std::sync::Arc<Meters>,
    finished: bool,
}

impl Cards {
    /// The meter for file `index`, creating it the first time that file starts.
    fn open(&self, index: usize) -> std::sync::Arc<Meters> {
        let mut v = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(c) = v.iter_mut().find(|c| c.index == index) {
            c.finished = false;
            return c.meters.clone();
        }
        let m = std::sync::Arc::new(Meters::new());
        v.push(CardState {
            index,
            meters: m.clone(),
            finished: false,
        });
        m
    }

    fn finish(&self, index: usize) {
        if let Some(c) = self
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter_mut()
            .find(|c| c.index == index)
        {
            c.finished = true;
        }
    }

    fn list(&self) -> Vec<(usize, std::sync::Arc<Meters>, bool)> {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|c| (c.index, c.meters.clone(), c.finished))
            .collect()
    }

    fn clear(&self) {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).clear();
    }

    fn live(&self) -> usize {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .filter(|c| !c.finished)
            .count()
    }
}

/// Which browser the window is showing, if any.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pick {
    /// Files and folders to queue.
    Queue,
    OutDir,
    Ir,
}

/// One row of the queue table.
struct Row {
    src: PathBuf,
    dst: PathBuf,
    phase: Phase,
    note: String,
    /// Why this file's destination is not the obvious one, from discovery.
    ///
    /// Kept apart from `note` on purpose: `note` is overwritten by the runner the moment
    /// the file starts, and a file that had to be written somewhere other than where its
    /// name pointed is worth knowing before it runs, not just after.
    collision: String,
}

impl Row {
    /// The one line to show beside the file name: the result if the file has run, and
    /// otherwise the reason its output path moved.
    fn shout(&self) -> &str {
        if self.note.is_empty() {
            &self.collision
        } else {
            &self.note
        }
    }

    fn tooltip(&self) -> String {
        let mut s = format!("\u{2192} {}", self.dst.display());
        if !self.collision.is_empty() {
            s.push('\n');
            s.push_str(&self.collision);
        }
        if !self.note.is_empty() {
            s.push('\n');
            s.push_str(&self.note);
        }
        s
    }
}

pub struct App {
    settings: Settings,
    tools: Tools,
    /// Filled in by a background probe: `probe_caps` runs ffmpeg several times and
    /// a window that appears a second later is better than one that appears frozen.
    caps: Mutex<Option<Caps>>,
    caps_note: Mutex<Option<String>>,
    runner: Runner,
    cards: Cards,
    palette: Palette,
    /// What is installed in the context right now, so the palette is rewritten on a
    /// change rather than sixty times a second.
    installed: Option<Palette>,
    tab: Tab,
    queue: Vec<Row>,
    pick: Option<Pick>,
    browser: Browser,
    log: Vec<(Level, String)>,
    summary: Option<Summary>,
    /// Every file that has finished this session, with the levels it was measured at.
    ///
    /// This is the reason the report carries true peak, sample peak and RMS at all. The
    /// live bars fall to rest in a third of a second, which is what a bar should do and
    /// the opposite of what a record needs: after a render the user asks "did that one
    /// come in over the ceiling", and the answer has to still be on the screen. Kept
    /// across runs, cleared by hand, newest first.
    history: Vec<MeterRow>,
    status: String,
    show_caps: bool,
    dry_run: Mutex<Option<DryRun>>,
    log_scroll: bool,
    last_repaint: std::time::Instant,
}

/// A planned render, shown without writing anything.
#[derive(Clone)]
struct DryRun {
    src: PathBuf,
    lines: Vec<String>,
    error: Option<String>,
}

/// One finished file, as the meters remembered it.
///
/// The strings are built once, when the file finishes, rather than formatted every frame:
/// a queue of two thousand files redraws sixty times a second and should not be asked to
/// re-render two thousand float formats to do it.
struct MeterRow {
    /// Compact levels for the list row.
    levels: String,
    /// The same measurement in full, for copying.
    text: String,
    /// Came in at or over 0 dBTP, or failed to render.
    hot: bool,
    ok: bool,
}

/// A dB figure the user can read, including the one value that has no logarithm.
fn db(v: f64) -> String {
    if v == f64::NEG_INFINITY {
        "−inf".to_string()
    } else if !v.is_finite() {
        "n/a".to_string()
    } else {
        format!("{v:+.2}")
    }
}

impl MeterRow {
    fn from_report(r: &Report) -> Self {
        let name = r
            .output
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let list = |v: &[f64]| {
            if v.is_empty() {
                "—".to_string()
            } else {
                v.iter().map(|x| db(*x)).collect::<Vec<_>>().join("/")
            }
        };
        let mut levels = format!(
            "{} — tp {} dBTP @{} · peak {} dBFS · rms {} dBFS",
            name,
            db(r.true_peak_dbtp),
            r.true_peak_ratio.label(),
            list(&r.sample_peak_dbfs),
            list(&r.rms_dbfs)
        );
        // DC follows the rule the CLI headline uses: measured on every file, printed in the
        // full copy text on every file, and on the row a person scans only when some channel
        // is over the threshold where an offset is a finding instead of a floor.
        if let Some(n) = Report::dc_notice(&r.dc_offset) {
            levels.push_str(" · ");
            levels.push_str(&n);
        }
        let hot = r.true_peak_dbtp > -0.05
            || r.limiter.as_ref().is_some_and(|l| !l.within_ceiling())
            // A file whose own decoded measurement disagrees with the stream is as urgent
            // as one that overhangs the ceiling: the headline number is not the number on
            // disk, and the row that says "-3.61 dBTP" about a file that clamped to zero is
            // the row nobody should have to read closely to distrust.
            || r.delivered_note.is_some();
        Self {
            levels,
            text: format!("{}\n{}", name, r.peak_line()),
            hot,
            ok: true,
        }
    }
}

impl App {
    fn new(cc: &CreationContext<'_>) -> Self {
        let mut app = Self {
            settings: Settings::default(),
            tools: Tools::discover(),
            caps: Mutex::new(None),
            caps_note: Mutex::new(None),
            runner: Runner::new(),
            cards: Cards::default(),
            palette: Palette::default(),
            installed: None,
            tab: Tab::Output,
            queue: Vec::new(),
            pick: None,
            browser: Browser::default(),
            log: Vec::new(),
            summary: None,
            history: Vec::new(),
            status: "idle".to_string(),
            show_caps: false,
            dry_run: Mutex::new(None),
            log_scroll: true,
            last_repaint: std::time::Instant::now(),
        };
        app.runner.set_context(cc.egui_ctx.clone());
        // Palette before the first frame: the window should not flash dark while a
        // light-mode desktop asks for light.
        app.palette.apply(&cc.egui_ctx);
        app.installed = Some(app.palette);
        app.push_log(
            Level::Info,
            format!(
                "G6 Media Encoder — {} cores, up to {} files at once",
                util::cpu_count(),
                Settings::max_threads()
            ),
        );
        // Probe capabilities off the UI thread, reporting through the same channel the
        // batch uses. Until it lands, Start stays disabled: offering "24-bit FLAC" on a
        // machine with no reference encoder is exactly the mistake this project already
        // made once and had to unwind.
        let tools = app.tools.clone();
        let tx = app.runner.sender();
        let ctx = cc.egui_ctx.clone();
        std::thread::spawn(move || {
            let caps = probe_caps(&tools);
            let note = caps.summary();
            let _ = tx.send(Event::Caps(caps, note));
            ctx.request_repaint();
        });
        app
    }

    fn caps(&self) -> Option<Caps> {
        self.caps.lock().ok().and_then(|c| c.clone())
    }

    fn push_log(&mut self, level: Level, msg: impl Into<String>) {
        let msg = msg.into();
        if msg.trim().is_empty() {
            return;
        }
        // A batch can emit hundreds of lines; keep the tail and say so.
        const KEEP: usize = 400;
        self.log.push((level, msg));
        if self.log.len() > KEEP {
            let dropped = self.log.len() - KEEP;
            self.log.drain(..dropped);
            self.log.insert(
                0,
                (Level::Info, format!("… {dropped} earlier line(s) dropped")),
            );
        }
    }

    // ---- panels ------------------------------------------------------------

    fn top_panel(&mut self, ctx: &egui::Context) {
        let caps = self.caps();
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("G6 Media Encoder");
                ui.separator();
                match &caps {
                    None => {
                        ui.weak("probing ffmpeg…");
                    }
                    Some(c) => {
                        let flag = |b: bool| if b { "yes" } else { "NO" };
                        ui.label(format!(
                            "ffmpeg {} · soxr {} · afir {} · flac {} · 24-bit {}",
                            c.ffmpeg_version,
                            flag(c.soxr),
                            flag(c.afir),
                            flag(c.flac),
                            flag(c.flac24),
                        ));
                        if ui.link("details").clicked() {
                            self.show_caps = !self.show_caps;
                        }
                    }
                }
            });
            ui.horizontal(|ui| {
                let running = self.runner.is_running();
                let ready = caps.is_some() && !self.queue.is_empty();
                let start = ui.add_enabled(!running && ready, egui::Button::new("▶ Start"));
                if start.clicked() {
                    self.start_run();
                }
                if ui
                    .add_enabled(running, egui::Button::new("■ Cancel"))
                    .on_hover_text(
                        "Stops after the current block and removes partial outputs, so a \
                         cancelled run leaves nothing half-written.",
                    )
                    .clicked()
                {
                    self.runner.cancel();
                    self.status = "cancelling…".to_string();
                }
                ui.separator();
                let mut t = self.settings.threads as u32;
                // The ceiling is the shared one from `util`, so the slider cannot offer
                // a number the CLI would reject.
                let max = Settings::max_threads() as u32;
                ui.add_enabled(
                    !running,
                    egui::Slider::new(&mut t, 1..=max.max(1)).text("files at once"),
                );
                self.settings.threads = util::clamp_threads(t as usize);
                ui.separator();
                ui.monospace(describe(&self.settings));
                ui.separator();
                // Here rather than in a tab: it is a window property, like the title
                // bar, not a rendering setting, and hunting for it in a page is the
                // worst place to look once the screen is bright enough to hurt.
                // "system" is not a colour, so the button says what it currently means.
                let cur = match self.palette {
                    Palette::Auto => format!(
                        "system ({})",
                        match self.palette.resolved(ctx) {
                            egui::Theme::Dark => "dark",
                            egui::Theme::Light => "light",
                        }
                    ),
                    p => p.label().to_string(),
                };
                egui::ComboBox::from_id_salt("palette")
                    .selected_text(format!("theme: {cur}"))
                    .show_ui(ui, |ui| {
                        for p in Palette::ALL {
                            ui.selectable_value(&mut self.palette, p, p.label());
                        }
                    })
                    .response
                    .on_hover_text(
                        "system follows the desktop's light/dark setting. pure black sets every \
                         large surface to #000 for OLED panels; widgets keep a faint fill so \
                         they stay visible.",
                    );
            });
            ui.add_space(3.0);
        });
    }

    fn bottom_panel(&mut self, ctx: &egui::Context) {
        let counters = self.runner.counters();
        let running = self.runner.is_running();
        egui::TopBottomPanel::bottom("bottom").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(c) = &counters {
                    let bar = egui::ProgressBar::new(c.fraction()).text(format!(
                        "{}/{} files · {} running",
                        c.done + c.failed + c.skipped,
                        c.total,
                        c.running
                    ));
                    ui.add(bar);
                    ui.separator();
                    ui.label(format!(
                        "{} · {} of audio rendered",
                        util::human_bytes(c.bytes),
                        util::human_time(c.seconds_done)
                    ));
                    // ETA from the observed throughput so far, using the average
                    // duration of a finished file for the ones still queued.
                    let finished = c.done + c.failed;
                    let per_item = if finished > 0 {
                        c.seconds_done / finished as f64
                    } else {
                        0.0
                    };
                    if let Some(eta) = self.runner.elapsed().and_then(|e| c.eta(e, per_item)) {
                        ui.weak(format!(
                            "about {} left",
                            util::human_time(eta.as_secs_f64())
                        ));
                    }
                } else {
                    ui.weak("nothing has run yet");
                }
                if let Some(s) = &self.summary {
                    let col = if s.failed > 0 {
                        egui::Color32::from_rgb(226, 140, 90)
                    } else {
                        egui::Color32::from_rgb(120, 190, 120)
                    };
                    ui.colored_label(col, s.one_line());
                }
            });
            ui.separator();
            let _ = running;
            self.meters_panel(ui, ctx);
        });
    }

    /// One bordered card per file that is rendering, in a scrollable list.
    ///
    /// The area takes a slice of the window height and the cards inside it share
    /// that slice: one file rendering gets a tall card, eight get eight shorter ones
    /// and the list starts scrolling. Nothing here is pinned at a size that only
    /// happens to suit the screen it was written on.
    fn meters_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let cards = self.cards.list();
        // Measured from the window, not the screen. `screen_rect` was the bug: on a
        // 4 K display it hands the meter strip 420 px whether the window is 1 400 px
        // tall or 500, and in a short window that is the entire UI.
        let win_h = ctx
            .input(|i| i.raw.viewport().inner_rect)
            .map(|r| r.height())
            .filter(|h| *h > 0.0)
            .unwrap_or(800.0);
        // A third of the window, but always leaving 300 px for the top bar, the queue
        // and the settings page, and never less than one readable card.
        let area = (win_h * 0.34)
            .clamp(96.0, 340.0)
            .min((win_h - 300.0).max(96.0));
        if cards.is_empty() {
            ui.vertical(|ui| {
                ui.set_height(area.min(48.0));
                ui.weak(
                    "no file is rendering — each file gets its own card with ten levels the \
                     moment it starts",
                );
            });
        } else {
            // Equal slices, with a floor: below about 84 px the bars stop being
            // readable and scrolling is better than squashing them.
            let card_h = (area / cards.len() as f32).clamp(84.0, 240.0);
            egui::ScrollArea::vertical()
                .max_height(area)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for (idx, meters, finished) in cards {
                        let (title, status) = self.card_label(idx, finished);
                        let card = widgets::Card {
                            title: &title,
                            status: &status,
                            finished,
                        };
                        // A card is per file, so clicking it clears that file's holds
                        // and nobody else's.
                        if widgets::meter_card(ui, &meters, &card, card_h) {
                            meters.reset();
                        }
                        ui.add_space(5.0);
                    }
                });
        }
        self.history_panel(ui, ctx);
        ui.horizontal(|ui| {
            ui.weak(widgets::caption(self.cards.live()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(self.status.clone());
            });
        });
    }

    /// The retained numbers, one line per finished file.
    ///
    /// Text rather than bars, deliberately: a bar is a memory of a moment and cannot be
    /// read off a screen two seconds later, copied into a delivery note, or compared
    /// between two runs. Every figure is what was measured on the bytes that were
    /// written, so this is a record of the output, not of the rendering.
    fn history_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if self.history.is_empty() {
            return;
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.weak(format!("finished files — {}", self.history.len()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("copy all")
                    .on_hover_text("Copy every line as text, newest last.")
                    .clicked()
                {
                    let text = self.history_text();
                    ctx.copy_text(text);
                }
                if ui
                    .small_button("clear")
                    .on_hover_text("Forget this session's history.")
                    .clicked()
                {
                    self.history.clear();
                }
            });
        });
        let rows = self.history.len().min(12) as f32;
        egui::ScrollArea::vertical()
            .max_height((rows * 17.0).clamp(34.0, 210.0))
            .auto_shrink([false, false])
            .stick_to_bottom(false)
            .show(ui, |ui| {
                // Drawn newest first: the file that just finished is the one being asked
                // about, and a list that grows downwards pushes it off screen.
                for row in self.history.iter().rev() {
                    ui.horizontal(|ui| {
                        ui.colored_label(widgets::status_dot(ui, row.hot, row.ok), "●");
                        // Selectable so a single line can be copied without the button.
                        ui.add(egui::Label::new(&row.levels).selectable(true));
                        if ui
                            .small_button("⧉")
                            .on_hover_text("Copy this file's full measurement as text.")
                            .clicked()
                        {
                            ctx.copy_text(row.text.clone());
                        }
                    });
                }
            });
    }

    /// The whole history as plain text, newest last, for a log or a delivery note.
    fn history_text(&self) -> String {
        let mut out = String::new();
        for row in self.history.iter().rev() {
            out.push_str(&row.text);
            out.push('\n');
        }
        out
    }

    /// Title and status for a card, read from the queue row: a card must never
    /// claim to be a different file than the table says it is.
    fn card_label(&self, index: usize, finished: bool) -> (String, String) {
        let Some(row) = self.queue.get(index) else {
            return (format!("file #{index}"), String::new());
        };
        let title = row
            .src
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| row.src.display().to_string());
        let status = if finished {
            truncate(row.shout(), 44)
        } else {
            "rendering".to_string()
        };
        (title, status)
    }

    fn queue_panel(&mut self, ctx: &egui::Context) {
        let running = self.runner.is_running();
        egui::SidePanel::left("queue")
            .default_width(360.0)
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.strong(format!("Queue ({})", self.queue.len()));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(!running, egui::Button::new("clear"))
                            .clicked()
                        {
                            self.queue.clear();
                            self.settings.inputs.clear();
                        }
                    });
                });
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!running, egui::Button::new("add audio files…"))
                        .clicked()
                    {
                        self.pick = Some(Pick::Queue);
                        self.browser.mode = Mode::AudioFile;
                        self.browser.set_single(None);
                    }
                    if ui
                        .add_enabled(!running, egui::Button::new("add folder…"))
                        .clicked()
                    {
                        self.pick = Some(Pick::Queue);
                        self.browser.mode = Mode::Dir;
                    }
                });
                ui.checkbox(&mut self.settings.recursive, "scan folders recursively");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.settings.use_out_dir, "write outputs to");
                    if ui
                        .add_enabled(!running, egui::Button::new("…"))
                        .on_hover_text("choose an output folder")
                        .clicked()
                    {
                        self.pick = Some(Pick::OutDir);
                        self.browser.mode = Mode::Dir;
                    }
                });
                if self.settings.use_out_dir {
                    ui.label(
                        egui::RichText::new(widgets::short_path(&self.settings.out_dir, 52))
                            .monospace()
                            .size(11.0),
                    );
                } else {
                    ui.weak("outputs go next to each input");
                }
                ui.separator();
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if self.queue.is_empty() {
                            ui.weak(
                                "Nothing queued.\n\nAdd files, or add a folder and the whole \
                                 tree is scanned for anything ffmpeg can decode — .mka \
                                 (TrueHD/Atmos), .wav, .flac, .mkv, .m4a…",
                            );
                            return;
                        }
                        let mut remove: Option<usize> = None;
                        for (i, r) in self.queue.iter().enumerate() {
                            let col = match r.phase {
                                Phase::Queued => egui::Color32::GRAY,
                                Phase::Running => egui::Color32::from_rgb(120, 170, 220),
                                Phase::Done => egui::Color32::from_rgb(120, 190, 120),
                                Phase::Failed => egui::Color32::from_rgb(226, 110, 90),
                                Phase::Skipped => egui::Color32::from_rgb(200, 170, 90),
                            };
                            let head = format!(
                                "{}  {}",
                                r.phase.label(),
                                r.src
                                    .file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| r.src.display().to_string())
                            );
                            let resp = ui
                                .horizontal(|ui| {
                                    ui.colored_label(col, head);
                                    if !r.shout().is_empty() {
                                        ui.weak(truncate(r.shout(), 40));
                                    }
                                    if !running && ui.small_button("✕").clicked() {
                                        remove = Some(i);
                                    }
                                })
                                .response
                                .on_hover_text(r.tooltip());
                            let _ = resp;
                        }
                        // Removed after the loop: the table is borrowed while it is
                        // being drawn.
                        if let Some(i) = remove {
                            let row = self.queue.remove(i);
                            self.settings.inputs.retain(|p| {
                                // A folder entry stays: dropping one file out of a
                                // scanned tree is not the same as un-queuing it.
                                *p != row.src && !p.is_dir()
                            });
                        }
                    });
            });
    }

    fn central_panel(&mut self, ctx: &egui::Context) {
        let caps = self.caps();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                for t in Tab::ALL {
                    ui.selectable_value(&mut self.tab, t, t.label());
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button("dry run")
                        .on_hover_text(
                            "Plan the first queued file: show the filter chain and the exact \
                             commands, write nothing.",
                        )
                        .clicked()
                    {
                        self.run_dry_run();
                    }
                });
            });
            ui.separator();
            // The log gets a fixed slice of the height and the settings scroll inside
            // what is left. Leaving this to `auto_shrink([false, false])` let the
            // settings area claim every pixel and pushed the log out of the window.
            const LOG_BUDGET: f32 = 168.0;
            let settings_h = (ui.available_height() - LOG_BUDGET).max(120.0);
            egui::ScrollArea::vertical()
                .max_height(settings_h)
                .auto_shrink([false, false])
                .show(ui, |ui| match self.tab {
                    Tab::Output => self.tab_output(ui, caps.as_ref()),
                    Tab::TruePeak => self.tab_true_peak(ui),
                    Tab::Resample => self.tab_resample(ui, caps.as_ref()),
                    Tab::Ir => self.tab_ir(ui, caps.as_ref()),
                    Tab::Advanced => self.tab_advanced(ui),
                });
            ui.separator();
            self.log_panel(ui);
        });
    }

    fn tab_output(&mut self, ui: &mut egui::Ui, caps: Option<&Caps>) {
        ui.horizontal(|ui| {
            ui.label("container");
            for f in Format::ALL {
                ui.selectable_value(&mut self.settings.format, f, f.label());
            }
            if self.settings.format == Format::Flac {
                ui.separator();
                ui.label("level");
                let mut l = self.settings.flac_level;
                // ffmpeg's FLAC encoder counts 0..=12 and refuses 13; the reference
                // flac CLI's -0..-8 presets are a different scale and not in use.
                if ui.add(egui::DragValue::new(&mut l).range(0..=12)).changed() {
                    self.settings.flac_level = l.min(12);
                }
                ui.weak("(0 fast … 12 smallest)");
            }
        });
        ui.horizontal(|ui| {
            ui.label("bit depth");
            egui::ComboBox::from_id_salt("depth")
                .selected_text(self.settings.depth.label())
                .show_ui(ui, |ui| {
                    for d in BitDepth::ALL {
                        let why = depth_note(d, self.settings.format, caps);
                        let label = match why {
                            Some(w) => format!("{} — {}", d.label(), w),
                            None => d.label().to_string(),
                        };
                        ui.selectable_value(&mut self.settings.depth, d, label);
                    }
                });
            if self.settings.depth == BitDepth::S24 && self.settings.format == Format::Flac {
                // Not a claim about a version string: the probe encodes a known
                // 24-bit block with this ffmpeg and checks the bytes come back.
                let ok = caps.map(|c| c.flac24).unwrap_or(true);
                ui.colored_label(
                    if ok {
                        egui::Color32::from_rgb(120, 190, 120)
                    } else {
                        egui::Color32::from_rgb(226, 110, 90)
                    },
                    if ok {
                        "24-bit FLAC verified lossless on this ffmpeg"
                    } else {
                        "this ffmpeg failed the 24-bit FLAC check — see capabilities"
                    },
                );
            }
        });
        if self.settings.format == Format::Flac {
            ui.weak(
                "FLAC carries 16- or 24-bit integers only: ffmpeg writes a genuine bps=24 \
                   stream, but it cannot carry 32-bit PCM or float — WAV does both.",
            );
        }
        ui.checkbox(&mut self.settings.copy_metadata, "copy source metadata");
        ui.checkbox(
            &mut self.settings.overwrite,
            "overwrite existing outputs (otherwise they are skipped)",
        );
        ui.checkbox(
            &mut self.settings.sha256,
            "hash each output (the sample stream, and the file as sha256sum sees it)",
        );
        ui.separator();
        ui.weak(
            "The internal maths is 64-bit float in both the downmixer and the convolver, and \
             that is not a choice: it is what makes the output match the reference build for \
             build. Bit depth here only decides how the result is written.",
        );
    }

    /// True-peak metering and limiting.
    ///
    /// The consequences shown here — guard band, latency, the lookahead actually used —
    /// are computed by the core from the same functions the render uses, not restated,
    /// because a panel that recalculates them is a panel that will quietly disagree with
    /// the file it produced.
    fn tab_true_peak(&mut self, ui: &mut egui::Ui) {
        use g6me_core::limiter::Cfg as LCfg;
        use g6me_core::tp::Ratio;

        // The rate the DSP runs at: the resampler's target when resampling is on, and
        // otherwise whatever the source is. Lookahead in ms and latency in samples trade
        // on that number, so it is stated rather than assumed.
        let rate = if self.settings.resample {
            self.settings.rate
        } else {
            48_000
        };
        ui.horizontal(|ui| {
            ui.label("measure at");
            egui::ComboBox::from_id_salt("tp-ratio")
                .selected_text(self.settings.tp_ratio.label())
                .show_ui(ui, |ui| {
                    for r in Ratio::ALL {
                        ui.selectable_value(
                            &mut self.settings.tp_ratio,
                            r,
                            format!(
                                "{} — {} MAC/sample, ≤{} dB under-read",
                                r.label(),
                                r.mac_per_sample(),
                                r.worst_underread_db()
                            ),
                        );
                    }
                });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(format!("at {}", self.settings.tp_ratio.chain()));
            });
        });
        ui.weak(format!(
            "Every render reports the true peak of what it wrote, in dBTP. The ratio is the \
             detector's over-sampling factor: a lower one cannot see a crest that falls \
             between its grid points, so it under-reads — x2 by up to {} dB. Metering changes \
             no output bytes.",
            Ratio::X2.worst_underread_db()
        ));
        ui.add_space(6.0);
        // Above the limiter, and above its early return, because DC does not care whether
        // limiting is on: an offset ships in the file either way. It earns its keep under a
        // limiter too, since a limiter reads an offset as level and will hold gain down for
        // a signal that never clips. Every render reports the file's own `dc_offset`, so the
        // box gets ticked from a measurement rather than from a hunch.
        ui.checkbox(&mut self.settings.dc_block, "block DC (5 Hz high-pass)")
            .on_hover_text(
                "Removes DC before the gain trim and the limiter: exactly zero at DC, \
                 exactly unity at Nyquist, so the top octave is left alone. Off is the \
                 bit-exact path -- the downmixer contributes a settled +4e-10, which is \
                 not the reason to press this. What is: a 7.1 matrix carries an input \
                 offset at +5.18 on the left, and real recordings arrive with 1e-4 of it. \
                 The meter history shows each file's measured dc_offset. Turning this on \
                 changes every byte, so every hash changes with it.",
            );
        ui.add_space(6.0);
        ui.checkbox(&mut self.settings.limiter, "limit the true peak");
        if !self.settings.limiter {
            ui.weak("Off: files are measured, never changed.");
            return;
        }
        let st = &mut self.settings;
        let mut cfg = LCfg {
            enable: true,
            ceiling_db: st.setpoint_db.unwrap_or(st.ceiling_db),
            ratio: st.limiter_ratio(),
            lookahead_ms: st.lookahead_ms,
            attack_ms: st.attack_ms,
            release_ms: st.release_ms,
            link_transient: st.link_transient_pct / 100.0,
            link_release: st.link_release_pct / 100.0,
        };
        // One field, two effects: the trim and the ceiling. It is the first control here
        // because it is the one that decides what the others are for — with a setpoint in
        // play the limiter is only ever catching peaks, since the level came down as a
        // constant gain on the way in.
        ui.horizontal(|ui| {
            ui.label("setpoint");
            let mut sp = st.setpoint_db.unwrap_or(st.ceiling_db);
            if ui
                .add(
                    egui::DragValue::new(&mut sp)
                        .range(-24.0..=0.0)
                        .speed(0.05)
                        .suffix(" dBTP"),
                )
                .changed()
            {
                st.setpoint_db = Some(sp.clamp(-24.0, 0.0));
            }
            if st.setpoint_db.is_some() {
                ui.label(format!(
                    "(- trim, - cap{}; clear it to set the two apart)",
                    if st.gain_db != 0.0 {
                        format!(", plus your own {:+.2} dB gain", st.gain_db)
                    } else {
                        String::new()
                    }
                ));
            }
            if ui.button("separate").clicked() {
                st.setpoint_db = None;
            }
        });
        ui.horizontal(|ui| {
            ui.label("ceiling");
            let mut c = st.ceiling_db;
            if ui
                .add(
                    egui::DragValue::new(&mut c)
                        .range(-24.0..=0.0)
                        .speed(0.05)
                        .suffix(" dBTP"),
                )
                .changed()
            {
                st.ceiling_db = c.clamp(-24.0, 0.0);
            }
            ui.label("lookahead");
            let mut la = st.lookahead_ms;
            if ui
                .add(
                    egui::DragValue::new(&mut la)
                        .range(
                            g6me_core::limiter::MIN_LOOKAHEAD_MS
                                ..=g6me_core::limiter::MAX_LOOKAHEAD_MS,
                        )
                        .speed(0.02)
                        .suffix(" ms"),
                )
                .changed()
            {
                st.lookahead_ms = la;
            }
            ui.label("limit at");
            // The label used to read "same as meter", which is how a fallback to the meter's x4
            // passed for a design decision for as long as it did. The two defaults are different
            // numbers on purpose, so the label says which one it is, derived from the constant.
            let default_label = format!("default ({})", g6me_core::limiter::DEFAULT_RATIO.label());
            egui::ComboBox::from_id_salt("limit-ratio")
                .selected_text(match st.limit_ratio {
                    None => default_label.clone(),
                    Some(r) => r.label().to_string(),
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut st.limit_ratio, None, default_label.as_str());
                    for r in Ratio::ALL {
                        ui.selectable_value(&mut st.limit_ratio, Some(r), r.label());
                    }
                });
        });
        // The slow stage's shape and how much of it the channels share. Both linking
        // controls live here rather than behind a menu, because the damage they prevent
        // (a stereo image that pumps) is only noticed after a render, and the fix is two
        // numbers.
        ui.horizontal(|ui| {
            ui.label("attack");
            let mut a = st.attack_ms;
            if ui
                .add(
                    egui::DragValue::new(&mut a)
                        .range(
                            g6me_core::limiter::MIN_ATTACK_MS..=g6me_core::limiter::MAX_ATTACK_MS,
                        )
                        .speed(0.5)
                        .suffix(" ms"),
                )
                .changed()
            {
                st.attack_ms = a;
            }
            ui.label("release");
            let mut r = st.release_ms;
            if ui
                .add(
                    egui::DragValue::new(&mut r)
                        .range(
                            g6me_core::limiter::MIN_RELEASE_MS..=g6me_core::limiter::MAX_RELEASE_MS,
                        )
                        .speed(2.0)
                        .suffix(" ms"),
                )
                .changed()
            {
                st.release_ms = r;
            }
            ui.label("link");
            let mut lt = st.link_transient_pct;
            if ui
                .add(
                    egui::DragValue::new(&mut lt)
                        .range(0.0..=100.0)
                        .speed(1.0)
                        .suffix("% transient"),
                )
                .changed()
            {
                st.link_transient_pct = lt;
            }
            let mut lr = st.link_release_pct;
            if ui
                .add(
                    egui::DragValue::new(&mut lr)
                        .range(0.0..=100.0)
                        .speed(1.0)
                        .suffix("% release"),
                )
                .changed()
            {
                st.link_release_pct = lr;
            }
        });
        cfg.ceiling_db = st.setpoint_db.unwrap_or(st.ceiling_db);
        cfg.ratio = st.limiter_ratio();
        cfg.lookahead_ms = st.lookahead_ms;
        cfg.attack_ms = st.attack_ms;
        cfg.release_ms = st.release_ms;
        cfg.link_transient = st.link_transient_pct / 100.0;
        cfg.link_release = st.link_release_pct / 100.0;
        let lat = cfg.latency_samples(rate);
        let asked = cfg.lookahead_ms;
        let used = cfg.effective_lookahead_ms(rate);
        ui.weak(format!(
            "two stages: the lookahead one catches peaks, then a level one with a {:.0} ms \
             attack and {:.0} ms release rides under it. Peaks are ducked in both channels \
             {:.0}% together, level {:.0}%.",
            cfg.attack_ms, cfg.release_ms, st.link_transient_pct, st.link_release_pct
        ));
        // The ceiling named in this sentence has to be the one the box is set to. It used to
        // print a literal "-1.00 dBTP" beside a number calculated from the actual ceiling, so
        // at the default -3.00 it read "a -1.00 dBTP ceiling lands near -3.62 dBTP" — one
        // claim, two ceilings, and a shortfall that belonged to the meter's 4x rather than to
        // the 8x on offer. Everything here is now read off the config the render will use.
        ui.weak(format!(
            "aims {:.2} dB below the ceiling ({:.2} dB the meter cannot see + {:.2} dB of \
             gain slew), so the {:.2} dBTP ceiling set above lands near {:.2} dBTP.",
            cfg.guard_db(),
            cfg.ratio.worst_underread_db(),
            cfg.guard_db() - cfg.ratio.worst_underread_db(),
            cfg.ceiling_db,
            cfg.ceiling_db - cfg.guard_db()
        ));
        ui.weak(format!(
            "latency {} samples ({:.3} ms) at {} Hz, added to the end of the file. {}",
            lat,
            lat as f64 / rate as f64 * 1000.0,
            rate,
            if used > asked + 1e-6 {
                format!(
                    "{} ms lookahead asked for is shallower than the {} filter's own memory \
                     ({} samples), so {:.2} ms is used.",
                    asked,
                    cfg.ratio.label(),
                    cfg.ratio.group_delay().ceil() as usize,
                    used
                )
            } else {
                format!("{:.2} ms of lookahead is honoured as asked.", used)
            }
        ));
        if st.limit_ratio.is_some() && st.limit_ratio != Some(st.tp_ratio) {
            ui.weak(format!(
                "Meter and limiter are running different ratios, so each measures once \
                 more than they would agree: the report's dBTP is the {} reading, the \
                 ceiling is held by the {} one.",
                st.tp_ratio.label(),
                cfg.ratio.label()
            ));
        }
        if let Err(e) = cfg.validate() {
            ui.colored_label(egui::Color32::from_rgb(226, 110, 90), e);
        }
    }

    fn tab_resample(&mut self, ui: &mut egui::Ui, caps: Option<&Caps>) {
        let soxr = caps.map(|c| c.soxr).unwrap_or(true);
        if !soxr {
            ui.colored_label(
                egui::Color32::from_rgb(226, 140, 90),
                "This ffmpeg has no libsoxr, so resampling is unavailable. Everything else \
                 still works, and the default path is resampleless anyway.",
            );
            ui.add_enabled_ui(false, |ui| {
                self.resample_controls(ui);
            });
        } else {
            self.resample_controls(ui);
        }
    }

    fn resample_controls(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(
            &mut self.settings.resample,
            "resample the input before the downmix (libsoxr)",
        );
        ui.weak(
            "Rate conversion happens before the downmix, never after: the filter coefficients \
             model the hardware at its own rate, so resampling the stereo result would be a \
             different curve.",
        );
        ui.add_enabled_ui(self.settings.resample, |ui| {
            ui.horizontal(|ui| {
                ui.label("target rate");
                let mut r = self.settings.rate;
                if ui
                    .add(
                        egui::DragValue::new(&mut r)
                            .speed(100)
                            .range(8_000..=384_000),
                    )
                    .changed()
                {
                    self.settings.rate = r;
                }
                ui.label("Hz");
                for preset in [44_100u32, 48_000, 88_200, 96_000, 192_000] {
                    if ui
                        .selectable_label(self.settings.rate == preset, format!("{preset}"))
                        .clicked()
                    {
                        self.settings.rate = preset;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.settings.use_resample_depth, "quantize to");
                egui::ComboBox::from_id_salt("rsdepth")
                    .selected_text(self.settings.resample_depth.label())
                    .show_ui(ui, |ui| {
                        for d in BitDepth::ALL {
                            ui.selectable_value(&mut self.settings.resample_depth, d, d.label());
                        }
                    });
                ui.weak(g6me_core::pcm::BitDepth::quantize_note());
            });
            ui.horizontal(|ui| {
                ui.label("soxr precision");
                let mut p = self.settings.precision;
                if ui
                    .add(
                        egui::Slider::new(&mut p, 15.0..=33.0)
                            .text("bits")
                            .step_by(1.0),
                    )
                    .changed()
                {
                    self.settings.precision = p;
                }
                ui.checkbox(&mut self.settings.cheby, "Chebyshev passband");
            });
            ui.horizontal(|ui| {
                ui.label("dither");
                egui::ComboBox::from_id_salt("dither")
                    .selected_text(self.settings.dither.label())
                    .show_ui(ui, |ui| {
                        for d in Dither::ALL {
                            ui.selectable_value(&mut self.settings.dither, d, d.label());
                        }
                    });
                if self.settings.dither != Dither::None {
                    ui.weak("deterministic: the same input gives the same output every run");
                }
            });
        });
    }

    fn tab_ir(&mut self, ui: &mut egui::Ui, caps: Option<&Caps>) {
        let afir = caps.map(|c| c.afir).unwrap_or(true);
        if !afir {
            ui.colored_label(
                egui::Color32::from_rgb(226, 140, 90),
                "This ffmpeg has no afir filter, so IR calibration is unavailable.",
            );
        }
        ui.checkbox(
            &mut self.settings.ir,
            "convolve every input channel with a headphone correction IR",
        );
        ui.weak(
            "Same shape mpv uses: mono IR, applied to all input channels before the downmix, \
             with no normalisation by default so a measured curve keeps its measured level.",
        );
        ui.add_enabled_ui(self.settings.ir, |ui| {
            ui.horizontal(|ui| {
                ui.label("IR file");
                let shown = if self.settings.ir_file.as_os_str().is_empty() {
                    "(none chosen)".to_string()
                } else {
                    widgets::short_path(&self.settings.ir_file, 44)
                };
                ui.monospace(shown);
                if ui.button("…").clicked() {
                    self.pick = Some(Pick::Ir);
                    self.browser.mode = Mode::AnyFile;
                    self.browser.set_single(Some(self.settings.ir_file.clone()));
                }
            });
            ui.horizontal(|ui| {
                ui.label("irnorm");
                let mut n = self.settings.ir_norm;
                if ui
                    .add(egui::DragValue::new(&mut n).speed(0.05).range(-1.0..=2.0))
                    .changed()
                {
                    self.settings.ir_norm = n;
                }
                if ui
                    .selectable_label(self.settings.ir_norm == -1.0, "none (-1)")
                    .clicked()
                {
                    self.settings.ir_norm = -1.0;
                }
                ui.weak("(-1 = raw level, what a measured filter needs)");
            });
            ui.horizontal(|ui| {
                ui.label("wet");
                let mut w = self.settings.ir_wet;
                if ui
                    .add(egui::Slider::new(&mut w, 0.0..=2.0).step_by(0.05))
                    .changed()
                {
                    self.settings.ir_wet = w;
                }
                ui.checkbox(&mut self.settings.use_ir_dry, "dry");
                if self.settings.use_ir_dry {
                    let mut d = self.settings.ir_dry;
                    ui.add(egui::Slider::new(&mut d, 0.0..=2.0).step_by(0.05));
                    self.settings.ir_dry = d;
                }
            });
            ui.horizontal(|ui| {
                ui.label("graph type");
                let mut cur = self.settings.ir_gtype;
                egui::ComboBox::from_id_salt("gtype")
                    .selected_text(match cur {
                        None => "default".to_string(),
                        Some(g) => g.ffmpeg_name().to_string(),
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut cur, None, "default");
                        for g in [
                            Gtype::Peak,
                            Gtype::Dc,
                            Gtype::Gn,
                            Gtype::Ac,
                            Gtype::Rms,
                            Gtype::None,
                        ] {
                            ui.selectable_value(&mut cur, Some(g), g.ffmpeg_name());
                        }
                    });
                self.settings.ir_gtype = cur;
                ui.label("partition");
                let mut mn = self.settings.ir_minp;
                let mut mx = self.settings.ir_maxp;
                if ui
                    .add(egui::DragValue::new(&mut mn).range(64..=65536))
                    .changed()
                {
                    self.settings.ir_minp = mn;
                }
                ui.label("…");
                if ui
                    .add(egui::DragValue::new(&mut mx).range(64..=65536))
                    .changed()
                {
                    self.settings.ir_maxp = mx;
                }
            });
            ui.horizontal(|ui| {
                ui.label("convolution length");
                let mut l = self.settings.ir_length;
                // A fraction, because that is the unit FFmpeg's afir defines. The old
                // 0..8 "seconds" control happily produced values afir refuses outright
                // ("out of range [0 - 1]", exit 222), which surfaced as a failed render
                // with a filter-graph message that never mentioned this control.
                ui.add(
                    egui::DragValue::new(&mut l)
                        .speed(0.02)
                        .range(0.0..=1.0)
                        .fixed_decimals(2),
                );
                self.settings.ir_length = l;
                ui.weak("(fraction of the IR; 0 or 1 = whole)");
            });
            ui.weak(
                "Convolution runs in double precision, fixed — the mpv equivalent of \
                 precision=double.",
            );
        });
    }

    fn tab_advanced(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("channel order");
            let before = self.settings.order_text.clone();
            ui.add_sized(
                [ui.available_width() - 90.0, 0.0],
                egui::TextEdit::singleline(&mut self.settings.order_text)
                    .font(egui::FontId::monospace(11.0))
                    .hint_text("leave empty to trust the decoder"),
            );
            if before != self.settings.order_text {
                if let Err(e) = self.settings.apply_order_text() {
                    self.status = e;
                }
            }
        });
        ui.weak(format!(
            "known names: {}",
            mask::format_order(&mask::Slot::ALL)
        ));
        if let Some(o) = &self.settings.order {
            ui.label(format!(
                "override in effect: {} → mask {}",
                mask::format_order(o),
                mask::format_mask_hex(mask::mask_of(o))
            ));
        }
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("block");
            let mut b = self.settings.block;
            if ui
                .add(egui::DragValue::new(&mut b).range(64..=262_144))
                .changed()
            {
                self.settings.block = b;
            }
            ui.label("frames · queue");
            let mut q = self.settings.queue_blocks;
            if ui
                .add(egui::DragValue::new(&mut q).range(1..=256))
                .changed()
            {
                self.settings.queue_blocks = q;
            }
            ui.label("blocks");
            if ui
                .selectable_label(
                    self.settings.block == 4096 && self.settings.queue_blocks == 8,
                    "default",
                )
                .clicked()
            {
                self.settings.block = 4096;
                self.settings.queue_blocks = 8;
            }
        });
        ui.weak(format!(
            "peak memory ≈ {} per file (decode and encode overlap the downmixer; the \
             downmixer itself stays single-threaded)",
            util::human_bytes(self.settings.peak_bytes_per_file())
        ));
        ui.horizontal(|ui| {
            ui.label("audio stream");
            let mut s = self.settings.stream;
            ui.add(egui::DragValue::new(&mut s).range(0..=31));
            self.settings.stream = s;
            ui.weak("(0 = first audio stream)");
        });
        ui.separator();
        ui.weak(
            "Decode, downmix and encode run as three concurrent stages per file. Filter \
             threads inside ffmpeg are pinned to one when more than one file runs at once, \
             because N files × M ffmpeg threads is how a batch ends up slower than one thread.",
        );
    }

    fn log_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("Log");
            if ui.small_button("clear").clicked() {
                self.log.clear();
            }
            ui.checkbox(&mut self.log_scroll, "follow");
        });
        egui::ScrollArea::vertical()
            .id_salt("log")
            .max_height(140.0)
            .stick_to_bottom(self.log_scroll)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                for (lvl, msg) in &self.log {
                    let col = match lvl {
                        Level::Info => egui::Color32::from_gray(180),
                        Level::Warn => egui::Color32::from_rgb(226, 180, 90),
                        Level::Error => egui::Color32::from_rgb(226, 120, 100),
                    };
                    ui.label(egui::RichText::new(msg).color(col).monospace().size(11.0));
                }
                if self.log.is_empty() {
                    ui.weak("(nothing logged yet)");
                }
            });
    }

    fn windows(&mut self, ctx: &egui::Context) {
        // The browser is a window rather than a native dialog on purpose: a native
        // dialog on Linux means GTK or an XDG portal, and this has to run on a plain
        // Wayland or dwm session.
        let pick = self.pick.take();
        if let Some(p) = pick {
            let title = match p {
                Pick::Queue => "Add to queue",
                Pick::OutDir => "Choose output folder",
                Pick::Ir => "Choose impulse response",
            };
            let mut open = true;
            let mut done: Option<Vec<PathBuf>> = None;
            egui::Window::new(title)
                .open(&mut open)
                .default_pos([220.0, 160.0])
                .default_size([520.0, 420.0])
                .show(ctx, |ui| {
                    // Several paths at once is the point: ctrl- and shift-clicking a
                    // list of tracks and adding them all is how a queue actually gets
                    // built.
                    self.browser.ui(ui, |paths| done = Some(paths));
                });
            self.pick = if open { Some(p) } else { None };
            if let Some(paths) = done {
                match p {
                    Pick::Queue => {
                        let mut added = 0usize;
                        for path in paths {
                            if !self.settings.inputs.contains(&path) {
                                self.settings.inputs.push(path);
                                added += 1;
                            }
                        }
                        if added > 1 {
                            // Worth saying out loud: a ctrl-drag that lands on four
                            // tracks should be visibly four, not one row appearing.
                            self.push_log(Level::Info, format!("added {added} files to the queue"));
                        }
                    }
                    // These take exactly one path; the first of a selection is what was
                    // clicked, so extras are ignored rather than guessed at.
                    Pick::OutDir => {
                        if let Some(path) = paths.into_iter().next() {
                            self.settings.use_out_dir = true;
                            self.settings.out_dir = path;
                        }
                    }
                    Pick::Ir => {
                        if let Some(path) = paths.into_iter().next() {
                            self.settings.ir_file = path;
                        }
                    }
                }
                if p != Pick::Ir {
                    self.rebuild_queue();
                }
            }
        }

        if self.show_caps {
            let caps = self.caps();
            let mut open = self.show_caps;
            egui::Window::new("Capabilities")
                .open(&mut open)
                .show(ctx, |ui| match &caps {
                    None => {
                        ui.weak("still probing…");
                    }
                    Some(c) => {
                        ui.label(format!("ffmpeg {}", c.ffmpeg_version));
                        for (k, v) in [
                            ("libsoxr resampler", c.soxr),
                            ("afir convolution", c.afir),
                            ("FLAC encode", c.flac),
                            ("lavfi synthetic sources", c.lavfi),
                            ("24-bit FLAC round-trip", c.flac24),
                        ] {
                            let (mark, col) = if v {
                                ("available", egui::Color32::from_rgb(120, 190, 120))
                            } else {
                                ("missing", egui::Color32::from_rgb(226, 120, 100))
                            };
                            ui.horizontal(|ui| {
                                ui.label(k);
                                ui.colored_label(col, mark);
                            });
                        }

                        for n in &c.notes {
                            ui.weak(n.clone());
                        }
                    }
                });
            self.show_caps = open;
        }

        let dr = self.dry_run.lock().ok().and_then(|d| d.clone());
        if let Some(dr) = dr {
            let mut open = true;
            egui::Window::new(format!("Dry run — {}", widgets::short_path(&dr.src, 40)))
                .open(&mut open)
                .default_pos([260.0, 200.0])
                .default_size([680.0, 420.0])
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            if let Some(e) = &dr.error {
                                ui.colored_label(egui::Color32::from_rgb(226, 120, 100), e.clone());
                            }
                            for l in &dr.lines {
                                ui.label(egui::RichText::new(l).monospace().size(11.0));
                            }
                        });
                });
            if !open {
                if let Ok(mut d) = self.dry_run.lock() {
                    *d = None;
                }
            }
        }
    }

    // ---- actions -----------------------------------------------------------

    fn rebuild_queue(&mut self) {
        let cfg = self.settings.batch_cfg();
        let ext = cfg.render.out.format.extension().to_string();
        let items = batch::discover(&cfg.inputs, cfg.out_dir.as_deref(), &ext, cfg.recursive);
        self.queue = items
            .iter()
            .map(|i| Row {
                src: i.src.clone(),
                dst: i.dst.clone(),
                phase: i.phase,
                note: String::new(),
                collision: i.note.clone().unwrap_or_default(),
            })
            .collect();
        let skipped = self
            .queue
            .iter()
            .filter(|r| r.phase == Phase::Skipped)
            .count();
        self.status = format!(
            "{} file(s) queued{}",
            self.queue.len(),
            if skipped > 0 {
                format!(", {skipped} already have output")
            } else {
                String::new()
            }
        );
    }

    fn start_run(&mut self) {
        let caps = match self.caps() {
            Some(c) => c,
            None => {
                self.status = "still probing ffmpeg — one moment".to_string();
                return;
            }
        };
        let problems = self.settings.problems(&caps);
        if !problems.is_empty() {
            for p in &problems {
                self.push_log(Level::Error, p.clone());
            }
            self.summary = Some(Summary {
                error: Some(problems.join("; ")),
                ..Default::default()
            });
            self.status = "not started — fix the problems above".to_string();
            return;
        }
        let mut cfg = self.settings.batch_cfg();
        // Meters for the render itself: a card per file, made as a worker picks the
        // file up, fed every block that goes to the encoder.
        cfg.taps = self.meter_factory();
        self.cards.clear();
        self.summary = None;
        match self.runner.start(&cfg, self.tools.clone(), caps) {
            Ok(items) => {
                self.queue = items
                    .iter()
                    .map(|i| Row {
                        src: i.src.clone(),
                        dst: i.dst.clone(),
                        phase: i.phase,
                        note: String::new(),
                        collision: i.note.clone().unwrap_or_default(),
                    })
                    .collect();
                self.status = format!("rendering {} file(s)", items.len());
                self.push_log(
                    Level::Info,
                    format!(
                        "start · {} · {} → {}",
                        describe(&self.settings),
                        items.len(),
                        "files"
                    ),
                );
            }
            Err(e) => {
                self.push_log(Level::Error, e.clone());
                self.summary = Some(Summary {
                    error: Some(e),
                    ..Default::default()
                });
                self.status = "not started".to_string();
            }
        }
    }

    fn run_dry_run(&mut self) {
        let Some(first) = self.queue.first() else {
            self.status = "queue something first".to_string();
            return;
        };
        let Some(caps) = self.caps() else {
            self.status = "still probing ffmpeg".to_string();
            return;
        };
        let cfg = self.settings.render_cfg();
        let src = first.src.clone();
        let dst = first.dst.clone();
        let tools = self.tools.clone();
        let out = match pipeline::plan_file(&src, &dst, &cfg, &tools, &caps) {
            Ok(p) => {
                let mut lines = vec![
                    format!(
                        "source   {}{}  [{} ch, {} Hz, {}, {}]",
                        p.info.codec_name,
                        p.info
                            .profile
                            .as_deref()
                            .map(|x| format!(" ({x})"))
                            .unwrap_or_default(),
                        p.info.channels,
                        p.info.sample_rate,
                        p.info.layout.clone().unwrap_or_else(|| "?".into()),
                        p.info
                            .bits
                            .map(|b| format!("{b}-bit"))
                            .unwrap_or_else(|| "? bit".into()),
                    ),
                    format!(
                        "order    {}  mask {}  stage {}",
                        mask::format_order(&p.order),
                        mask::format_mask_hex(p.mask),
                        p.stage.label()
                    ),
                    format!("rate     {} Hz working", p.working_rate),
                    String::new(),
                    "chain".to_string(),
                ];
                for (i, step) in p.chain.iter().enumerate() {
                    lines.push(format!("  {}. {step}", i + 1));
                }
                lines.push(String::new());
                lines.push("decode".to_string());
                lines.push(format!("  {}", p.input_args.join(" ")));
                lines.push(String::new());
                lines.push("encode".to_string());
                lines.push(format!("  {}", p.encode_cmd.join(" ")));
                for w in &p.warnings {
                    lines.push(format!("warning: {w}"));
                }
                DryRun {
                    src,
                    lines,
                    error: None,
                }
            }
            Err(e) => DryRun {
                src,
                lines: vec![
                    "This file cannot be rendered with the current settings.".to_string(),
                    String::new(),
                    format!("  {e}"),
                ],
                error: Some(e.to_string()),
            },
        };
        if let Ok(mut d) = self.dry_run.lock() {
            *d = Some(out);
        }
        self.status = "dry run".to_string();
    }

    /// One meter per file, handed to the batch as it starts each one.
    ///
    /// This is the whole reason the app can show levels without making a sound: the
    /// render thread passes each block to the meter on its way to the encoder, so the
    /// picture is the same arithmetic that is being written to disk — not a parallel
    /// "just measure it" path that can drift away from it.
    fn meter_factory(&self) -> g6me_core::batch::MeterFactory {
        let cards = self.cards.clone();
        g6me_core::batch::MeterFactory(Some(std::sync::Arc::new(move |index: usize| {
            // No redraw hook here on purpose: the window already runs a 20 fps repaint
            // clock while a batch is live (see `update`), and asking for a repaint on
            // every block — a file goes through ~400 of them a second at speed — would
            // only spin the UI faster than the eye can read.
            let tap: std::sync::Arc<dyn g6me_core::pipeline::SampleTap> =
                std::sync::Arc::new(MeterTap::new(cards.open(index)));
            Some(tap)
        })))
    }

    fn drain_events(&mut self) {
        for e in self.runner.drain() {
            match e {
                Event::ItemState {
                    index,
                    phase,
                    note,
                    report,
                } => {
                    if let Some(r) = self.queue.get_mut(index) {
                        r.phase = phase;
                        r.note = note.clone();
                    }
                    if let Some(r) = report {
                        self.history.push(MeterRow::from_report(&r));
                        // A library is thousands of files; the list is a record of the
                        // session, not a database. Oldest rows go first.
                        const HISTORY_CAP: usize = 4000;
                        if self.history.len() > HISTORY_CAP {
                            self.history.remove(0);
                        }
                    }
                    if phase == Phase::Failed {
                        self.push_log(Level::Error, note);
                    }
                    // A finished file's card stays on screen, dimmed, with its last
                    // peaks on it: the clipping you missed while it ran is still worth
                    // seeing after it stopped. It goes when the next run starts.
                    if matches!(phase, Phase::Done | Phase::Failed | Phase::Skipped) {
                        self.cards.finish(index);
                    }
                }
                Event::Finished(s) => {
                    self.push_log(
                        if s.failed > 0 {
                            Level::Warn
                        } else {
                            Level::Info
                        },
                        s.one_line(),
                    );
                    self.status = "idle".to_string();
                    self.summary = Some(s);
                }
                Event::Caps(caps, note) => {
                    // Handling the probe here rather than in a callback keeps every UI
                    // state change on the UI thread.
                    self.status = note.clone();
                    if let Ok(mut c) = self.caps.lock() {
                        *c = Some(caps);
                    }
                    if let Ok(mut n) = self.caps_note.lock() {
                        *n = Some(note);
                    }
                    // Now that the caps are known, the queue can be scanned: which
                    // files already have an output depends on the chosen extension.
                    self.rebuild_queue();
                }
            }
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    format!("{}…", &s[..max])
}

/// Note shown next to a bit-depth choice, or `None` when there is nothing to warn
/// about.
fn depth_note(d: BitDepth, f: Format, caps: Option<&Caps>) -> Option<&'static str> {
    if f != Format::Flac {
        return None;
    }
    // FLAC is an integer container and ffmpeg's encoder stops at 24 bits, so these
    // combinations are refused by the validator rather than quietly downgraded.
    if d.is_float() {
        return Some("not available in FLAC — use WAV");
    }
    if d == BitDepth::S32 {
        return Some("not available in FLAC — use WAV");
    }
    if d == BitDepth::S24 && caps.map(|c| !c.flac24).unwrap_or(false) {
        return Some("unverified on this ffmpeg");
    }
    None
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_events();
        // The palette is installed on a change, not every frame: `apply` rewrites
        // egui's stored styles, and doing that sixty times a second would throw away
        // the caching it keeps them for.
        if self.installed != Some(self.palette) {
            self.palette.apply(ctx);
            self.installed = Some(self.palette);
        }
        self.top_panel(ctx);
        self.bottom_panel(ctx);
        self.queue_panel(ctx);
        self.central_panel(ctx);
        self.windows(ctx);
        // The meters decay on the render thread, so a live meter needs a redraw
        // clock. 20 fps is enough to read and cheap enough to leave running; when
        // nothing is active the window goes quiet.
        if self.runner.is_running() {
            self.last_repaint = std::time::Instant::now();
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        } else if self.last_repaint.elapsed() < std::time::Duration::from_millis(400) {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }
}

fn main() -> eframe::Result {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 820.0])
            .with_min_inner_size([900.0, 620.0])
            .with_title("G6 Media Encoder"),
        // Glow, not Wgpu: this has to come up on an old Windows GPU driver and on a
        // Wayland compositor with no hardware acceleration, and it does so on GL.
        ..Default::default()
    };
    eframe::run_native(
        "G6 Media Encoder",
        opts,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real report from a real render, decoded back.
    ///
    /// A fixture rather than a literal, for two reasons: `Report` has thirty fields and a
    /// hand-written one would be all noise, and this way a field that stops reaching the
    /// panel — a rename that drops DC, or the decoded-back pass going unreported — fails a
    /// test instead of quietly vanishing from the copy text.
    fn rendered() -> Report {
        let raw = include_str!("../testdata/clamped.json");
        match serde_json::from_str::<Report>(raw) {
            Ok(r) => r,
            Err(e) => panic!("the fixture no longer matches Report: {e}"),
        }
    }

    /// The meters decay to rest inside a third of a second. What a finished file is
    /// remembered by has to be the numbers, not the memory of a bar.
    #[test]
    fn a_finished_file_keeps_its_numbers_after_the_render() {
        let r = rendered();
        assert!(r.delivered.is_some(), "fixture must carry the decode-back");
        let row = MeterRow::from_report(&r);
        for want in ["tp ", "dBTP", "peak ", "rms "] {
            assert!(
                row.levels.contains(want),
                "the row lost {want}: {}",
                row.levels
            );
        }
        assert!(
            row.levels.contains(".wav"),
            "a row with no file: {}",
            row.levels
        );
        // The full text is what "copy all" hands to a log or a delivery note, so it carries
        // the parts that make a number checkable: the detector ratio, the DC, and the file's
        // own measurement next to the stream's.
        for want in ["@x4", "dc:", "file on disk", "WARNING"] {
            assert!(row.text.contains(want), "the copy text lost {want}");
        }
        // …and the compact row stays quiet about an offset this file does not have: 1.9e-4
        // is measured, reported in the text above, and not a finding.
        assert!(
            !row.levels.contains("dc "),
            "an immaterial offset took the compact row: {}",
            row.levels
        );
    }

    /// A row that reads calm about a file whose own decoded measurement disagrees with its
    /// report is a row that will be trusted anyway.
    #[test]
    fn a_file_disagreeing_with_its_own_report_shows_as_hot() {
        let r = rendered();
        assert!(r.delivered_note.is_some(), "fixture must disagree");
        assert!(
            MeterRow::from_report(&r).hot,
            "a clamped file is listed as calm"
        );

        let mut calm = r.clone();
        calm.delivered_note = None;
        calm.true_peak_dbtp = -20.0;
        calm.limiter = None;
        assert!(
            !MeterRow::from_report(&calm).hot,
            "a cold file is flagged hot, which trains the colour to be ignored"
        );
    }

    /// The dB figure with no logarithm still has to print as itself.
    #[test]
    fn db_prints_the_value_that_has_no_logarithm() {
        assert_eq!(db(f64::NEG_INFINITY), "\u{2212}inf");
        assert_eq!(db(f64::NAN), "n/a");
        assert_eq!(db(-3.0), "-3.00");
    }
}
