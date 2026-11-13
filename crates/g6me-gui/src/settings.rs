//! Every knob the CLI exposes, as a single plain struct.
//!
//! The rule this file exists to enforce: **the GUI and the CLI must not be able to
//! disagree.** Both build a [`RenderCfg`] from the same core types, and the
//! translations here are written to match `Cli::render_cfg` in the CLI crate
//! field for field — including the parts that are easy to get subtly wrong, like
//! keeping `irnorm=-1` (so `afir` does not normalise the IR and change the
//! calibration) and leaving `out.resample` at `None` (rate changes belong before
//! the downmix, because the filter coefficients model the hardware at its own
//! rate).
//!
//! Options that don't exist are kept as `checkbox + value` pairs rather than
//! `Option`, because egui binds to `&mut T` and an unchecking action needs to
//! leave the last value visible for re-checking.

use g6me_core::batch::BatchCfg;
use g6me_core::mask::{self, Slot};
use g6me_core::pcm::{BitDepth, Dither};
use g6me_core::probe::Caps;
use g6me_core::stages::{Gtype, IrCfg, OutFormat, RenderCfg};
use g6me_core::tp::Ratio;
use g6me_core::util;
use std::path::PathBuf;

/// Output container, mirroring `--format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Flac,
    Wav,
}

impl Format {
    pub const ALL: [Format; 2] = [Format::Flac, Format::Wav];

    pub fn label(self) -> &'static str {
        match self {
            Format::Flac => "FLAC",
            Format::Wav => "WAV",
        }
    }
}

/// Everything the user can set, in the order the panels show it.
#[derive(Clone, Debug)]
pub struct Settings {
    // ---- queue -------------------------------------------------------------
    /// Files and folders, in the order added.
    pub inputs: Vec<PathBuf>,
    pub use_out_dir: bool,
    pub out_dir: PathBuf,
    pub recursive: bool,
    /// Files rendered at once. DSP stays single-threaded inside each file.
    pub threads: usize,

    // ---- output ------------------------------------------------------------
    pub format: Format,
    /// FLAC compression level, 0..=12 on ffmpeg's encoder scale. Lossless at every
    /// level; the reference `flac` CLI's `-0..-8` presets are a different scale.
    pub flac_level: u32,
    pub depth: BitDepth,
    pub copy_metadata: bool,
    pub overwrite: bool,
    pub sha256: bool,

    // ---- true peak ---------------------------------------------------------
    /// Over-sampling ratio every file's true peak is measured at. Affects the reported
    /// number only, never a byte of output.
    pub tp_ratio: Ratio,
    /// Hold the delivered file under a true-peak ceiling. The last stage before the
    /// encoder, so this is the real ceiling on the file. On by default: a file that leaves
    /// here over 0 dBTP is a file that clips somewhere downstream, and the limiter is what
    /// makes the reported ceiling true.
    pub limiter: bool,
    /// One number doing two jobs: the file is trimmed by this much and capped here. The
    /// cheap way to hold a level, because a constant gain does the work on every sample and
    /// the limiter only ever sees the peaks that survive the trim. `None` leaves the gain
    /// and the ceiling to their own fields.
    pub setpoint_db: Option<f64>,
    /// Gain staging on its own, dB, applied before the limiter.
    pub gain_db: f64,
    /// How fast the slow stage's reduction arrives, ms.
    pub attack_ms: f64,
    /// How fast the slow stage lets go, ms.
    pub release_ms: f64,
    /// Channel linking of the transient stage, percent.
    pub link_transient_pct: f64,
    /// Channel linking of the slow stage, percent. Lower than the transient one by
    /// default: a slow shared gain move is what pulls a stereo image about.
    pub link_release_pct: f64,
    /// Ceiling in dBTP.
    pub ceiling_db: f64,
    /// Lookahead the limiter may ask for, ms; the effective value is reported per file
    /// because a deep filter cannot honour a shallow request.
    pub lookahead_ms: f64,
    /// Ratio the limiter's own detector runs at. `None` = the meter's ratio, which is
    /// also the cheap case: one measurement serves both jobs.
    pub limit_ratio: Option<Ratio>,
    /// Strip DC before the trim and the limiter, with a 5 Hz first-order blocker.
    ///
    /// Off by default, which is the bit-exact path: the downmixer's own contribution is a
    /// settled +4e-10 and reaching for it would mean editing the port. What this is for is
    /// DC that came in with the source — every render reports `dc_offset`, so the decision
    /// is made from the file's own number rather than from a guess. It belongs beside the
    /// limiter rather than inside it because it earns its keep with the limiter off too:
    /// an offset ships in the file either way.
    pub dc_block: bool,

    // ---- resample ----------------------------------------------------------
    pub resample: bool,
    pub rate: u32,
    pub resample_depth: BitDepth,
    pub use_resample_depth: bool,
    /// soxr quality in bits, 15..=33.
    pub precision: f64,
    pub dither: Dither,
    pub cheby: bool,

    // ---- IR calibration ----------------------------------------------------
    pub ir: bool,
    pub ir_file: PathBuf,
    pub ir_norm: f64,
    pub ir_gtype: Option<Gtype>,
    pub ir_wet: f64,
    pub ir_dry: f64,
    pub use_ir_dry: bool,
    pub ir_minp: u32,
    pub ir_maxp: u32,
    /// `afir`'s `length`: a fraction of the IR, not a duration. `0` means "the whole IR",
    /// which is what the option already defaults to, so it is sent as absent.
    pub ir_length: f64,

    // ---- advanced ----------------------------------------------------------
    /// `None` = trust the decoder's channel order.
    pub order: Option<Vec<Slot>>,
    /// Text box the order is edited in (`FL,FR,FC,LFE,BL,BR,SL,SR`).
    pub order_text: String,
    pub block: u32,
    pub queue_blocks: u32,
    pub stream: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            use_out_dir: false,
            out_dir: PathBuf::from("."),
            // A music library is a tree; the exception is rarer than the rule.
            recursive: true,
            threads: g6me_core::batch::default_workers(),
            format: Format::Flac,
            flac_level: 8,
            depth: BitDepth::S24,
            copy_metadata: true,
            overwrite: false,
            sha256: false,
            // x4 is what BS.1770 measures at, so a reported dBTP is comparable with any
            // other tool's without the user knowing it.
            tp_ratio: Ratio::X4,
            // On: a file that leaves here above the ceiling is a file that clips somewhere
            // downstream, and the ceiling in the report would be a claim rather than a
            // fact. Turning it off is `--no-limit`'s job, or this checkbox's.
            limiter: true,
            setpoint_db: None,
            gain_db: 0.0,
            ceiling_db: g6me_core::limiter::DEFAULT_CEILING_DB,
            lookahead_ms: g6me_core::limiter::DEFAULT_LOOKAHEAD_MS,
            attack_ms: default_attack_ms(),
            release_ms: default_release_ms(),
            link_transient_pct: default_link_transient(),
            link_release_pct: default_link_release(),
            limit_ratio: None,
            dc_block: false,
            resample: false,
            rate: 48000,
            resample_depth: BitDepth::S32,
            use_resample_depth: false,
            precision: 33.0,
            dither: Dither::None,
            cheby: false,
            ir: false,
            ir_file: PathBuf::new(),
            // -1 = no automatic gain: a measured correction filter must keep the
            // level it was measured at.
            ir_norm: -1.0,
            ir_gtype: None,
            ir_wet: 1.0,
            ir_dry: 1.0,
            use_ir_dry: false,
            ir_minp: 4096,
            ir_maxp: 4096,
            ir_length: 0.0,
            order: None,
            order_text: String::new(),
            // 4096 frames is ~85 ms at 48 kHz: large enough that ffmpeg's pipe
            // rarely sees a short read, small enough to stay responsive.
            block: 4096,
            queue_blocks: 8,
            stream: 0,
        }
    }
}

/// Settings keep linking as percent because that is what the panel shows; core takes a
/// fraction. These are the boundary, so a default cannot be written twice in two units.
fn default_attack_ms() -> f64 {
    g6me_core::limiter::DEFAULT_ATTACK_MS
}

fn default_release_ms() -> f64 {
    g6me_core::limiter::DEFAULT_RELEASE_MS
}

fn default_link_transient() -> f64 {
    g6me_core::limiter::DEFAULT_LINK_TRANSIENT * 100.0
}

fn default_link_release() -> f64 {
    g6me_core::limiter::DEFAULT_LINK_RELEASE * 100.0
}

impl Settings {
    /// The ceiling both CLI and GUI share, so neither can offer a thread count
    /// the machine cannot run.
    pub fn max_threads() -> usize {
        util::max_threads()
    }

    pub fn threads(&self) -> usize {
        util::clamp_threads(self.threads)
    }

    pub fn out_format(&self) -> OutFormat {
        match self.format {
            Format::Flac => OutFormat::Flac {
                // ffmpeg's FLAC encoder counts 0..=12; 12 packs smallest.
                level: self.flac_level.min(12),
            },
            Format::Wav => OutFormat::Wav,
        }
    }

    /// Mirror of the CLI's translation. Keep the two in step.
    /// The side-chain ratio, with the limiter's own default behind it.
    ///
    /// One accessor rather than a fallback written at each call site: the panel built its
    /// `Cfg` twice, both times with `unwrap_or_default()`, and `Ratio::default` is the *meter's*
    /// x4. So the limiter ran a step shallower than `Cfg::default`, the CLI help and the
    /// README each said it did, and the prose under the controls printed the 0.62 dB guard
    /// that goes with 4x while the box offered 8x.
    #[must_use]
    pub fn limiter_ratio(&self) -> Ratio {
        self.limit_ratio
            .unwrap_or(g6me_core::limiter::DEFAULT_RATIO)
    }

    pub fn render_cfg(&self) -> RenderCfg {
        let ir = IrCfg {
            enable: self.ir,
            file: (!self.ir_file.as_os_str().is_empty()).then(|| self.ir_file.clone()),
            // `IrCfg::default()` is the only source of truth for these two; the
            // CLI does the same, so an edit here applies to both front ends.
            irnorm: Some(self.ir_norm),
            gtype: self.ir_gtype,
            wet: Some(self.ir_wet),
            dry: self.use_ir_dry.then_some(self.ir_dry),
            length: (self.ir_length > 0.0).then_some(self.ir_length),
            link: None,
            minp: self.ir_minp,
            maxp: self.ir_maxp,
        };
        RenderCfg {
            dc_block: self.dc_block,
            resample: g6me_core::stages::ResampleCfg {
                enable: self.resample,
                rate: self.rate,
                depth: self.use_resample_depth.then_some(self.resample_depth),
                precision: self.precision,
                dither: self.dither,
                cheby: self.cheby,
            },
            ir,
            out: g6me_core::stages::OutCfg {
                format: self.out_format(),
                depth: self.depth,
                // Rate conversion happens before the downmix, never after.
                resample: None,
                precision: self.precision,
                dither: self.dither,
                copy_metadata: self.copy_metadata,
                overwrite: self.overwrite,
                filter_threads: None,
            },
            order: self.order.clone(),
            true_peak: self.tp_ratio,
            output_gain_db: self.setpoint_db.unwrap_or(self.gain_db),
            limiter: g6me_core::limiter::Cfg {
                enable: self.limiter,
                ceiling_db: self.setpoint_db.unwrap_or(self.ceiling_db),
                ratio: self.limiter_ratio(),
                lookahead_ms: self.lookahead_ms,
                attack_ms: self.attack_ms,
                release_ms: self.release_ms,
                link_transient: self.link_transient_pct / 100.0,
                link_release: self.link_release_pct / 100.0,
            },
            block: self.block as usize,
            queue_blocks: self.queue_blocks as usize,
            stream: self.stream as usize,
            sha256: self.sha256,
        }
    }

    /// The batch as the settings describe it. `taps` is left empty: meters belong to
    /// the window, not to the render settings, so `App::start_run` fills this in and
    /// anything else driving the same settings renders blind.
    pub fn batch_cfg(&self) -> BatchCfg {
        BatchCfg {
            inputs: self.inputs.clone(),
            out_dir: self.use_out_dir.then(|| self.out_dir.clone()),
            recursive: self.recursive,
            workers: self.threads(),
            render: self.render_cfg(),
            fail_on_empty: true,
            taps: Default::default(),
        }
    }

    /// `None` when the queue can start; otherwise the reasons, one per line, in
    /// the order a person would want to fix them.
    ///
    /// These are the cases that would otherwise fail halfway through a batch,
    /// after the first file has already burned a few minutes.
    pub fn problems(&self, caps: &Caps) -> Vec<String> {
        let mut p = Vec::new();
        if self.inputs.is_empty() {
            p.push("Nothing queued — add a file or a folder.".to_string());
        }
        let missing: Vec<PathBuf> = self
            .inputs
            .iter()
            .filter(|i| !i.exists())
            .cloned()
            .collect();
        if !missing.is_empty() {
            p.push(format!(
                "{} queued path(s) no longer exist, e.g. {}",
                missing.len(),
                missing[0].display()
            ));
        }
        if self.use_out_dir && !self.out_dir.is_dir() {
            p.push(format!(
                "output folder {} is not a folder",
                self.out_dir.display()
            ));
        }
        if self.resample && !caps.can_resample() {
            p.push(
                "Resample is on but this ffmpeg has no libsoxr — turn it off, or install \
                 an ffmpeg built with --enable-libsoxr."
                    .to_string(),
            );
        }
        if self.ir && !caps.afir {
            p.push("IR calibration is on but this ffmpeg has no `afir` filter.".to_string());
        }
        if self.ir && self.ir_file.as_os_str().is_empty() {
            p.push("IR calibration is on but no IR file is chosen.".to_string());
        } else if self.ir && !self.ir_file.is_file() {
            p.push(format!(
                "IR file {} does not exist.",
                self.ir_file.display()
            ));
        }
        if self.depth == BitDepth::S24 && self.format == Format::Flac && !caps.flac24 {
            p.push(
                "This ffmpeg failed the 24-bit FLAC check (see the capabilities window). \
                 Choose 16-bit FLAC, or WAV for 24/32-bit and float."
                    .to_string(),
            );
        }
        if self.depth == BitDepth::S32 && self.format == Format::Flac {
            p.push(
                "FLAC cannot carry 32-bit PCM: ffmpeg writes a 24-bit stream and drops the \
                 low byte of every sample. Use WAV for 32-bit or float."
                    .to_string(),
            );
        }
        if self.ir && !(self.ir_norm >= -1.0 && self.ir_norm <= 2.0) {
            p.push("IR `irnorm` must be between -1 and 2.".to_string());
        }
        if self.ir && (self.ir_minp == 0 || self.ir_maxp < self.ir_minp) {
            p.push("IR partition size needs 0 < minp <= maxp.".to_string());
        }
        if self.block < 16 || self.block > 1_048_576 {
            p.push("Block size must be between 16 and 1048576 frames.".to_string());
        }
        if self.queue_blocks == 0 || self.queue_blocks > 4096 {
            p.push("Queue depth must be between 1 and 4096 blocks.".to_string());
        }
        if self.format == Format::Flac && self.flac_level > 12 {
            p.push("FLAC level is 0..=12 (ffmpeg's encoder; 12 is smallest).".to_string());
        }
        if self.format == Format::Flac
            && matches!(self.depth, BitDepth::S32 | BitDepth::F32 | BitDepth::F64)
        {
            p.push(
                "FLAC carries 16- or 24-bit integers only. Use WAV for 32-bit or float."
                    .to_string(),
            );
        }
        if self.use_resample_depth && self.resample_depth.is_float() {
            p.push(format!(
                "Resample target {} makes the intermediate float — the point of the \
                 resample step is to land on integer samples. {}",
                self.resample_depth.label(),
                BitDepth::quantize_note()
            ));
        }
        p
    }

    /// Peak memory a single file can hold, for the label next to the queue depth.
    pub fn peak_bytes_per_file(&self) -> u64 {
        // f64 interleaved, `queue_blocks` in flight, plus the same again for the
        // encode side: this is the bound `RenderCfg::queue_blocks` documents.
        (self.block as u64) * (self.queue_blocks as u64) * 8 * 12 * 2
    }

    /// Parse [`Self::order_text`], leaving it untouched if it does not parse.
    /// Returns an error message the panel shows in place of the text box.
    pub fn apply_order_text(&mut self) -> Result<(), String> {
        let t = self.order_text.trim();
        if t.is_empty() {
            self.order = None;
            return Ok(());
        }
        let mut out = Vec::new();
        for part in t.split([',', ' ', ';', '|']) {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            let s = Slot::parse(p).ok_or_else(|| {
                format!(
                    "'{p}' is not a channel name. Known: {}",
                    mask::format_order(&Slot::ALL)
                )
            })?;
            out.push(s);
        }
        if out.is_empty() {
            return Err("Channel order is empty.".to_string());
        }
        if !mask::is_downmixable(&out) {
            return Err(format!(
                "{} has no front left and right pair, so nothing can be downmixed",
                mask::format_order(&out)
            ));
        }
        self.order = Some(out);
        Ok(())
    }
}

/// A one-line summary of what a render will do, for the status bar and the log.
///
/// Deliberately written in plain words rather than filter syntax: this is the line
/// a person reads to confirm the run is what they meant before pressing Start.
pub fn describe(s: &Settings) -> String {
    let mut v = Vec::new();
    v.push(format!(
        "{} {}",
        s.depth.label(),
        match s.format {
            Format::Flac => format!("FLAC -{}", s.flac_level.min(12)),
            Format::Wav => "WAV".to_string(),
        }
    ));
    if s.resample {
        v.push(format!(
            "soxr {} Hz/{} bits{}",
            s.rate,
            s.precision,
            if s.dither == Dither::None {
                " no dither".to_string()
            } else {
                format!(" {}", s.dither.label())
            }
        ));
    } else {
        v.push("resampleless".to_string());
    }
    if s.ir {
        v.push(format!(
            "IR {} (irnorm={g}, wet={w})",
            s.ir_file
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "<none>".to_string()),
            g = s.ir_norm,
            w = s.ir_wet
        ));
    }
    v.push(format!("{} threads", s.threads()));
    v.push("f64 dsp".to_string());
    v.join(" · ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_loss_neutral() {
        // The stock path must be decode -> f64 -> downmix -> encode with nothing
        // done to the samples. Any default that alters audio is a bug.
        let s = Settings::default();
        let c = s.render_cfg();
        assert!(!c.resample.enable, "no resampling by default");
        assert!(!c.ir.enable, "no convolution by default");
        assert_eq!(c.out.dither, Dither::None, "no dither by default");
        assert!(c.out.resample.is_none(), "never resample after the downmix");
        assert!(c.order.is_none(), "trust the decoder's channel order");
        assert!(!c.sha256, "hashing is opt-in, not a surprise cost");
        assert!(
            c.out.copy_metadata,
            "metadata is carried unless told otherwise"
        );
        assert!(!c.out.overwrite, "never clobber an existing file unasked");
    }

    #[test]
    fn irnorm_survives_every_path() {
        // afir's own default is peak normalisation, which would silently change the
        // IR's gain and therefore the whole calibration.
        let mut s = Settings {
            ir: true,
            ir_file: PathBuf::from("/tmp/ir.wav"),
            ..Default::default()
        };
        let c = s.render_cfg();
        assert_eq!(c.ir.irnorm, Some(-1.0));
        assert_eq!(c.ir.wet, Some(1.0));
        s.ir_norm = 0.5;
        assert_eq!(s.render_cfg().ir.irnorm, Some(0.5));
    }

    #[test]
    fn unset_optionals_stay_unset() {
        let mut s = Settings {
            ir: true,
            ..Default::default()
        };
        let c = s.render_cfg();
        assert!(c.ir.dry.is_none(), "dry must not be sent unless asked for");
        assert!(c.ir.length.is_none());
        assert!(c.ir.gtype.is_none());
        assert!(c.resample.depth.is_none());
        s.use_ir_dry = true;
        s.ir_length = 0.5;
        s.use_resample_depth = true;
        let c = s.render_cfg();
        assert_eq!(c.ir.dry, Some(1.0));
        assert_eq!(c.ir.length, Some(0.5));
        assert_eq!(c.resample.depth, Some(BitDepth::S32));
    }

    #[test]
    fn threads_are_clamped_to_the_machine() {
        let mut s = Settings {
            threads: 100_000,
            ..Default::default()
        };
        assert_eq!(s.threads(), Settings::max_threads());
        s.threads = 0;
        assert_eq!(s.threads(), 1);
    }

    /// The bound is the encoder's, not the reference CLI's: ffmpeg's FLAC encoder
    /// accepts 0..=12 and refuses 13 at init.
    #[test]
    fn flac_level_is_bounded_like_ffmpeg_flac() {
        let s = Settings {
            flac_level: 99,
            ..Default::default()
        };
        match s.out_format() {
            OutFormat::Flac { level } => assert_eq!(level, 12),
            other => panic!("expected flac, got {other:?}"),
        }
        let ok = Settings {
            flac_level: 12,
            ..Default::default()
        };
        assert!(!ok
            .problems(&caps_none())
            .iter()
            .any(|p| p.contains("FLAC level")));
    }

    fn caps_none() -> Caps {
        Caps {
            ffmpeg_version: "test".into(),
            lavfi: true,
            ..Default::default()
        }
    }

    /// The panel's DC control has to reach the pipeline, and its default has to be the
    /// bit-exact one.
    ///
    /// Tested at the boundary rather than trusting the field: `render_cfg` is the only path
    /// from a ticked box to a rendered byte, and a control that sets a field nobody reads is
    /// indistinguishable from no control at all.
    /// The panel's fallback has to be the limiter's ratio, not the meter's.
    ///
    /// The same slip the CLI had: `unwrap_or_default()` reads as "the default", and
    /// `Ratio::default` is x4 because that is where metering starts. A limiter at x4 aims
    /// 0.62 dB below its ceiling instead of 0.04 dB, so every file landed half a dB lower
    /// than the number in the box said.
    #[test]
    fn the_limiter_ratio_default_is_the_limiters_not_the_meters() {
        assert_eq!(Ratio::default(), Ratio::X4, "the meter's default moved");
        assert_eq!(g6me_core::limiter::DEFAULT_RATIO, Ratio::X8);
        assert_eq!(
            Settings::default().limiter_ratio(),
            g6me_core::limiter::DEFAULT_RATIO,
            "the panel fell back to the meter's x4"
        );
        assert_eq!(
            Settings::default().render_cfg().limiter.ratio,
            g6me_core::limiter::DEFAULT_RATIO,
            "the accessor and the render config disagree about the default"
        );
        let s = Settings {
            limit_ratio: Some(Ratio::X16),
            ..Settings::default()
        };
        assert_eq!(s.limiter_ratio(), Ratio::X16);
        assert_eq!(s.render_cfg().limiter.ratio, Ratio::X16);
    }

    #[test]
    fn dc_block_defaults_off_and_reaches_the_render_config() {
        let s = Settings::default();
        assert!(!s.dc_block, "DC blocking must be off by default");
        assert!(
            !s.render_cfg().dc_block,
            "the default render must stay bit-exact"
        );

        let on = Settings {
            dc_block: true,
            ..Settings::default()
        };
        assert!(on.render_cfg().dc_block, "the ticked box never arrived");
        // Blocking DC is a DSP step of its own: it has to survive with the limiter off,
        // because an offset ships in the file whether or not anything is holding peaks.
        let no_limit = Settings {
            dc_block: true,
            limiter: false,
            ..Settings::default()
        };
        let cfg = no_limit.render_cfg();
        assert!(cfg.dc_block, "DC blocking vanished with the limiter off");
        assert!(
            !cfg.limiter.enable,
            "the limiter stopped being opt-out-able"
        );
    }

    #[test]
    fn problems_catches_the_expensive_mistakes() {
        let caps = Caps {
            ffmpeg_version: "test".into(),
            soxr: false,
            afir: false,
            flac: false,
            lavfi: true,
            flac24: false,
            notes: Vec::new(),
        };
        let mut s = Settings::default();
        assert!(
            s.problems(&caps)
                .iter()
                .any(|p| p.contains("Nothing queued")),
            "empty queue must be caught"
        );
        s.inputs = vec![PathBuf::from("/nonexistent/thing.mka")];
        assert!(s
            .problems(&caps)
            .iter()
            .any(|p| p.contains("no longer exist")));
        s.resample = true;
        s.ir = true;
        assert!(s.problems(&caps).iter().any(|p| p.contains("libsoxr")));
        assert!(s.problems(&caps).iter().any(|p| p.contains("afir")));
        assert!(s
            .problems(&caps)
            .iter()
            .any(|p| p.contains("failed the 24-bit FLAC check")));
        // 24-bit on an ffmpeg that fails the round-trip check must be refused
        // before any file is opened.
        s.depth = BitDepth::S16;
        assert!(!s
            .problems(&caps)
            .iter()
            .any(|p| p.contains("24-bit needs the reference")));
    }

    #[test]
    fn channel_order_parses_and_rejects() {
        let mut s = Settings {
            order_text: "FL, FR, FC, LFE, BL, BR, SL, SR".into(),
            ..Default::default()
        };
        s.apply_order_text().expect("valid order");
        assert_eq!(s.order.as_ref().map(|o| o.len()), Some(8));

        s.order_text = "FL,FR,NOPE".into();
        assert!(s.apply_order_text().is_err());
        assert_eq!(
            s.order.as_ref().map(|o| o.len()),
            Some(8),
            "kept the last good order"
        );

        s.order_text = String::new();
        s.apply_order_text().unwrap();
        assert!(s.order.is_none(), "empty box means auto");

        // Re-arm a working order, then check that a rejected edit leaves it alone.
        s.order_text = "FL, FR".into();
        s.apply_order_text()
            .expect("a stereo order is still an order");

        s.order_text = "FC,LFE".into();
        let e = s
            .apply_order_text()
            .expect_err("no front pair means nothing to downmix to");
        assert!(e.contains("front left"), "unhelpful error: {e}");
        assert_eq!(
            s.order.as_ref().map(|o| o.len()),
            Some(2),
            "a rejected edit must leave the working order alone"
        );
    }

    #[test]
    fn description_names_every_lossy_step() {
        let mut s = Settings::default();
        let d = describe(&s);
        assert!(d.contains("resampleless"), "{d}");
        assert!(d.contains("24-bit") || d.contains("24"), "{d}");
        assert!(!d.contains("IR"), "{d}");
        s.resample = true;
        s.ir = true;
        s.ir_file = PathBuf::from("/home/u/.config/mpv/ir.wav");
        let d = describe(&s);
        assert!(d.contains("soxr") && d.contains("ir.wav"), "{d}");
    }

    #[test]
    fn peak_memory_label_tracks_the_queue_settings() {
        let mut s = Settings::default();
        let bytes = s.peak_bytes_per_file();
        assert!(bytes > 0 && bytes < 1 << 30, "{bytes} bytes is not sane");
        s.queue_blocks *= 2;
        assert_eq!(
            s.peak_bytes_per_file(),
            bytes * 2,
            "the label must be honest"
        );
    }
}
