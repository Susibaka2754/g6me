//! Stage configuration and the exact ffmpeg command lines each stage emits.
//!
//! Everything here is a pure function of the config, which buys two things:
//! `--dry-run` can print precisely what will run, and the tests can assert the
//! filter strings (a typo in `dither_method` makes ffmpeg fail every render, so
//! it belongs in a test).
//!
//! ## Defaults are loss-neutral on purpose
//!
//! With the stock configuration the pipeline is
//! `decode -> f64 -> downmix -> encode`, with no resampler, no dither and no
//! convolution anywhere: each of those steps is opt-in, one control each. The
//! input side asks ffmpeg for `pcm_f64le`, which is an exact widening of the
//! decoder's `s32` output (all 2^31 integers are representable in f64, and the
//! scale factor is a power of two), so the downmixer sees the source values
//! unchanged.
//!
//! ## Channel identity
//!
//! Every input chain starts with `channelmap=channel_layout=<mask hex>`, which
//! *relabels* the decoded planes in place instead of letting swresample convert
//! between layouts. `aformat=channel_layouts=` would remix whenever the container's
//! label differs from the order g6me resolved (an explicit `--order`, or a layout
//! name read differently), folding a surround channel into the wrong slot.

use crate::error::{Error, Result};
use crate::limiter::Cfg as LimiterCfg;
use crate::mask::format_mask_hex;
use crate::pcm::{BitDepth, Dither, Grid24};
use crate::probe::{AudioInfo, Caps};
use crate::tp::Ratio;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Input-side normalization: rate conversion and (optionally) a bit-depth
/// reduction *before* the downmixer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResampleCfg {
    pub enable: bool,
    pub rate: u32,
    /// `None` keeps the source precision. Restricted to what libswresample can
    /// express as a filter format — see [`BitDepth::quantize_note`].
    pub depth: Option<BitDepth>,
    /// soxr precision in bits, 15..=33.
    pub precision: f64,
    pub dither: Dither,
    pub cheby: bool,
}

impl Default for ResampleCfg {
    fn default() -> Self {
        Self {
            enable: false,
            rate: 48000,
            depth: None,
            precision: 33.0,
            dither: Dither::None,
            cheby: false,
        }
    }
}

impl ResampleCfg {
    /// Formats a caller may ask for on the input side. 24-bit is deliberately
    /// absent: swresample has no `s24` sample format, so 24-bit only exists at
    /// the encoder.
    pub fn allowed_depths() -> [BitDepth; 4] {
        [BitDepth::S16, BitDepth::S32, BitDepth::F32, BitDepth::F64]
    }

    fn effective_depth(&self) -> Option<BitDepth> {
        self.depth.filter(|d| Self::allowed_depths().contains(d))
    }
}

/// afir's IR auto-gain types, by the exact names ffmpeg parses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Gtype {
    None,
    Peak,
    Dc,
    Gn,
    Ac,
    Rms,
}

impl Gtype {
    pub fn ffmpeg_name(self) -> &'static str {
        match self {
            Gtype::None => "none",
            Gtype::Peak => "peak",
            Gtype::Dc => "dc",
            Gtype::Gn => "gn",
            Gtype::Ac => "ac",
            Gtype::Rms => "rms",
        }
    }
}

/// `afir`'s `precision`, fixed to double.
///
/// The convolver and the downmixer both run their state and accumulation in f64,
/// and that is an invariant rather than a setting: a 200 ms IR convolved in f32
/// loses the detail a measured headphone correction exists to apply, and the
/// downmixer's recursive sections are bit-exact against the reference C port only
/// in its `G6CT_DOUBLE` (f64) build. Bit depth at the *output* is a user choice;
/// arithmetic inside the app is not. There is therefore no `float` option here.
pub const AFIR_PRECISION: &str = "double";

/// Linear-phase headphone correction, applied per input channel before the
/// downmixer — the same thing `mpv`'s
/// `af=lavfi=[amovie=ir.wav[ir2];[in][ir2]afir=...]` does.
///
/// The defaults reproduce that graph option-for-option
/// (`irfmt=mono:irnorm=-1:wet=1:minp=4096:maxp=4096:precision=double`); knobs
/// left `None` are *omitted* from the filter string so ffmpeg's own defaults
/// apply, which is what keeps parity with an mpv config.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IrCfg {
    pub enable: bool,
    pub file: Option<PathBuf>,
    /// `irnorm`, -1..=2. `-1` = no auto gain (raw IR), which is what a measured
    /// correction filter needs.
    pub irnorm: Option<f64>,
    pub gtype: Option<Gtype>,
    pub wet: Option<f64>,
    pub dry: Option<f64>,
    pub length: Option<f64>,
    pub link: Option<bool>,
    pub minp: u32,
    pub maxp: u32,
}

impl Default for IrCfg {
    fn default() -> Self {
        Self {
            enable: false,
            file: None,
            irnorm: Some(-1.0),
            gtype: None,
            wet: Some(1.0),
            dry: None,
            length: None,
            link: None,
            minp: 4096,
            maxp: 4096,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OutFormat {
    /// FLAC, written by ffmpeg's own encoder.
    ///
    /// Depth is 16 or 24 bit integer. ffmpeg's encoder advertises `s16 s32` as
    /// its input sample formats and derives the stream's bits-per-sample from what
    /// it is fed: fed 24-bit samples it writes a genuine `bps=24` stream, and
    /// decoding that stream returns the input bytes exactly (`flac -t` says ok).
    /// It cannot carry 32-bit PCM — fed true 32-bit it still writes bps=24 and
    /// drops the low byte of every sample — so 32-bit output is WAV only, and
    /// asking for 32-bit FLAC is rejected rather than quietly downgraded.
    ///
    /// `level` is ffmpeg's FLAC compression level, 0..=12 (12 packs smallest);
    /// 13 is refused by the encoder at init, which `g6me --caps` demonstrates.
    Flac {
        level: u32,
    },
    Wav,
    /// Write no file at all. This is what live monitoring uses: the same pump,
    /// the same filters and the same downmixer, with the samples handed to a
    /// device instead of an encoder — so what you hear is what a render produces
    /// rather than a second implementation of the chain.
    Null,
}

impl OutFormat {
    pub fn label(&self) -> String {
        match self {
            OutFormat::Flac { level } => format!("FLAC (level {level})"),
            OutFormat::Wav => "WAV".to_string(),
            OutFormat::Null => "no file (monitor only)".to_string(),
        }
    }
    pub fn extension(&self) -> &'static str {
        match self {
            OutFormat::Flac { .. } => "flac",
            OutFormat::Wav => "wav",
            OutFormat::Null => "",
        }
    }

    /// True when nothing is written to disk.
    pub fn is_null(&self) -> bool {
        matches!(self, OutFormat::Null)
    }
}

/// Output-side encoding, including the optional second resample.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutCfg {
    pub format: OutFormat,
    pub depth: BitDepth,
    /// `None` = passthrough: no resampling, output at the working rate.
    pub resample: Option<u32>,
    pub precision: f64,
    pub dither: Dither,
    pub copy_metadata: bool,
    pub overwrite: bool,
    /// `-filter_complex_threads`; 1 keeps a multi-file batch from oversubscribing
    /// the CPU with one slice-threaded ffmpeg per file.
    pub filter_threads: Option<u32>,
}

impl Default for OutCfg {
    fn default() -> Self {
        Self {
            format: OutFormat::Flac { level: 8 },
            depth: BitDepth::S24,
            resample: None,
            precision: 33.0,
            dither: Dither::None,
            copy_metadata: true,
            overwrite: false,
            filter_threads: None,
        }
    }
}

impl OutCfg {
    /// True when g6me rounds the stream onto the 24-bit grid itself ([`Grid24`]), so the
    /// encoder gets no dither filter of its own. An output resampler would move the
    /// samples off the grid again, so that case is left to ffmpeg.
    pub fn grid24(&self, work_rate: u32) -> bool {
        self.depth == BitDepth::S24
            && !self.format.is_null()
            && !self.resample.is_some_and(|r| r != work_rate)
    }

    /// Validate a format/depth pair against what is actually available.
    pub fn validate(&self, caps: &Caps) -> Result<()> {
        if self.format.is_null() {
            return Ok(());
        }
        if self.depth == BitDepth::S24 && !Grid24::supports(self.dither) {
            return Err(Error::msg(format!(
                "dither '{}' is not available for 24-bit output: ffmpeg's noise-shaped \
                 dithers go unstable at 24 bits. Use none, rectangular, triangular or \
                 triangular_hp, or 16-bit output for noise shaping.",
                self.dither.ffmpeg_name()
            )));
        }
        if let OutFormat::Flac { level } = &self.format {
            if self.depth.is_float() {
                return Err(Error::msg(
                    "FLAC is integer-only; choose WAV for float output",
                ));
            }
            if !matches!(level, 0..=12) {
                // Measured against ffmpeg n9.0.2: 0..=12 encode, 13 and up fail at
                // encoder init with "invalid compression level". The reference flac
                // CLI's -0..-8 presets are a different scale and no longer in play.
                return Err(Error::msg(
                    "FLAC compression level must be 0..=12 (ffmpeg's FLAC encoder; \
                     12 is the smallest file, 13 is refused)",
                ));
            }
            if self.depth == BitDepth::S24 && !caps.flac24 {
                return Err(Error::msg(
                    "this ffmpeg was not verified able to write a lossless 24-bit FLAC \
                     (run `g6me --caps` for the reason). Use '--depth 16', or \
                     '--format wav --depth 24' which never touches the FLAC encoder.",
                ));
            }
            if self.depth == BitDepth::S32 {
                return Err(Error::msg(
                    "32-bit FLAC is not available: ffmpeg's FLAC encoder writes at \
                     most 24-bit and drops the low byte of every 32-bit sample, so \
                     the file would be labelled 24-bit while looking 32-bit. Use \
                     '--format flac --depth 24' for the lossless int file, or \
                     '--format wav --depth 32' / '--depth f32' for the full width \
                     of the DSP output.",
                ));
            }
        }
        if self.resample == Some(0) {
            return Err(Error::msg("output sample rate must be > 0"));
        }
        Ok(())
    }
}

/// Everything one render needs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RenderCfg {
    pub resample: ResampleCfg,
    pub ir: IrCfg,
    pub out: OutCfg,
    /// Override the decoder-reported channel order (interleaved order).
    pub order: Option<Vec<crate::mask::Slot>>,
    /// Frames per processing block.
    pub block: usize,
    /// Blocks allowed to be in flight between the decode, DSP and encode stages.
    ///
    /// This is the whole memory bound of a render: peak is roughly
    /// `queue_blocks * block * channels * 8` bytes per file, so a 3-hour Atmos
    /// concert streams at the same cost as a 3-second clip. Deep enough to absorb
    /// a scheduler hiccup, shallow enough that cancelling does not wait for a
    /// queue to drain.
    pub queue_blocks: usize,
    /// Strip DC before the gain trim and the limiter, with a 5 Hz first-order blocker.
    ///
    /// Off by default, and off is the bit-exact path. `Dsp` is the vendor port: its
    /// all-pass cascade has `H(1) = 1` exactly and adds the module's anti-denormal bias,
    /// a settled +188 dBFS-below-full-scale `+4 * EPSILON` on the left and zero on the
    /// right, which is 0.0034 of one 24-bit step and not worth disturbing. What is worth
    /// catching is DC that came in from the source or from an IR that was not mean-free:
    /// the 7.1 matrix sums eight channels into two, and carries an input offset at +5.179
    /// on the left, where it reads to the limiter as level.
    pub dc_block: bool,
    /// Which audio stream to take, 0-based.
    pub stream: usize,
    /// Over-sampling ratio the true peak is measured at, for every render.
    ///
    /// Measured always rather than behind a flag: it is the one number in a report that
    /// says something about the signal instead of about the pipeline, and the cost is
    /// [`Ratio::mac_per_sample`] multiply-adds per sample per channel — 48 at the
    /// default x4, which is a fraction of what the downmixer already does.
    ///
    /// The reading is of the signal the encoder receives, so it is the same number a
    /// BS.1770 consumer gets from the finished file, not a pre-limiter estimate.
    #[serde(default)]
    pub true_peak: Ratio,
    /// True-peak limiter, applied after the downmixer and before the encoder.
    ///
    /// Off by default, like every other stage here. When enabled it is the last thing
    /// that touches the samples, because a ceiling asserted before the last gain stage
    /// is not a ceiling.
    #[serde(default)]
    pub limiter: LimiterCfg,
    /// Gain staging: one constant multiplier applied to every sample, in dB, before the
    /// limiter and the encoder.
    ///
    /// This is the level half of a setpoint. It is a separate step from the ceiling on
    /// purpose: a constant gain costs nothing and changes no shape, while a ceiling does
    /// both, so the level work belongs here and the limiter keeps only the peaks.
    #[serde(default)]
    pub output_gain_db: f64,
    /// Hash every finished output, twice over: the samples handed to the encoder
    /// (container-independent, so a WAV and a FLAC of one render agree) and the file on
    /// disk (what `sha256sum` prints, so the number is checkable with a tool the user
    /// already has). See [`crate::pipeline::Report::file_sha256`].
    pub sha256: bool,
}

impl RenderCfg {
    pub fn with_defaults() -> Self {
        Self {
            block: 4096,
            ..Default::default()
        }
    }

    /// One number, two effects: the file is trimmed by `db` and the true-peak ceiling is
    /// set to `db` dBTP.
    ///
    /// This is what the panel's setpoint means. -4 leaves the file 4 dB quieter and puts
    /// nothing above -4 dBTP, and it is two mechanisms doing it: a constant gain takes the
    /// level off every sample, which costs nothing and bends nothing, and the limiter is
    /// left with only the peaks that survive the trim. Asking a ceiling to do the level
    /// work as well is how a limiter ends up riding gain through 94 % of a file rather than
    /// catching peaks in it.
    pub fn apply_setpoint(&mut self, db: f64) {
        self.output_gain_db = db;
        self.limiter.ceiling_db = db;
        self.limiter.enable = true;
    }
}

/// What the input stage will actually do, after reconciling the config with the
/// source. `Noop` means "enabled, but there is nothing to do", which the report
/// says out loud instead of pretending a resampler ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputStage {
    Passthrough,
    Noop,
    Soxr,
    SoxrAndDepth,
    DepthOnly,
}

impl InputStage {
    pub fn label(self) -> &'static str {
        match self {
            InputStage::Passthrough => "decode only",
            InputStage::Noop => "enabled, no-op (same rate, no dither)",
            InputStage::Soxr => "soxr resample",
            InputStage::SoxrAndDepth => "soxr resample + depth",
            InputStage::DepthOnly => "depth only",
        }
    }
}

/// Decide + build the filter steps for the input chain. Steps are joined with
/// `,` for `-af`, or embedded in a `-filter_complex` when the IR stage is on.
pub fn input_stage(cfg: &RenderCfg, info: &AudioInfo, mask: u32) -> (InputStage, Vec<String>) {
    let want_rate = cfg.resample.enable && cfg.resample.rate != info.sample_rate;
    let want_dither = cfg.resample.enable && cfg.resample.dither != Dither::None;
    let depth = if cfg.resample.enable {
        cfg.resample.effective_depth()
    } else {
        None
    };
    // A depth request is dropped only when it provably cannot change anything. swresample
    // computes in double, so asking for `f64` is not a step at all.
    //
    // Nothing else is dropped. The probe reports a bit *width*, not a sample format, so a
    // file with bits=32 could be `s32` or `flt` or `flt64`, and `--depth s32` on the float
    // one is exactly the conversion the caller named. What the old condition did was drop
    // `s32` for every source that was *not* 32-bit — the one case where the step is
    // guaranteed to do something — so a user asking for 32-bit output of a 24-bit file got
    // the file back unchanged and a chain line that no longer matched the command. A
    // redundant filter argument costs one parse; a requested depth that silently evaporates
    // costs the render.
    let depth = depth.filter(|d| !matches!(d, BitDepth::F64));

    let stage = match (want_rate || want_dither, depth.is_some()) {
        (true, true) => InputStage::SoxrAndDepth,
        (true, false) => InputStage::Soxr,
        (false, true) => InputStage::DepthOnly,
        (false, false) if cfg.resample.enable => InputStage::Noop,
        (false, false) => InputStage::Passthrough,
    };

    // Declared identity of the decoded stream; first so every later filter sees a
    // known layout. A relabel, never a conversion: see the module docs.
    let mut steps = vec![format!(
        "channelmap=channel_layout={}",
        format_mask_hex(mask)
    )];

    match stage {
        InputStage::Passthrough | InputStage::Noop => {}
        InputStage::Soxr | InputStage::SoxrAndDepth => {
            // Without this, an integer source reaches soxr as float32 and the
            // precision asked for below is not the one delivered.
            steps.push("aformat=sample_fmts=dbl".into());
            let mut s = format!(
                "aresample={}:resampler=soxr:precision={}:dither_method={}",
                cfg.resample.rate,
                cfg.resample.precision as u32,
                cfg.resample.dither.ffmpeg_name()
            );
            if cfg.resample.cheby {
                s.push_str(":cheby=true");
            }
            steps.push(s);
        }
        InputStage::DepthOnly => {
            steps.push(format!(
                "aresample=dither_method={}",
                cfg.resample.dither.ffmpeg_name()
            ));
        }
    }

    if let Some(d) = depth {
        steps.push(format!("aformat=sample_fmts={}", d.swr_fmt()));
    }

    (stage, steps)
}

/// afir options, in the order mpv writes them, omitting what the user left alone.
pub fn afir_filter(cfg: &IrCfg) -> String {
    let mut f = String::from("afir=irfmt=mono");
    if let Some(v) = cfg.irnorm {
        f.push_str(&format!(":irnorm={v}"));
    }
    if let Some(g) = cfg.gtype {
        f.push_str(&format!(":gtype={}", g.ffmpeg_name()));
    }
    if let Some(v) = cfg.wet {
        f.push_str(&format!(":wet={v}"));
    }
    if let Some(v) = cfg.dry {
        f.push_str(&format!(":dry={v}"));
    }
    if let Some(v) = cfg.length {
        f.push_str(&format!(":length={v}"));
    }
    if let Some(l) = cfg.link {
        f.push_str(&format!(":irlink={}", if l { "true" } else { "false" }));
    }
    f.push_str(&format!(":minp={}:maxp={}", cfg.minp, cfg.maxp));
    f.push_str(&format!(":precision={AFIR_PRECISION}"));
    f
}

/// Command line for the producing (decode) child. Reads PCM to us on stdout as
/// interleaved `f64le`, `mask.count_ones()` channels, source rate (or the
/// resampled rate).
pub fn input_args(
    cfg: &RenderCfg,
    info: &AudioInfo,
    mask: u32,
    src: &Path,
    ir_info: Option<&AudioInfo>,
) -> Result<(Vec<OsString>, u32, usize)> {
    let (_, steps) = input_stage(cfg, info, mask);
    let work_rate = if cfg.resample.enable {
        cfg.resample.rate
    } else {
        info.sample_rate
    };
    let channels = mask.count_ones() as usize;

    let mut args: Vec<OsString> = vec!["-v".into(), "error".into(), "-nostdin".into()];
    if let Some(t) = cfg.out.filter_threads {
        args.push("-filter_complex_threads".into());
        args.push(t.to_string().into());
    }
    args.push("-i".into());
    args.push(crate::util::ffmpeg_path_arg(src));

    if cfg.ir.enable {
        let ir_path = cfg
            .ir
            .file
            .as_ref()
            .ok_or_else(|| Error::msg("IR calibration is enabled but no IR file is set"))?;
        args.push("-i".into());
        args.push(crate::util::ffmpeg_path_arg(ir_path));

        // afir convolves in the main path's time base; an IR recorded at another
        // rate is brought to it, which is the only case where we resample an IR.
        let mut ir_steps = vec!["aformat=sample_fmts=dbl:channel_layouts=mono".to_string()];
        if let Some(i) = ir_info {
            if i.sample_rate != work_rate {
                ir_steps.insert(
                    0,
                    format!(
                        "aresample={}:resampler=soxr:precision=33:dither_method=none",
                        work_rate
                    ),
                );
                // The calibration gain, and the reason it is not optional.
                //
                // A resampler treats an impulse response as the *signal* it is: the
                // 96 kHz boxcar of height h comes out the far side as a 48 kHz boxcar of
                // the same height h and half the taps. Physically that is the same
                // acoustic response, but afir's raw convolution (`irnorm=-1`, which is
                // what a measured correction filter needs, and the default here) sums
                // taps without the sample period, so half the taps means half the gain:
                // an IR authored at 96 kHz comes out 6.02 dB quieter at 48 kHz than the
                // identical IR convolved at its own rate. Nothing about the file changed
                // except the rate it was recorded at, and the level did.
                //
                // Measured on the installed FFmpeg 9.0.2, one boxcar IR of height 0.002
                // over 1000 samples at 96 kHz, 100 Hz probe tone, `irnorm=-1`:
                //   IR resampled to 48 kHz                    -44.095 dB
                //   same IR convolved at native 96 kHz        -38.052 dB   <- reference
                //   resampled + this volume= factor           -38.075 dB   <- matches
                // Multiplying by `f_ir / f_work` (here 2, +6.02 dB) restores the
                // native-rate answer. It goes BEFORE the aresample, because the
                // expression reads the sample rate of the stream at that point in the
                // chain, and after the resampler it would read the target rate and apply
                // a factor of 1 — a no-op that looks like it was applied.
                //
                // Only when the IR is really being moved, and only when nothing else is
                // already fixing the level: with `irnorm` >= 0, or left unset (affirming
                // afir's own default of 1), afir normalizes the IR itself and this factor
                // would be a 6 dB error of its own.
                if cfg.ir.irnorm.is_some_and(|v| v < 0.0) {
                    ir_steps.insert(
                        0,
                        format!("volume=volume='sample_rate/{work_rate}':eval=once"),
                    );
                }
            }
        }
        let mut main = steps.clone();
        main.push("aformat=sample_fmts=dbl".into());
        let graph = format!(
            "[0:a:{}]{}[am];[1:a:0]{}[ir];[am][ir]{}[o]",
            cfg.stream,
            main.join(","),
            ir_steps.join(","),
            afir_filter(&cfg.ir)
        );
        args.push("-filter_complex".into());
        args.push(graph.into());
        args.push("-map".into());
        args.push("[o]".into());
    } else {
        // Always pass -af, even when the only step is the layout declaration:
        // for a container that reports no layout it pins the channel count we
        // expect instead of letting the muxer guess.
        args.push("-af".into());
        args.push(steps.join(",").into());
        // The stream that was probed, not whichever one ffmpeg would pick: the
        // layout above is only true of that one.
        args.push("-map".into());
        args.push(format!("0:a:{}", cfg.stream).into());
    }

    args.push("-f".into());
    args.push("f64le".into());
    args.push("-c:a".into());
    args.push("pcm_f64le".into());
    // The literal stdout marker, deliberately not run through ffmpeg_path_arg: see
    // the guard in that function that keeps `./-` from being built out of it.
    args.push("-".into());
    Ok((args, work_rate, channels))
}

/// What the encoder will be fed: interleaved `f64le`, 2 channels, `work_rate`.
pub fn encoder_pump_args(work_rate: u32) -> Vec<String> {
    vec![
        "-f".into(),
        "f64le".into(),
        "-c:a".into(),
        "pcm_f64le".into(),
        "-ar".into(),
        work_rate.to_string(),
        "-ch_layout".into(),
        "stereo".into(),
        "-i".into(),
        "pipe:0".into(),
    ]
}

/// The encode-side filter, or `None` when the encoder is fed exactly what the DSP
/// produced and has nothing to convert.
///
/// Lives apart from [`output_args`] because the two reasons it exists — a rate change
/// and a dither — are independent, and the third thing it carries (the depth the dither
/// lands on) belongs to the dither rather than to either.
fn encoder_filter(o: &OutCfg, work_rate: u32) -> Option<String> {
    let need_resample = o.resample.is_some_and(|r| r != work_rate);
    // At 24 bits without an output resample the dither is applied in-app with the
    // rounding (`OutCfg::grid24`), so ffmpeg must not add a second one.
    let need_dither = o.dither != Dither::None && !o.grid24(work_rate);
    if !(need_resample || need_dither) {
        return None;
    }
    let rate = o.resample.unwrap_or(work_rate);
    let mut s = if need_resample {
        format!(
            "aresample={rate}:resampler=soxr:precision={}:dither_method={}",
            o.precision as u32,
            o.dither.ffmpeg_name()
        )
    } else {
        format!("aresample=dither_method={}", o.dither.ffmpeg_name())
    };
    if o.resample.is_some() && !need_resample && o.precision > 0.0 {
        s.push_str(&format!(":precision={}", o.precision as u32));
    }
    // Dither belongs at the depth the file will actually carry.
    //
    // libswresample has no 24-bit sample format, so a 24-bit request lands on s32
    // (`-sample_fmt` in [`output_args`]) and the encoder derives `bps=24` from it. Left
    // alone, the dither is generated and applied at the s32 least significant bit and the
    // low 8 bits are then dropped on the way into the encoder: a dither 48.13 dB (2^8)
    // below the bit it is supposed to straddle. Measured with the installed FFmpeg 9.0.2,
    // feeding a constant equal to a quarter of the 24-bit LSB through this exact chain:
    //   as it was              dither on -> -inf dB, the same file no dither produces
    //   output_sample_bits=24  peak -138.47 dB, RMS -143.61 dB, i.e. real +-1 LSB noise
    // The option is `output_sample_bits`; `out_sample_bits` does not exist on this filter
    // and fails the graph at init, which is worth writing down because the wrong name is
    // the one that circulates.
    //
    // It moves bytes only when there is a dither to place: with `dither_method=none` a
    // 2-second slice of 810.mka hashes identically with and without it
    // (0fa5a4bc18c68f13...), and identically to the plain auto-convert path. So the
    // undithered pipeline — the reference render among them — is untouched. This changes
    // what dither does, not what undithered audio is.
    if o.depth == BitDepth::S24 {
        s.push_str(":output_sample_bits=24");
    }
    Some(s)
}

/// Command line for the consuming (encode) child — one ffmpeg process, whatever
/// the format.
pub fn output_args(
    cfg: &RenderCfg,
    work_rate: u32,
    out_path: &Path,
    src: &Path,
    caps: &Caps,
) -> Result<Vec<OsString>> {
    let o = &cfg.out;
    o.validate(caps)?;

    if o.format.is_null() {
        // No encoder child at all; the pump hands samples to the tap.
        return Ok(Vec::new());
    }
    let mut args: Vec<OsString> = vec!["-v".into(), "error".into(), "-nostdin".into()];
    if let Some(t) = o.filter_threads {
        args.push("-filter_complex_threads".into());
        args.push(t.to_string().into());
    }
    args.extend(encoder_pump_args(work_rate).into_iter().map(OsString::from));
    if o.copy_metadata {
        // Second input, read for its header/tags only.
        args.push("-i".into());
        args.push(crate::util::ffmpeg_path_arg(src));
    }

    if let Some(f) = encoder_filter(o, work_rate) {
        args.push("-af".into());
        args.push(f.into());
    }
    if o.copy_metadata {
        args.push("-map".into());
        args.push("0:a".into());
        args.push("-map_metadata".into());
        args.push("1".into());
        args.push("-map_metadata:s:0".into());
        args.push("1:s:0".into());
    }
    args.push("-y".into());

    match &o.format {
        OutFormat::Flac { level } => {
            args.push("-c:a".into());
            args.push("flac".into());
            args.push("-compression_level".into());
            args.push(level.to_string().into());
            // Ask for the depth explicitly instead of letting the encoder pick: a
            // 24-bit request lands on s32 (libswresample has no s24 format) and the
            // encoder then derives bps=24 from it, which is what makes the stream
            // genuinely 24-bit rather than a widened one.
            args.push("-sample_fmt".into());
            args.push(o.depth.swr_fmt().into());
            args.push(crate::util::ffmpeg_path_arg(out_path));
        }
        OutFormat::Wav => {
            args.push("-c:a".into());
            args.push(o.depth.codec().into());
            // A plain RIFF header tops out at 4 GiB; past that ffmpeg writes a broken
            // one and still exits 0. RF64 is used only when needed.
            args.push("-rf64".into());
            args.push("auto".into());
            args.push(crate::util::ffmpeg_path_arg(out_path));
        }
        OutFormat::Null => unreachable!("handled above"),
    }
    Ok(args)
}

/// Plain-language description of what will happen to the samples, in order. The
/// GUI shows this so the loss chain is never something you have to reconstruct
/// from three dropdowns.
pub fn loss_chain(cfg: &RenderCfg, info: &AudioInfo, mask: u32) -> Vec<String> {
    let mut v = Vec::new();
    let (stage, _) = input_stage(cfg, info, mask);
    v.push(format!(
        "1. decode {} {} ch {} Hz -> f64 (exact widening)",
        info.codec_name, info.channels, info.sample_rate
    ));
    match stage {
        InputStage::Passthrough => {
            v.push("2. input stage: off (source values reach the DSP unchanged)".into())
        }
        InputStage::Noop => v.push(format!("2. input stage: {}", stage.label())),
        other => v.push(format!(
            "2. input stage: {} -> {} Hz, dither {}, precision {} bits",
            other.label(),
            cfg.resample.rate,
            cfg.resample.dither.label(),
            cfg.resample.precision as u32
        )),
    }
    if cfg.ir.enable {
        v.push(format!(
            "3. IR convolution (afir, mono, {}): {}",
            AFIR_PRECISION,
            cfg.ir
                .file
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<unset>".into())
        ));
    } else {
        v.push("3. IR convolution: off".into());
    }
    v.push(format!(
        "4. downmix mask {mask:#06x} in {} -> stereo (bit-exact port, {})",
        crate::mask::format_order(&order_for_mask(mask)),
        f64::PRECISION_NAME
    ));
    // 5. the true-peak stage. Printed whether or not it is limiting, because the meter
    // runs either way and a dry run that hides the one stage that measures the file
    // would leave `--dry-run` unable to explain the number in the report.
    let la = cfg.limiter;
    let work_rate = effective_input_rate(cfg, info);
    let trim = if cfg.output_gain_db == 0.0 {
        String::new()
    } else {
        format!("output gain {:+.2} dB, ", cfg.output_gain_db)
    };
    v.push(if !la.enable {
        format!(
            "5. {trim}true-peak: meter at {}x ({}), limiter off",
            cfg.true_peak.n(),
            cfg.true_peak.chain()
        )
    } else {
        format!(
            "5. {trim}true-peak: limit to {:.2} dBTP at {}x ({}), lookahead {:.2} ms -> \
             {:.2} ms effective, +{} samples latency, aims {:.2} dB low ({}x meter)",
            la.ceiling_db,
            la.ratio.n(),
            la.ratio.chain(),
            la.lookahead_ms,
            la.effective_lookahead_ms(work_rate),
            la.latency_samples(work_rate),
            la.guard_db(),
            cfg.true_peak.n(),
        )
    });
    let o = &cfg.out;
    let rate = match o.resample {
        Some(r) if r != effective_input_rate(cfg, info) => {
            format!("soxr -> {r} Hz, dither {}", o.dither.label())
        }
        _ => "no resample".to_string(),
    };
    let grid = if o.grid24(work_rate) {
        format!(
            " (rounded to the 24-bit grid by g6me, dither {})",
            o.dither.label()
        )
    } else {
        String::new()
    };
    v.push(format!(
        "6. encode {} {}{grid}{}",
        o.format.label(),
        o.depth.label(),
        if rate == "no resample" {
            String::new()
        } else {
            format!(", {rate}")
        },
    ));
    v
}

/// Rate the DSP sees, given the config and the source.
pub fn effective_input_rate(cfg: &RenderCfg, info: &AudioInfo) -> u32 {
    if cfg.resample.enable {
        cfg.resample.rate
    } else {
        info.sample_rate
    }
}

/// Best-effort inverse of [`crate::mask::mask_of`], for the summary line only.
fn order_for_mask(mask: u32) -> Vec<crate::mask::Slot> {
    use crate::mask::Slot;
    Slot::ALL
        .into_iter()
        .filter(|s| mask & s.bit() != 0)
        .collect()
}

/// Marker trait-ish helper so `loss_chain` prints the same word the DSP reports.
trait PrecisionName {
    const PRECISION_NAME: &'static str;
}
impl PrecisionName for f64 {
    const PRECISION_NAME: &'static str = "float64";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::Slot;
    use crate::probe::Caps;

    fn info() -> AudioInfo {
        AudioInfo {
            stream_index: 0,
            codec_name: "truehd".into(),
            profile: Some("Dolby TrueHD + Dolby Atmos".into()),
            sample_rate: 48000,
            channels: 8,
            layout: Some("7.1".into()),
            bits: Some(24),
            duration: 240.0,
        }
    }

    #[test]
    fn defaults_emit_no_resampler_at_all() {
        let cfg = RenderCfg::with_defaults();
        let (stage, steps) = input_stage(&cfg, &info(), crate::dsp::MASK_7_1);
        assert_eq!(stage, InputStage::Passthrough);
        // only the layout declaration, no aresample anywhere
        assert_eq!(steps, vec!["channelmap=channel_layout=0x63f".to_string()]);
        assert!(!steps.join(",").contains("aresample"));
    }

    #[test]
    fn decode_maps_the_probed_stream_and_relabels_instead_of_remixing() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.stream = 2;
        let (args, _, _) = input_args(
            &cfg,
            &info(),
            crate::dsp::MASK_7_1,
            Path::new("/x/i.mka"),
            None,
        )
        .unwrap();
        let j = crate::util::args_display(&args);
        assert!(j.contains("-map 0:a:2"), "{j}");
        assert!(!j.contains("aformat=channel_layouts"), "{j}");
    }

    #[test]
    fn wav_output_switches_to_rf64_past_4_gib() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.format = OutFormat::Wav;
        let args = output_args(
            &cfg,
            48000,
            Path::new("/tmp/o.wav"),
            Path::new("/tmp/i.mka"),
            &caps_flac24(),
        )
        .unwrap();
        assert!(
            crate::util::args_display(&args).contains("-rf64 auto"),
            "{args:?}"
        );
    }

    #[test]
    fn dither_at_24_bit_is_done_in_app_and_noise_shaping_is_refused() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.dither = Dither::Triangular;
        let args = output_args(
            &cfg,
            48000,
            Path::new("/tmp/o.flac"),
            Path::new("/tmp/i.mka"),
            &caps_flac24(),
        )
        .unwrap();
        assert!(
            !crate::util::args_display(&args).contains("dither"),
            "{args:?}"
        );
        assert!(cfg.out.grid24(48000));

        cfg.out.depth = BitDepth::S16;
        let args = output_args(
            &cfg,
            48000,
            Path::new("/tmp/o.flac"),
            Path::new("/tmp/i.mka"),
            &caps_flac24(),
        )
        .unwrap();
        assert!(
            crate::util::args_display(&args).contains("dither_method=triangular"),
            "{args:?}"
        );

        cfg.out.depth = BitDepth::S24;
        cfg.out.dither = Dither::Shibata;
        assert!(cfg.out.validate(&caps_flac24()).is_err());
    }

    #[test]
    fn default_output_is_flac24_with_no_dither_and_no_resample() {
        let cfg = RenderCfg::with_defaults();
        let args = output_args(
            &cfg,
            48000,
            Path::new("/tmp/o.flac"),
            Path::new("/tmp/i.mka"),
            &caps_flac24(),
        )
        .unwrap();
        assert_eq!(cfg.out.depth, BitDepth::S24);
        let j = crate::util::args_display(&args);
        assert!(j.contains("-c:a flac"), "{j}");
        assert!(j.contains("-sample_fmt s32"), "{j}");
        assert!(!j.contains("aresample"), "{j}");
        assert!(!j.contains("dither"), "{j}");
    }

    /// The dither has to be generated at the depth the file carries, not at the width
    /// libswresample happens to work in.
    ///
    /// Sabotage check: delete the `output_sample_bits` branch in `encoder_filter` and the
    /// first assertion fails while everything else still passes — which is exactly how it
    /// shipped: a dither 48 dB too quiet to reach the bit being quantised is invisible to
    /// every test that only checks that a dither was *requested*.
    #[test]
    fn a_24_bit_dither_is_generated_at_the_24_bit_lsb() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.dither = Dither::Triangular;
        assert_eq!(cfg.out.depth, BitDepth::S24);
        // Only an output resample leaves the 24-bit dither to ffmpeg; without one it is
        // done in-app (`OutCfg::grid24`).
        cfg.out.resample = Some(44100);
        assert!(!cfg.out.grid24(48000));
        let j = crate::util::args_display(
            &output_args(
                &cfg,
                48000,
                Path::new("/tmp/o.flac"),
                Path::new("/tmp/i.mka"),
                &caps_flac24(),
            )
            .unwrap(),
        );
        assert!(j.contains("dither_method="), "{j}");
        assert!(
            j.contains("output_sample_bits=24"),
            "24-bit output quantises through s32, so without this the dither lands on \
             the 32-bit LSB and the encoder throws it away: {j}"
        );

        // 16-bit asks libswresample for s16, a format it has, so the dither is already
        // straddling the right bit. Adding the option there would be noise at best.
        let mut c16 = RenderCfg::with_defaults();
        c16.out.depth = BitDepth::S16;
        c16.out.dither = Dither::Triangular;
        let j16 = crate::util::args_display(
            &output_args(
                &c16,
                48000,
                Path::new("/tmp/o.flac"),
                Path::new("/tmp/i.mka"),
                &caps_flac24(),
            )
            .unwrap(),
        );
        assert!(j16.contains("dither_method="), "{j16}");
        assert!(
            !j16.contains("output_sample_bits"),
            "s16 is a real swr format, the depth option belongs to the s32 detour: {j16}"
        );
    }

    /// An impulse response carries its own sample rate, and afir's raw convolution sums
    /// taps without the sample period that makes them a response. See the measurement in
    /// `input_args`: a 96 kHz IR convolved at 48 kHz comes out 6.02 dB quiet unless the
    /// ratio of the two rates is put back.
    #[test]
    fn a_resampled_ir_is_recalibrated_to_its_own_rate() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.ir.enable = true;
        cfg.ir.file = Some(PathBuf::from("/x/ir96.wav"));
        let mut ir = info();
        ir.sample_rate = 96_000;
        ir.channels = 1;
        ir.layout = Some("mono".into());
        let (args, rate, _) = input_args(
            &cfg,
            &info(),
            crate::dsp::MASK_7_1,
            Path::new("/x/i.mka"),
            Some(&ir),
        )
        .unwrap();
        assert_eq!(rate, 48_000, "the graph is built at the working rate");
        let j = crate::util::args_display(&args);
        let vol = j.find("volume=volume='sample_rate/48000':eval=once");
        let resample = j.find("aresample=48000");
        assert!(
            vol.is_some() && resample.is_some(),
            "a 96 kHz IR at a 48 kHz working rate needs both the resample and the \
             2x calibration gain: {j}"
        );
        assert!(
            vol.unwrap() < resample.unwrap(),
            "the factor reads the sample rate of the stream where it sits, so it has to \
             come before the resampler or it multiplies by 1 and looks applied: {j}"
        );
    }

    #[test]
    fn an_ir_that_is_already_level_normalised_gets_no_second_gain() {
        // Leaving `irnorm` unset affirms afir's own default of 1, which normalises the IR.
        // Applying the rate ratio on top of that is a 6 dB error, so the calibration is
        // conditional on the raw-IR mode actually being in force.
        let mut cfg = RenderCfg::with_defaults();
        cfg.ir.enable = true;
        cfg.ir.file = Some(PathBuf::from("/x/ir96.wav"));
        cfg.ir.irnorm = None;
        let mut ir = info();
        ir.sample_rate = 96_000;
        ir.channels = 1;
        ir.layout = Some("mono".into());
        let (args, _, _) = input_args(
            &cfg,
            &info(),
            crate::dsp::MASK_7_1,
            Path::new("/x/i.mka"),
            Some(&ir),
        )
        .unwrap();
        let j = crate::util::args_display(&args);
        assert!(j.contains("aresample=48000"), "{j}");
        assert!(!j.contains("volume="), "{j}");
    }

    /// A name ffmpeg reads as `[protocol:]target` never reaches the filesystem, and the
    /// loss is not the same on the two sides: on input the probe fails, on output the
    /// encoder refuses to start. Both measured on FFmpeg 9.0.2.
    #[test]
    fn names_ffmpeg_would_read_as_a_protocol_reach_it_as_a_path() {
        let cfg = RenderCfg::with_defaults();
        let (args, _, _) = input_args(
            &cfg,
            &info(),
            crate::dsp::MASK_7_1,
            Path::new("a:b.mka"),
            None,
        )
        .unwrap();
        let j = crate::util::args_display(&args);
        assert!(j.contains(" -i ./a:b.mka "), "{j}");
        assert!(
            j.ends_with(" -"),
            "the stdout marker must stay a bare dash or the decoded audio becomes a file \
             called '-' and the reader at the other end waits forever: {j}"
        );
        assert!(
            !j.contains("./- "),
            "a prefixed dash is the bug above wearing a plausible path: {j}"
        );

        let out = output_args(
            &cfg,
            48000,
            Path::new("out:v.flac"),
            Path::new("i.mka"),
            &caps_flac24(),
        )
        .unwrap();
        let jo = crate::util::args_display(&out);
        assert!(
            jo.ends_with(" ./out:v.flac"),
            "the output name is positional, so a leading colon reads as a muxer and the \
             encoder never starts: {jo}"
        );
    }

    #[test]
    fn ir_filter_string_matches_the_mpv_graph() {
        let s = afir_filter(&IrCfg::default());
        assert_eq!(
            s,
            "afir=irfmt=mono:irnorm=-1:wet=1:minp=4096:maxp=4096:precision=double"
        );
    }

    #[test]
    fn ir_graph_shape() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.ir.enable = true;
        cfg.ir.file = Some(PathBuf::from("/x/ir.wav"));
        let (args, rate, ch) = input_args(
            &cfg,
            &info(),
            crate::dsp::MASK_7_1,
            Path::new("/x/i.mka"),
            None,
        )
        .unwrap();
        assert_eq!((rate, ch), (48000, 8));
        let j = crate::util::args_display(&args);
        assert!(j.contains("-filter_complex"), "{j}");
        assert!(j.contains("[0:a:0]"), "{j}");
        assert!(
            j.contains("[1:a:0]aformat=sample_fmts=dbl:channel_layouts=mono[ir]"),
            "{j}"
        );
        assert!(j.contains("[am][ir]afir=irfmt=mono"), "{j}");
        assert!(j.contains("-map [o]"), "{j}");
    }

    #[test]
    fn resample_on_adds_soxr_once_with_dither_spelling() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.resample.enable = true;
        cfg.resample.rate = 44100;
        cfg.resample.dither = Dither::LowShibata;
        let (stage, steps) = input_stage(&cfg, &info(), crate::dsp::MASK_7_1);
        assert_eq!(stage, InputStage::Soxr);
        let joined = steps.join(",");
        assert_eq!(
            joined,
            "channelmap=channel_layout=0x63f,aformat=sample_fmts=dbl,aresample=44100:resampler=soxr:precision=33:dither_method=low_shibata"
        );
        assert_eq!(joined.matches("aresample").count(), 1);
    }

    #[test]
    fn enabling_resample_at_the_source_rate_is_reported_as_a_noop() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.resample.enable = true;
        cfg.resample.rate = 48000;
        let (stage, _) = input_stage(&cfg, &info(), crate::dsp::MASK_7_1);
        assert_eq!(stage, InputStage::Noop);
    }

    /// A depth asked for is a depth delivered, unless it provably does nothing.
    ///
    /// The condition this covers dropped `s32` for every source that was *not* already
    /// 32-bit — the one case where the step is guaranteed to do something — so a 24-bit file
    /// asked for 32-bit came back unchanged. The fixture is exactly that file: the probe
    /// reports 24-bit, and `s32` has to survive into a step. `f64` is the only request that
    /// can be proven a no-op from what the probe reports, because swresample computes in
    /// double; `bits` is a width, not a format, so nothing else may be quietly dropped.
    #[test]
    fn a_requested_depth_survives_unless_it_provably_does_nothing() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.resample.enable = true;
        cfg.resample.rate = 48000; // the source rate: depth is then the only step
        for (d, want) in [
            (BitDepth::S32, InputStage::DepthOnly),
            (BitDepth::S16, InputStage::DepthOnly),
            (BitDepth::F32, InputStage::DepthOnly),
            (BitDepth::F64, InputStage::Noop),
        ] {
            cfg.resample.depth = Some(d);
            let (stage, steps) = input_stage(&cfg, &info(), crate::dsp::MASK_7_1);
            assert_eq!(
                stage, want,
                "{d:?} asked of a 24-bit source produced {stage:?}, steps {steps:?}"
            );
        }
    }

    /// A capability set whose ffmpeg has been measured writing lossless 24-bit
    /// FLAC, i.e. the state a machine with a working ffmpeg reports.
    fn caps_flac24() -> Caps {
        Caps {
            flac24: true,
            ..Caps::default()
        }
    }

    #[test]
    fn flac_rejects_float_and_unverified_24bit() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.format = OutFormat::Flac { level: 8 };
        cfg.out.depth = BitDepth::F64;
        assert!(cfg.out.validate(&caps_flac24()).is_err());
        // The default depth is 24-bit, and it is refused on a build where the
        // probe could not prove 24-bit FLAC is lossless — a file labelled 24-bit
        // that quietly dropped bits is worse than an error.
        cfg.out.depth = BitDepth::S24;
        assert!(cfg.out.validate(&Caps::default()).is_err());
        cfg.out.validate(&caps_flac24()).unwrap();
    }

    #[test]
    fn flac_refuses_32_bit_instead_of_writing_a_24_bit_file() {
        // ffmpeg labels what it writes bps=24 and throws the low byte of every
        // 32-bit sample away, so the combination has to be refused by name.
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.format = OutFormat::Flac { level: 8 };
        cfg.out.depth = BitDepth::S32;
        let e = cfg.out.validate(&caps_flac24()).unwrap_err().to_string();
        assert!(e.contains("32-bit FLAC"), "{e}");
        assert!(e.contains("--depth 24"), "{e}");
    }

    #[test]
    fn wav_allows_all_depths() {
        for d in BitDepth::ALL {
            let mut cfg = RenderCfg::with_defaults();
            cfg.out.format = OutFormat::Wav;
            cfg.out.depth = d;
            cfg.out.validate(&Caps::default()).unwrap();
        }
    }

    #[test]
    fn output_resample_uses_soxr_with_precision() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.resample = Some(96000);
        let args = output_args(
            &cfg,
            48000,
            Path::new("/tmp/o.flac"),
            Path::new("/tmp/i.mka"),
            &caps_flac24(),
        )
        .unwrap();
        let j = crate::util::args_display(&args);
        assert!(
            j.contains("aresample=96000:resampler=soxr:precision=33:dither_method=none"),
            "{j}"
        );
    }

    #[test]
    fn pump_declares_rate_and_layout_for_raw_stdin() {
        let j = encoder_pump_args(48000).join(" ");
        assert!(j.contains("-ar 48000"), "{j}");
        assert!(j.contains("-ch_layout stereo"), "{j}");
        assert!(j.contains("pipe:0"), "{j}");
    }

    #[test]
    fn loss_chain_says_off_when_off() {
        // "Off when off" needs a config with the limiter off, which is no longer what the
        // defaults are. What the defaults say is checked just below, so neither the off
        // text nor the on text can drift unnoticed.
        let mut cfg = RenderCfg::with_defaults();
        cfg.limiter.enable = false;
        let lines = loss_chain(&cfg, &info(), crate::dsp::MASK_7_1);
        assert_eq!(lines.len(), 6);
        assert!(lines[1].contains("off"), "{}", lines[1]);
        assert!(lines[2].contains("off"), "{}", lines[2]);
        assert!(lines[3].contains("bit-exact port"), "{}", lines[3]);
        assert!(
            lines[3].contains("FL FR FC LFE BL BR SL SR"),
            "{}",
            lines[3]
        );
        assert!(lines[4].contains("limiter off"), "{}", lines[4]);
        assert!(lines[4].contains("meter at 4x"), "{}", lines[4]);
        assert!(lines[5].contains("FLAC"), "{}", lines[5]);
    }

    /// The default render limits, and the chain has to say so rather than let a limited
    /// file look like a port. There is no "silent by default" in this table.
    #[test]
    fn loss_chain_names_the_default_limiter() {
        let lines = loss_chain(&RenderCfg::with_defaults(), &info(), crate::dsp::MASK_7_1);
        assert_eq!(lines.len(), 6);
        let tp = &lines[4];
        assert!(tp.contains("limit to -3.00 dBTP"), "{tp}");
        assert!(tp.contains("at 8x"), "{tp}");
        assert!(tp.contains("lookahead 2.00 ms"), "{tp}");
        assert!(tp.contains("(4x meter)"), "{tp}");
        assert!(!tp.contains("limiter off"), "{tp}");
    }

    #[test]
    fn order_for_mask_roundtrips() {
        use crate::mask::mask_of;
        assert_eq!(
            order_for_mask(mask_of(&[
                Slot::Fl,
                Slot::Fr,
                Slot::Fc,
                Slot::Lfe,
                Slot::Bl,
                Slot::Br,
                Slot::Sl,
                Slot::Sr
            ])),
            vec![
                Slot::Fl,
                Slot::Fr,
                Slot::Fc,
                Slot::Lfe,
                Slot::Bl,
                Slot::Br,
                Slot::Sl,
                Slot::Sr
            ]
        );
    }
}
