//! `g6me` — the **G6 Media Encoder** command line: batch-render 7.1/5.1 (and
//! Atmos) audio to binaural stereo, or monitor it live on an output device.
//!
//! Everything that touches the samples is opt-in and printed: the default path is
//! `decode → f64 → downmix → encode` with no resampler, no dither, no convolution.
//! `--dry-run` prints the exact ffmpeg command lines and the plain-language loss
//! chain before a single sample moves.

use clap::Parser;
use g6me_core::batch::{self, Batch, BatchCfg, Item, Phase, Progress};
use g6me_core::limiter::{
    Cfg as LimiterCfg, DEFAULT_ATTACK_MS, DEFAULT_CEILING_DB, DEFAULT_LOOKAHEAD_MS, DEFAULT_RATIO,
    DEFAULT_RELEASE_MS, MAX_ATTACK_MS, MAX_CEILING_DB, MAX_LOOKAHEAD_MS, MAX_RELEASE_MS,
    MIN_ATTACK_MS, MIN_CEILING_DB, MIN_LOOKAHEAD_MS, MIN_RELEASE_MS,
};
use g6me_core::mask::Slot;
#[cfg(feature = "monitor")]
use g6me_core::meter::Meters;
use g6me_core::pcm::{BitDepth, Dither};
#[cfg(feature = "monitor")]
use g6me_core::pipeline::RenderOpts;
use g6me_core::pipeline::{self, Plan};
use g6me_core::probe::{probe_caps, Caps, Tools};
use g6me_core::stages::{Gtype, IrCfg, OutFormat, RenderCfg};
use g6me_core::tp::Ratio;
use g6me_core::util;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// G6 Media Encoder — downmix 7.1 / 5.1 (and Atmos) to binaural stereo with the
/// G6 matrix curve, losslessly, or monitor it live.
#[derive(Parser, Debug, Clone)]
#[command(name = "g6me", version, about, long_about = None)]
pub struct Cli {
    /// Input files and/or folders. Folders are scanned recursively.
    #[arg(required_unless_present_any = ["caps", "list_devices"])]
    pub input: Vec<PathBuf>,

    /// Write outputs under this directory, mirroring the input tree.
    /// Without it, each output is written next to its input.
    #[arg(short = 'o', long)]
    pub out_dir: Option<PathBuf>,

    /// Output container.
    #[arg(short = 'f', long, default_value = "flac", value_parser = ["flac", "wav"])]
    pub format: String,

    /// FLAC compression level, 0 (fastest) .. 12 (smallest), as ffmpeg's encoder
    /// counts them. Lossless at every level; 13 is refused by the encoder.
    #[arg(long, default_value_t = 8, value_parser = parse_level)]
    pub level: u32,

    /// Output bit depth: 16, 24, 32, f32, f64. FLAC takes 16 or 24 only — ffmpeg's
    /// FLAC encoder writes at most 24-bit, so 32-bit and float need --format wav.
    #[arg(long, default_value = "s24", value_parser = parse_depth)]
    pub depth: BitDepth,

    /// Over-sampling ratio the true peak is measured at: 2, 4, 8, 16 or 32 ("x4" works
    /// too). Every render reports the result, quoted with this ratio, because a dBTP
    /// without one is not a comparable number.
    #[arg(long = "true-peak", default_value = "4", value_parser = parse_ratio)]
    pub true_peak: Ratio,

    /// Setpoint: one number, two effects. The file is trimmed by this many dB and nothing
    /// in it is left above this many dBTP, so -4 leaves the file 4 dB quieter with a -4
    /// dBTP ceiling. The trim is a constant gain on every sample; the limiter is then left
    /// with only the peaks that survive it, which is the cheap way to hold a ceiling.
    /// Conflicts with --gain and --ceiling, which set the two halves separately.
    #[arg(long, allow_negative_numbers = true, value_parser = parse_ceiling_db)]
    pub setpoint: Option<f64>,

    /// Gain staging alone: a constant gain on every sample, in dB (-24 ..= 0), before the
    /// limiter.
    #[arg(long, allow_negative_numbers = true, default_value_t = 0.0, value_parser = parse_gain_db)]
    pub gain: f64,

    /// Leave the limiter off: a true-peak meter reading and nothing else. The default is
    /// on, so this is the opt-out rather than the opt-in, and it is the only way to get a
    /// render that adds neither latency nor level management to the source.
    #[arg(long)]
    pub no_limit: bool,
    /// Remove DC before the gain trim and the limiter, with a 5 Hz first-order blocker
    /// (exactly zero at DC, exactly unity at Nyquist).
    ///
    /// Off by default: the downmixer's own contribution is a settled +4e-10 on the left
    /// and cannot be reached without breaking bit-exactness with the reference. Ask for
    /// this when the *source* carries DC, or an impulse response does, since the 7.1 matrix
    /// carries an input offset to +5.18 on the left, where the limiter reads it as level.
    #[arg(long)]
    pub dc_block: bool,

    /// Limit the true peak. On by default now; kept so an existing command line still
    /// means what it says.
    #[arg(long)]
    pub limit: bool,

    /// True-peak ceiling, in dBTP (-24 ..= 0). Defaults to -3.
    // `allow_negative_numbers` because `--ceiling -1` is how it is typed, and without it
    // clap reads the -1 as a flag and refuses the whole command line.
    #[arg(long, allow_negative_numbers = true, value_parser = parse_ceiling_db)]
    pub ceiling: Option<f64>,

    /// Lookahead the transient stage may use, in ms (0.2 ..= 2.0). A deeper filter needs
    /// more; what was actually used is reported, because a deep filter cannot honour a
    /// shallow ask and the report says so instead of showing the number you typed.
    #[arg(long, default_value_t = DEFAULT_LOOKAHEAD_MS, value_parser = parse_lookahead_ms)]
    pub lookahead: f64,

    /// How fast the slow stage's gain reduction arrives, in ms (1 ..= 200).
    #[arg(long, default_value_t = DEFAULT_ATTACK_MS, value_parser = parse_attack_ms)]
    pub attack: f64,

    /// How fast the slow stage lets go, in ms (10 ..= 5000).
    #[arg(long, default_value_t = DEFAULT_RELEASE_MS, value_parser = parse_release_ms)]
    pub release: f64,

    /// Channel linking of the transient stage, percent (0 ..= 100). At 100 a peak in one
    /// channel ducks both; turned down, a short peak is removed only where it is.
    #[arg(long, default_value_t = 100.0, value_parser = parse_link_pct)]
    pub link_transient: f64,

    /// Channel linking of the slow stage, percent (0 ..= 100). Worth turning down before
    /// the transient one: the slow stage moves level over tenths of a second, and a shared
    /// movement of that length is what pulls a stereo image about.
    #[arg(long, default_value_t = 75.0, value_parser = parse_link_pct)]
    pub link_release: f64,

    /// Over-sampling ratio the limiter runs at. Defaults to 8, which is deeper than the
    /// meter's 4 on purpose: the printed ITU bank reads up to 0.6 dB low on a crest
    /// between grid points, and a ceiling cannot be held with a meter that low.
    #[arg(long, value_parser = parse_ratio)]
    pub limit_ratio: Option<Ratio>,

    /// libsoxr precision in bits, 17..=33, used whenever a bit depth is reduced.
    #[arg(long, default_value_t = 33.0, value_parser = parse_precision_bits)]
    pub precision: f64,

    /// Dither applied when the output depth is lower than the working depth.
    /// `none` means truncation-free straight rounding — no noise shaping at all.
    #[arg(long, default_value = "none", value_parser = parse_dither)]
    pub dither: Dither,

    /// Do not copy source tags into the output.
    #[arg(long)]
    pub no_metadata: bool,

    /// Overwrite existing outputs (still never the input file).
    #[arg(short = 'y', long)]
    pub force: bool,

    /// Resample to this rate *before* the downmix, with libsoxr.
    /// Off unless given: the default path is bit-exact passthrough.
    #[arg(short = 'r', long, value_parser = parse_rate)]
    pub rate: Option<u32>,

    /// Dither used by the resampler itself (its internal depth reduction).
    #[arg(long, default_value = "none", value_parser = parse_dither)]
    pub resample_dither: Dither,

    /// Quantize to this format at the resampler output (input side).
    #[arg(long, value_parser = parse_resample_depth)]
    pub resample_depth: Option<BitDepth>,

    /// Ask soxr for the Chebyshev-windowed filter (sharper, more ripple).
    #[arg(long)]
    pub cheby: bool,

    /// Headphone calibration impulse to convolve into every input channel first,
    /// the way mpv's `afir` does it (mono IR, fp64, applied before the downmix).
    #[arg(long)]
    pub ir: Option<PathBuf>,

    /// IR gain normalisation: -1 = none (mpv's default), 0 = peak, 1 = RMS.
    #[arg(long, value_parser = parse_ir_norm)]
    pub ir_norm: Option<f64>,

    /// Wet/dry mix of the convolution.
    #[arg(long)]
    pub ir_wet: Option<f64>,
    #[arg(long)]
    pub ir_dry: Option<f64>,

    /// Convolution length as a fraction of the IR: 0.5 convolves half of it. 0 or 1
    /// means the whole IR, which is also the default.
    ///
    /// This is `afir`'s own unit, and it is a fraction, not a duration.
    // `allow_negative_numbers` for the same reason `--ceiling` has it: without it clap
    // reads a leading minus as a flag and refuses the command line before the value
    // parser can say what is actually wrong with the number.
    #[arg(long, allow_negative_numbers = true, value_parser = parse_ir_length)]
    pub ir_length: Option<f64>,

    /// IR phase: `none` keeps linear phase, `min`/`minimum` folds it to a
    /// minimum-phase response (mpv leaves this alone).
    #[arg(long, value_parser = parse_gtype)]
    pub ir_gtype: Option<Gtype>,

    /// FFT partition size bounds for the convolver.
    #[arg(long, value_parser = parse_part)]
    pub ir_minp: Option<u32>,
    #[arg(long, value_parser = parse_part)]
    pub ir_maxp: Option<u32>,

    /// Force the interleaved channel order, e.g. FL,FR,FC,LFE,BL,BR,SL,SR.
    /// Only needed for a container that mislabels its layout.
    #[arg(long, value_parser = parse_order)]
    pub order: Option<Order>,

    /// Which audio stream to take, 0-based.
    #[arg(long, default_value_t = 0)]
    pub stream: usize,

    /// Files to render at once, 1 .. number of logical CPUs.
    /// DSP stays single-threaded per file; the parallelism is between files.
    #[arg(short = 'j', long, value_parser = parse_threads)]
    pub threads: Option<usize>,

    /// Frames per processing block (the downmixer's stride).
    #[arg(long, default_value_t = 4096, value_parser = parse_block)]
    pub block: usize,

    /// Blocks allowed in flight between the decode, DSP and encode stages.
    /// This is the memory bound: queue * block * channels * 8 bytes per file.
    #[arg(long, default_value_t = 16, value_parser = parse_queue)]
    pub queue: usize,

    /// SHA-256 every finished output: the sample stream (container-independent, so two
    /// runs of different formats agree) and the file itself (`sha256sum` reproduces this
    /// one). Both appear in --json as `sha256` and `file_sha256`.
    #[arg(long)]
    pub sha256: bool,

    /// Print the planned chain and the exact ffmpeg command lines, then stop.
    #[arg(long)]
    pub dry_run: bool,

    /// One JSON object per finished file, plus a summary.
    #[arg(long)]
    pub json: bool,

    /// No progress output, just errors and the final line.
    #[arg(short = 'q', long)]
    pub quiet: bool,

    /// Play the result to an output device instead of writing a file.
    #[arg(long)]
    pub monitor: bool,

    /// Device to monitor through: an id, or a case-insensitive substring of its
    /// name. Empty means the default device.
    #[arg(long, default_value = "")]
    pub device: String,

    /// Print what this ffmpeg build can actually do, then exit.
    #[arg(long)]
    pub caps: bool,

    /// List output devices, then exit.
    #[arg(long)]
    pub list_devices: bool,
}

fn parse_depth(s: &str) -> Result<BitDepth, String> {
    match s.to_ascii_lowercase().as_str() {
        "16" | "s16" | "int16" => Ok(BitDepth::S16),
        "24" | "s24" | "int24" => Ok(BitDepth::S24),
        "32" | "s32" | "int32" => Ok(BitDepth::S32),
        "32f" | "f32" | "float" => Ok(BitDepth::F32),
        "64f" | "f64" | "double" => Ok(BitDepth::F64),
        other => Err(format!(
            "unknown bit depth '{other}' (16, 24, 32, f32, f64)"
        )),
    }
}

/// Input-side depths are narrower: libswresample has no `s24` sample format, so
/// 24-bit only exists at the encoder. Saying so here beats failing inside ffmpeg.
fn parse_resample_depth(s: &str) -> Result<BitDepth, String> {
    let d = parse_depth(s)?;
    if d == BitDepth::S24 {
        return Err(
            "24-bit is not a libswresample format; use --depth 24 for the FLAC output instead"
                .into(),
        );
    }
    Ok(d)
}

fn parse_dither(s: &str) -> Result<Dither, String> {
    let k = s.to_ascii_lowercase().replace(['-', ' '], "_");
    Ok(match k.as_str() {
        "none" | "off" => Dither::None,
        "rect" | "rectangular" => Dither::Rectangular,
        "tri" | "triangular" => Dither::Triangular,
        "tri_hp" | "triangular_hp" | "hp" => Dither::TriangularHp,
        "lipshitz" => Dither::Lipshitz,
        "shibata" => Dither::Shibata,
        "low_shibata" | "lowshibata" => Dither::LowShibata,
        "high_shibata" | "highshibata" => Dither::HighShibata,
        "f_weighted" | "fweighted" => Dither::FWeighted,
        "modified_e_weighted" => Dither::ModifiedEWeighted,
        "improved_e_weighted" => Dither::ImprovedEWeighted,
        other => {
            return Err(format!(
                "unknown dither '{other}' (none, rectangular, triangular, triangular_hp, \
                 lipshitz, shibata, low_shibata, high_shibata, f_weighted, \
                 modified_e_weighted, improved_e_weighted)"
            ))
        }
    })
}

fn parse_gtype(s: &str) -> Result<Gtype, String> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "none" | "off" => Gtype::None,
        "peak" => Gtype::Peak,
        "dc" => Gtype::Dc,
        "gn" => Gtype::Gn,
        "ac" => Gtype::Ac,
        "rms" => Gtype::Rms,
        other => {
            return Err(format!(
                "unknown IR phase '{other}' (none, peak, dc, gn, ac, rms)"
            ))
        }
    })
}

/// One `--order` value. A newtype, because clap reads `Option<Vec<T>>` as "repeatable
/// option, one `T` per value" and then panics when the parser hands back a whole `Vec`.
#[derive(Clone, Debug, PartialEq)]
pub struct Order(pub Vec<Slot>);

fn parse_order(s: &str) -> Result<Order, String> {
    let mut out = Vec::new();
    for part in s.split([',', ' ', '+']) {
        if part.trim().is_empty() {
            continue;
        }
        out.push(Slot::parse(part.trim()).ok_or_else(|| {
            format!(
                "unknown channel '{part}' (known: {})",
                Slot::ALL
                    .iter()
                    .map(|s| s.name())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        })?);
    }
    if out.is_empty() {
        return Err("--order needs at least one channel".into());
    }
    Ok(Order(out))
}

/// A bounded integer argument with a message that says which bound was missed.
/// (Clap's own ranged parser downcasts by type, and a `u32` field fed an `i64`
/// range panics at parse time rather than complaining.)
macro_rules! int_arg {
    ($name:ident, $ty:ty, $lo:expr, $hi:expr, $what:literal) => {
        fn $name(s: &str) -> Result<$ty, String> {
            let v: i64 = s
                .trim()
                .replace('_', "")
                .parse()
                .map_err(|_| format!("'{s}' is not a number ({})", $what))?;
            if !($lo..=$hi).contains(&v) {
                return Err(format!("{} must be {}..={}, got {v}", $what, $lo, $hi));
            }
            Ok(v as $ty)
        }
    };
}

int_arg!(parse_level, u32, 0, 12, "FLAC compression level");
int_arg!(parse_rate, u32, 8_000, 384_000, "sample rate (--rate)");
int_arg!(parse_part, u32, 256, 65_536, "convolver FFT partition size");
int_arg!(parse_block, usize, 64, 1_048_576, "block size (--block)");
int_arg!(parse_queue, usize, 1, 1_024, "queue depth (--queue)");

/// A dB or millisecond value on a stated range, with the range in the message.
fn parse_span(s: &str, what: &str, lo: f64, hi: f64) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("'{s}' is not a {what}"))?;
    if !v.is_finite() || !(lo..=hi).contains(&v) {
        return Err(format!("{what} must be {lo}..={hi}, got {v}"));
    }
    Ok(v)
}

fn parse_attack_ms(s: &str) -> Result<f64, String> {
    parse_span(s, "attack time in ms", MIN_ATTACK_MS, MAX_ATTACK_MS)
}

fn parse_release_ms(s: &str) -> Result<f64, String> {
    parse_span(s, "release time in ms", MIN_RELEASE_MS, MAX_RELEASE_MS)
}

/// Linking as a percentage, because that is what a limiter panel shows: `0.5` is half a
/// percent, not half.
fn parse_link_pct(s: &str) -> Result<f64, String> {
    parse_span(s, "channel linking in percent (0..=100)", 0.0, 100.0)
}

/// Same range as the ceiling, which it is usually set against, but its own message.
fn parse_gain_db(s: &str) -> Result<f64, String> {
    parse_span(s, "gain in dB", MIN_CEILING_DB, MAX_CEILING_DB)
}

fn parse_precision_bits(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a number of bits"))?;
    if !(17.0..=33.0).contains(&v) {
        return Err(format!("soxr precision must be 17..=33 bits, got {v}"));
    }
    Ok(v)
}

fn parse_ir_norm(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("'{s}' is not a number"))?;
    if !(-1.0..=1.0).contains(&v) {
        return Err(format!(
            "--ir-norm must be -1 (none), 0 (peak) or 1 (RMS), got {v}"
        ));
    }
    Ok(v)
}

/// `afir`'s `length` is a fraction of the IR, not a number of seconds.
///
/// Measured on FFmpeg 9.0.2, where the option is declared `set IR length (from 0 to 1)
/// (default 1)`: `length=0.5` runs, and `length=1.5` stops the whole filter graph with
/// "Value 1.500000 for parameter 'length' out of range [0 - 1]" and exit 222. Left alone,
/// that failure arrives per file, after the queue has started, phrased as a filter-graph
/// error that does not name the flag that caused it.
fn parse_ir_length(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("'{s}' is not a number"))?;
    if !(0.0..=1.0).contains(&v) {
        return Err(format!(
            "--ir-length is a fraction of the IR (1 or 0 = the whole IR, 0.5 = half of it), \
             not a duration in seconds; got {v}"
        ));
    }
    Ok(v)
}

fn parse_ratio(s: &str) -> Result<Ratio, String> {
    s.parse()
}

fn parse_ceiling_db(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a dBTP ceiling"))?;
    if !v.is_finite() || v < MIN_CEILING_DB || v > MAX_CEILING_DB {
        return Err(format!(
            "ceiling must be between {MIN_CEILING_DB} and {MAX_CEILING_DB} dBTP, got {v}"
        ));
    }
    Ok(v)
}

fn parse_lookahead_ms(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a lookahead in ms"))?;
    if !v.is_finite() || v < MIN_LOOKAHEAD_MS || v > MAX_LOOKAHEAD_MS {
        return Err(format!(
            "lookahead must be between {MIN_LOOKAHEAD_MS} and {MAX_LOOKAHEAD_MS} ms, got {v}"
        ));
    }
    Ok(v)
}

/// Threads are capped at the logical CPU count, the same ceiling the GUI slider
/// uses. More workers than cores only adds contention, and each worker already
/// runs three threads plus two ffmpeg children.
fn parse_threads(s: &str) -> Result<usize, String> {
    let n: usize = s
        .parse()
        .map_err(|_| format!("'{s}' is not a number of threads"))?;
    if n == 0 {
        return Err("at least one thread is required".into());
    }
    let max = util::max_threads();
    Ok(if n > max {
        eprintln!("note: --threads {n} exceeds the {max} logical CPUs; using {max}");
        max
    } else {
        n
    })
}

impl Cli {
    /// Translate the command line into the render configuration.
    pub fn render_cfg(&self) -> Result<RenderCfg, String> {
        // One number meaning two things cannot also be argued with. `--setpoint -4` says
        // "trim by 4 and cap at 4"; if either half is also spelled out, the command line
        // has two opinions about the same file, so it fails rather than picking one.
        if let Some(sp) = self.setpoint {
            if self.ceiling.is_some() || self.gain != 0.0 {
                return Err(format!(
                    "--setpoint {sp:+.2} already sets both the gain and the ceiling; \
                     drop it, or drop --gain/--ceiling and set the two halves separately"
                ));
            }
        }
        if self.no_limit && (self.limit || self.setpoint.is_some()) {
            return Err(
                "--no-limit cannot be combined with --limit or --setpoint, which both ask for a limiter".into(),
            );
        }
        let format = match self.format.as_str() {
            "wav" => OutFormat::Wav,
            _ => OutFormat::Flac { level: self.level },
        };
        // Start from the mpv-equivalent defaults and overwrite only what was
        // given. `irnorm=-1` in particular must survive: afir's own default is
        // peak normalisation, which would silently change the IR's level and so
        // the whole calibration.
        let ir = IrCfg {
            enable: self.ir.is_some(),
            file: self.ir.clone(),
            irnorm: self.ir_norm.or_else(|| IrCfg::default().irnorm),
            gtype: self.ir_gtype,
            wet: self.ir_wet.or_else(|| IrCfg::default().wet),
            dry: self.ir_dry,
            // 0 is the documented way to ask for the whole IR. afir's default is already
            // the whole IR, so the honest implementation of that promise is to leave the
            // option out: passing length=0 would set the fraction to zero and convolve
            // nothing, which is the opposite of what the flag says.
            length: self.ir_length.filter(|v| *v > 0.0),
            link: None,
            minp: self.ir_minp.unwrap_or(4096),
            maxp: self.ir_maxp.unwrap_or(4096),
        };
        let mut cfg = RenderCfg {
            resample: g6me_core::stages::ResampleCfg {
                enable: self.rate.is_some(),
                rate: self.rate.unwrap_or(48000),
                depth: self.resample_depth,
                precision: self.precision,
                dither: self.resample_dither,
                cheby: self.cheby,
            },
            ir,
            out: g6me_core::stages::OutCfg {
                format,
                depth: self.depth,
                // Rate changes belong before the downmix: the filter coefficients
                // are a property of the hardware being modelled, so resampling the
                // stereo result afterwards would be a different (wrong) curve.
                resample: None,
                precision: self.precision,
                dither: self.dither,
                copy_metadata: !self.no_metadata,
                overwrite: self.force,
                filter_threads: None,
            },
            order: self.order.as_ref().map(|o| o.0.clone()),
            output_gain_db: self.gain,
            dc_block: self.dc_block,
            true_peak: self.true_peak,
            limiter: LimiterCfg {
                enable: !self.no_limit,
                ceiling_db: self.ceiling.unwrap_or(DEFAULT_CEILING_DB),
                // The limiter's own default, not `Ratio::default()`: that one is the
                // meter's x4, and the help text below promises 8.
                ratio: self.limit_ratio.unwrap_or(DEFAULT_RATIO),
                lookahead_ms: self.lookahead,
                attack_ms: self.attack,
                release_ms: self.release,
                link_transient: self.link_transient / 100.0,
                link_release: self.link_release / 100.0,
            },
            block: self.block,
            queue_blocks: self.queue,
            stream: self.stream,
            sha256: self.sha256,
        };
        // Both halves at once, through the one function that defines what a setpoint is.
        if let Some(sp) = self.setpoint {
            cfg.apply_setpoint(sp);
        }
        Ok(cfg)
    }

    pub fn threads(&self) -> usize {
        util::clamp_threads(self.threads.unwrap_or_else(batch::default_workers))
    }

    fn batch_cfg(&self) -> Result<BatchCfg, String> {
        Ok(BatchCfg {
            inputs: self.input.clone(),
            out_dir: self.out_dir.clone(),
            recursive: true,
            workers: self.threads(),
            render: self.render_cfg()?,
            fail_on_empty: true,
            // The CLI has nothing to draw into, so it renders blind: no taps, and so
            // no meter allocation on the render path either.
            taps: Default::default(),
        })
    }
}

/// Per-file terminal output.
struct Ui {
    quiet: bool,
    json: bool,
    /// A terminal we can redraw a status line in.
    tty: bool,
    started: Instant,
    last_draw: Mutex<Instant>,
}

impl Ui {
    fn new(quiet: bool, json: bool) -> Self {
        Self {
            quiet: quiet || json,
            json,
            tty: std::io::IsTerminal::is_terminal(&std::io::stderr()),
            started: Instant::now(),
            last_draw: Mutex::new(Instant::now() - Duration::from_secs(10)),
        }
    }

    fn line(&self, s: &str) {
        if !self.quiet {
            eprintln!("{s}");
        }
    }

    /// Redraw the batch status at most ~5 times a second.
    fn status(&self, c: g6me_core::batch::CountersView, cancelled: bool) {
        // `cancelled` is shown, not just polled: a Ctrl-C that appears to do
        // nothing looks like a hang.
        if self.quiet || !self.tty {
            return;
        }
        {
            let mut last = self.last_draw.lock().unwrap_or_else(|e| e.into_inner());
            if last.elapsed() < Duration::from_millis(200) {
                return;
            }
            *last = Instant::now();
        }
        let per_item = if c.done + c.failed > 0 {
            c.seconds_done / (c.done + c.failed) as f64
        } else {
            0.0
        };
        let eta = c
            .eta(self.started.elapsed(), per_item)
            .map(|d| util::human_time(d.as_secs_f64()))
            .unwrap_or_else(|| "--:--".into());
        let bar = bar(c.fraction(), 24);
        let stopping = if cancelled { "  CANCELLING" } else { "" };
        eprint!(
            "\r{bar} {}/{} done:{} fail:{} skip:{} run:{}  eta {eta}  {}{stopping}     ",
            c.done + c.failed + c.skipped,
            c.total,
            c.done,
            c.failed,
            c.skipped,
            c.running,
            util::human_bytes(c.bytes),
        );
    }

    fn clear_status(&self) {
        if self.tty && !self.quiet {
            eprint!("\r{}\r", " ".repeat(100));
        }
    }

    fn item(&self, item: &Item) {
        if self.json {
            let v = serde_json::json!({
                "src": item.src,
                "dst": item.dst,
                "phase": item.phase.label(),
                "error": item.error,
                "report": item.report,
            });
            println!("{v}");
            return;
        }
        if self.quiet {
            return;
        }
        match item.phase {
            Phase::Done => {
                if let Some(r) = &item.report {
                    eprintln!("{}", r.one_line());
                    for w in &r.warnings {
                        eprintln!("  warning: {w}");
                    }
                }
            }
            Phase::Failed => eprintln!(
                "FAILED {}  {}",
                item.src.display(),
                item.error.clone().unwrap_or_default()
            ),
            Phase::Skipped => eprintln!(
                "skipped {}  (output exists; use --force)",
                item.src.display()
            ),
            _ => {}
        }
    }
}

fn bar(frac: f32, width: usize) -> String {
    let filled = ((frac.clamp(0.0, 1.0)) * width as f32).round() as usize;
    format!("[{}{}]", "#".repeat(filled), "-".repeat(width - filled))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let tools = Tools::discover();

    if cli.caps {
        return match caps_report(&tools, cli.json) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => fail(&e.to_string()),
        };
    }
    if cli.list_devices {
        return device_list_flag();
    }
    if !tools.is_usable() {
        return fail(&format!(
            "ffmpeg/ffprobe not found ({} / {}).\n\
             Install ffmpeg with libsoxr, or point G6ME_FFMPEG / G6ME_FFPROBE at them.",
            tools.ffmpeg.display(),
            tools.ffprobe.display()
        ));
    }

    let caps = probe_caps(&tools);
    // Reject an impossible format before discovering a tree and spawning children.
    // `--monitor` writes no file, so its output format is not validated.
    if !cli.monitor {
        if let Err(e) = cli
            .render_cfg()
            .and_then(|c| c.out.validate(&caps).map_err(|e| e.to_string()))
        {
            return fail(&e);
        }
    }
    if cli.monitor {
        return monitor_files(&cli, &tools, &caps);
    }
    match cli.batch_cfg() {
        Ok(cfg) => render_batch(&cli, cfg, &tools, &caps),
        Err(e) => fail(&e),
    }
}

fn fail(msg: &str) -> ExitCode {
    eprintln!("error: {msg}");
    ExitCode::from(2)
}

#[cfg(feature = "monitor")]
fn device_list_flag() -> ExitCode {
    list_devices()
}

#[cfg(not(feature = "monitor"))]
fn device_list_flag() -> ExitCode {
    fail("this build has no monitoring support (feature `monitor` is off)")
}

/// The ordinary path: discover, then render the queue (or describe it).
fn render_batch(cli: &Cli, cfg: BatchCfg, tools: &Tools, caps: &Caps) -> ExitCode {
    let ext = cfg.render.out.format.extension().to_string();
    let items = batch::discover(&cfg.inputs, cfg.out_dir.as_deref(), &ext, cfg.recursive);
    if items.is_empty() {
        return fail(&format!(
            "no audio files found in {}",
            cfg.inputs
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if cli.dry_run {
        return dry_run(cli, &items, &cfg.render, tools, caps);
    }
    warn_moved(&items);

    let ui = Arc::new(Ui::new(cli.quiet, cli.json));

    eprintln!(
        "{} file(s), {} worker(s), {}",
        items.len(),
        cfg.workers,
        caps.summary()
    );
    ui.line(&format!("loss chain: {}", cfg.render.loss_summary()));

    let ui2 = ui.clone();
    let progress: Progress = Arc::new(move |_i: usize, item: &Item| ui2.item(item));
    let started = Instant::now();
    let batch = Batch::start(items, &cfg, tools.clone(), caps.clone(), Some(progress));
    // The batch's own flag: the workers and every render poll this one. A separate flag
    // that only the status line read is how Ctrl-C used to print "cancelling" and then
    // render the rest of the queue.
    let cancel = batch.cancel.clone();
    {
        let cancel = cancel.clone();
        let _ = ctrlc::set_handler(move || {
            if !cancel.swap(true, Ordering::SeqCst) {
                eprintln!("\ncancelling — finishing the current block, then cleaning up");
            }
        });
    }
    // Watch the batch's own counters through Arc clones: `finish` consumes the
    // batch, so the watcher must not borrow it. It keeps drawing while a cancel
    // drains, and stops when `finish` has returned.
    let counters = batch.counters.clone();
    let finished = Arc::new(AtomicBool::new(false));
    let finished_watch = finished.clone();
    let ui_watch = ui.clone();
    let watcher = std::thread::spawn(move || loop {
        let c = counters.snapshot();
        if c.remaining() == 0 || finished_watch.load(Ordering::Relaxed) {
            break;
        }
        ui_watch.status(c, cancel.load(Ordering::Relaxed));
        std::thread::sleep(Duration::from_millis(120));
    });
    let outcome = batch.finish();
    finished.store(true, Ordering::Relaxed);
    let _ = watcher.join();
    ui.clear_status();

    print_summary(cli, &cfg, &outcome, started.elapsed());
    ExitCode::from(outcome.exit_code().clamp(0, 255) as u8)
}

fn print_summary(cli: &Cli, cfg: &BatchCfg, outcome: &batch::Outcome, elapsed: Duration) {
    if cli.json {
        let v = serde_json::json!({
            "done": outcome.done,
            "failed": outcome.failed,
            "skipped": outcome.skipped,
            "cancelled": outcome.cancelled,
            "seconds": elapsed.as_secs_f64(),
            "workers": cfg.workers,
            "chain": cfg.render.loss_summary(),
        });
        println!("{v}");
    } else {
        eprintln!(
            "{} in {}",
            outcome.one_line(),
            util::human_time(elapsed.as_secs_f64())
        );
        if outcome.cancelled {
            eprintln!("cancelled.");
        }
    }
}

/// Print exactly what would happen to each file, including the real command lines.
fn dry_run(cli: &Cli, items: &[Item], cfg: &RenderCfg, tools: &Tools, caps: &Caps) -> ExitCode {
    let mut items = items.to_vec();
    if cfg.out.overwrite {
        batch::refuse_overwriting_inputs(&mut items);
    }
    let mut bad = 0usize;
    for (i, item) in items.iter().enumerate() {
        println!(
            "# {}/{}  {} -> {}",
            i + 1,
            items.len(),
            item.src.display(),
            item.dst.display()
        );
        // Said where the plan is printed, because this is the moment the user is reading
        // the destinations and deciding whether to run it.
        if let Some(why) = &item.note {
            println!("#   {why}");
        }
        if let Some(e) = item.error.as_ref().filter(|_| item.phase == Phase::Failed) {
            bad += 1;
            println!("  would FAIL: {e}");
            continue;
        }
        match pipeline::plan_file(&item.src, &item.dst, cfg, tools, caps) {
            Ok(plan) => {
                println_plan(&plan, cli.json);
            }
            Err(e) => {
                bad += 1;
                println!("  would FAIL: {e}");
            }
        }
    }
    if cli.json {
        return ExitCode::SUCCESS;
    }
    println!(
        "# {} file(s), {} worker(s), {}",
        items.len(),
        cli.threads(),
        caps.summary()
    );
    if bad > 0 {
        eprintln!("{bad} of {} file(s) would fail", items.len());
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Say out loud, before any work, when files cannot use the output path their names imply.
///
/// A batch that quietly renames is a batch that loses files: the colliding write gets
/// skipped as already rendered, nothing fails, and the output folder is short by however
/// many collided. Going to stderr keeps `--json` on stdout machine-clean.
fn warn_moved(items: &[Item]) {
    let moved: Vec<&Item> = items.iter().filter(|i| i.note.is_some()).collect();
    if moved.is_empty() {
        return;
    }
    eprintln!(
        "{} file(s) cannot use the output path their name implies:",
        moved.len()
    );
    for it in moved.iter().take(8) {
        eprintln!(
            "  {} -> {}  ({})",
            it.src.display(),
            it.dst.display(),
            it.note.clone().unwrap_or_default()
        );
    }
    if moved.len() > 8 {
        eprintln!("  ... and {} more", moved.len() - 8);
    }
}

fn println_plan(plan: &Plan, json: bool) {
    if json {
        println!(
            "{}",
            serde_json::json!({
                "source": plan.info,
                "order": plan.order.iter().map(|s| s.name()).collect::<Vec<_>>(),
                "mask": format!("0x{:04x}", plan.mask),
                "working_rate": plan.working_rate,
                "input_stage": plan.stage.label(),
                "chain": plan.chain,
                "decode_cmd": plan.input_args,
                "encode_cmd": plan.encode_cmd,
                "warnings": plan.warnings,
            })
        );
        return;
    }
    println!("  {} -> stereo", plan.info.describe());
    println!(
        "  layout: {} (mask 0x{:04x})",
        g6me_core::mask::format_order(&plan.order),
        plan.mask
    );
    println!("  stage:  {}", plan.stage.label());
    // `loss_chain` already numbers its steps (the report and the GUI print the
    // same strings), so numbering here would give "1. 1. decode".
    for step in &plan.chain {
        println!("    {step}");
    }
    println!("  decode: {}", join_cmd(&plan.input_args));
    println!("  encode: {}", join_cmd(&plan.encode_cmd));
    for w in &plan.warnings {
        println!("  warning: {w}");
    }
}

fn join_cmd(args: &[String]) -> String {
    args.iter()
        .map(|a| {
            if a.contains([' ', '"', '\'']) {
                format!("\"{a}\"")
            } else {
                a.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn caps_report(tools: &Tools, json: bool) -> std::io::Result<()> {
    let caps = probe_caps(tools);
    if json {
        println!("{}", serde_json::to_string(&caps)?);
    } else {
        println!("ffmpeg      {}", tools.ffmpeg.display());
        println!("ffprobe     {}", tools.ffprobe.display());
        println!("version     {}", caps.ffmpeg_version);
        println!("libsoxr     {}", yes(caps.soxr));
        println!("afir        {}", yes(caps.afir));
        println!("flac enc    {}", yes(caps.flac));
        println!("lavfi       {}", yes(caps.lavfi));
        println!("24-bit FLAC {}", caps.flac_summary());
        for n in &caps.notes {
            println!("note        {n}");
        }
    }
    Ok(())
}

fn yes(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

#[cfg(feature = "monitor")]
fn list_devices() -> ExitCode {
    let devs = match g6me_core::monitor::list_devices() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    if devs.is_empty() {
        eprintln!("no output devices found");
        return ExitCode::from(1);
    }
    for (i, d) in devs.iter().enumerate() {
        println!("{i}  {}  id={}", d.label(), d.id);
    }
    ExitCode::SUCCESS
}

/// Play files to a device: same decode/DSP path, no file.
#[cfg(feature = "monitor")]
fn monitor_files(cli: &Cli, tools: &Tools, caps: &Caps) -> ExitCode {
    use g6me_core::monitor::Monitor;

    let base_cfg = match cli.render_cfg() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let mut base_cfg = base_cfg;
    base_cfg.out.format = OutFormat::Null;
    let cfg = &base_cfg;
    let ext = "flac".to_string();
    let items = batch::discover(&cli.input, cli.out_dir.as_deref(), &ext, true);
    // No flag for this on the check path; folders are scanned recursively, as documented.
    if items.is_empty() {
        eprintln!("error: no audio files found");
        return ExitCode::from(2);
    }

    let cancel = Arc::new(AtomicBool::new(false));
    {
        let cancel = cancel.clone();
        let _ = ctrlc::set_handler(move || {
            cancel.store(true, Ordering::SeqCst);
        });
    }

    let mut failed = 0usize;
    for item in &items {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let meters = Arc::new(Meters::new());
        let mon =
            match Monitor::open(&cli.device, cli.queue * cli.block, meters.clone()).map(Arc::new) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
        // The device sets the clock. If the file's rate differs, resample to the
        // device rate — and say so, because that is a real change to the samples.
        let info = match g6me_core::probe::probe_audio(tools, &item.src, cli.stream) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("error: {}: {e}", item.src.display());
                failed += 1;
                continue;
            }
        };
        // One config per file, derived from what the user asked for, never carried over
        // from the file before it.
        let file_cfg = monitor_cfg_for(cfg, info.sample_rate, mon.sample_rate(), cli.rate);
        if file_cfg.resample.enable && !cfg.resample.enable {
            eprintln!(
                "note: {} is {} Hz, device is {} Hz — resampling with soxr to play in tune",
                item.src.file_name().unwrap_or_default().to_string_lossy(),
                info.sample_rate,
                mon.sample_rate()
            );
        }
        eprintln!(
            "playing {} on {}  [{} ch, {} Hz]",
            item.src.file_name().unwrap_or_default().to_string_lossy(),
            mon.device().name,
            info.channels,
            mon.sample_rate()
        );
        let opts = RenderOpts {
            tools,
            caps,
            cancel: Some(cancel.clone()),
            progress: None,
            tap: Some(mon.tap()),
        };
        match mon.start() {
            Ok(()) => {}
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        }
        let t0 = Instant::now();
        let dst = Path::new("");
        match pipeline::render_file(&item.src, dst, &file_cfg, &opts) {
            Ok(r) => {
                let peak: Vec<String> = meters
                    .peaks()
                    .iter()
                    .map(|p| format!("{:>6.1}", g6me_core::pcm::Stats::dbfs(*p as f64)))
                    .collect();
                if !cli.quiet {
                    eprintln!(
                        "played {} frames in {} ({}x realtime), peak dBFS: {}",
                        r.frames,
                        util::human_time(t0.elapsed().as_secs_f64()),
                        if t0.elapsed().as_secs_f64() > 0.0 {
                            r.seconds / t0.elapsed().as_secs_f64()
                        } else {
                            0.0
                        },
                        peak.join(" ")
                    );
                }
                let stats = mon.stats();
                if stats.underruns > 0 {
                    eprintln!(
                        "warning: {} device underruns ({} silent frames) — raise --queue or lower --threads",
                        stats.underruns, stats.silent_frames
                    );
                }
            }
            Err(e) => {
                eprintln!("FAILED {}: {e}", item.src.display());
                failed += 1;
            }
        }
    }
    if failed > 0 {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

#[cfg(not(feature = "monitor"))]
fn monitor_files(cli: &Cli, _tools: &Tools, _caps: &Caps) -> ExitCode {
    let _ = cli;
    eprintln!("this build has no monitoring support (feature `monitor` is off)");
    ExitCode::from(2)
}

/// The render config for one file in a monitor run.
///
/// The device sets the clock, so a file recorded at another rate has to be resampled to
/// it, which is a real change to the samples. This is a function of one file's rate
/// rather than a few lines that edit a config in place, because `--monitor` takes
/// folders: a config mutated by the first file would carry its resample into the second,
/// so a file that already matched the device would be resampled anyway — silently, and
/// with the explanatory message attached to whichever file happened to come first.
///
/// A `--rate` the user gave outranks the device: they asked for a rate, and layering a
/// second, guessed decision on top of it is not mine to make.
// Only the monitor path calls this; the tests below run it either way so the rule it
// encodes is checked in every build configuration, not just the one CI happens to build.
#[cfg_attr(not(feature = "monitor"), allow(dead_code))]
#[must_use]
fn monitor_cfg_for(
    base: &RenderCfg,
    file_rate: u32,
    device_rate: u32,
    user_rate: Option<u32>,
) -> RenderCfg {
    let mut cfg = base.clone();
    if user_rate.is_none() && file_rate != device_rate {
        cfg.resample.enable = true;
        cfg.resample.rate = device_rate;
    }
    cfg
}

/// A one-line summary of what happens to the samples, for the startup banner.
trait LossSummary {
    fn loss_summary(&self) -> String;
    fn depth_label(&self) -> String;
}

impl LossSummary for RenderCfg {
    /// `s32` reads as "32-bit int" — clearer in a banner than the enum name.
    fn depth_label(&self) -> String {
        self.out.depth.label().to_string()
    }

    fn loss_summary(&self) -> String {
        // In the order the samples meet them: rate and IR act on the source, before the
        // downmix.
        let mut v = vec!["decode".into(), "float64".into()];
        if self.resample.enable {
            v.push(format!(
                "soxr {} Hz/{}",
                self.resample.rate,
                self.resample.dither.label()
            ));
        }
        if self.ir.enable {
            v.push(format!(
                "IR {:?}",
                self.ir.file.as_deref().unwrap_or(Path::new("?"))
            ));
        }
        v.push("downmix".into());
        v.push(match &self.out.format {
            OutFormat::Flac { level } => format!("FLAC {} @ level {level}", self.depth_label()),
            OutFormat::Wav => format!("WAV {}", self.depth_label()),
            OutFormat::Null => "monitor only".to_string(),
        });
        v.join(" -> ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_touch_nothing() {
        let c = Cli::try_parse_from(["g6me", "x.mka"]).unwrap();
        let cfg = c.render_cfg().unwrap();
        assert!(!cfg.resample.enable, "resampling must be off by default");
        assert!(!cfg.ir.enable, "IR must be off by default");
        assert_eq!(cfg.out.dither, Dither::None, "no dither by default");
        // 24-bit, because that is the deepest FLAC stream ffmpeg's encoder
        // actually writes. The bytes are the same ones the old "32-bit" default
        // produced — only the label was wrong.
        assert_eq!(cfg.out.depth, BitDepth::S24);
        assert!(matches!(cfg.out.format, OutFormat::Flac { level: 8 }));
        assert!(cfg.out.copy_metadata);
        assert!(!cfg.out.overwrite);
    }

    #[test]
    fn threads_are_capped_at_the_cpu_count() {
        assert_eq!(parse_threads("1").unwrap(), 1);
        let big = util::max_threads() * 8 + 3;
        assert_eq!(
            parse_threads(&big.to_string()).unwrap(),
            util::max_threads(),
            "the ceiling is the machine, not the flag"
        );
        assert!(parse_threads("0").is_err());
        assert!(parse_threads("abc").is_err());
        assert_eq!(parse_threads("999999").unwrap(), util::max_threads());
    }

    #[test]
    fn dither_names_are_forgiving() {
        assert_eq!(parse_dither("low_shibata").unwrap(), Dither::LowShibata);
        assert_eq!(parse_dither("Low-Shibata").unwrap(), Dither::LowShibata);
        assert_eq!(parse_dither("triangular").unwrap(), Dither::Triangular);
        assert_eq!(parse_dither("none").unwrap(), Dither::None);
        assert!(parse_dither("voodoo").is_err());
    }

    #[test]
    fn twenty_four_bit_input_is_refused_with_a_better_message() {
        assert!(parse_resample_depth("24").is_str_err());
        assert_eq!(parse_resample_depth("32").unwrap(), BitDepth::S32);
        // Output side still allows it.
        assert_eq!(parse_depth("24").unwrap(), BitDepth::S24);
    }

    #[test]
    fn order_parsing_accepts_the_usual_spellings() {
        let o = parse_order("FL,FR,FC,LFE,BL,BR,SL,SR").unwrap().0;
        assert_eq!(o.len(), 8);
        assert_eq!(g6me_core::mask::mask_of(&o), 0x063f);
        assert_eq!(parse_order("fl fr fc lfe bl br sl sr").unwrap().0.len(), 8);
        assert!(parse_order("FL,FR,BOGUS").is_str_err());
        assert!(parse_order("").is_str_err());
    }

    /// Through clap, not just the parser: the field type is what used to panic.
    #[test]
    fn order_flag_reaches_the_render_config() {
        let c = Cli::try_parse_from(["g6me", "--order", "FL,FR,FC,LFE,SL,SR", "x.mka"]).unwrap();
        let cfg = c.render_cfg().unwrap();
        assert_eq!(
            cfg.order,
            Some(vec![
                Slot::Fl,
                Slot::Fr,
                Slot::Fc,
                Slot::Lfe,
                Slot::Sl,
                Slot::Sr
            ])
        );
    }

    #[test]
    fn setpoint_sets_both_the_trim_and_the_ceiling() {
        let c = Cli::try_parse_from(["g6me", "--setpoint", "-6", "x.mka"]).unwrap();
        let cfg = c.render_cfg().unwrap();
        assert_eq!(cfg.output_gain_db, -6.0);
        assert_eq!(cfg.limiter.ceiling_db, -6.0);
        assert!(cfg.limiter.enable);

        let c = Cli::try_parse_from(["g6me", "--gain", "-2", "x.mka"]).unwrap();
        let cfg = c.render_cfg().unwrap();
        assert_eq!(
            (cfg.output_gain_db, cfg.limiter.ceiling_db),
            (-2.0, DEFAULT_CEILING_DB)
        );
        let e = Cli::try_parse_from(["g6me", "--gain", "3", "x.mka"])
            .unwrap_err()
            .to_string();
        assert!(e.contains("gain in dB"), "{e}");
    }

    #[test]
    fn banner_lists_input_stages_before_the_downmix() {
        let c = Cli::try_parse_from(["g6me", "-r", "44100", "x.mka"]).unwrap();
        let s = c.render_cfg().unwrap().loss_summary();
        assert!(s.find("soxr").unwrap() < s.find("downmix").unwrap(), "{s}");
    }

    #[test]
    fn ir_flags_map_to_the_mpv_shape() {
        let c = Cli::try_parse_from(["g6me", "--ir", "/tmp/ir.wav", "x.mka"]).unwrap();
        let cfg = c.render_cfg().unwrap();
        assert!(cfg.ir.enable);
        assert_eq!(cfg.ir.file.clone().unwrap(), PathBuf::from("/tmp/ir.wav"));
        // f64 arithmetic is an invariant, so it is asserted through the filter
        // string the encoder will actually run, not through a setting.
        assert!(
            g6me_core::stages::afir_filter(&cfg.ir).contains("precision=double"),
            "{}",
            g6me_core::stages::afir_filter(&cfg.ir)
        );
        assert_eq!(cfg.ir.minp, 4096);
        assert_eq!(cfg.ir.maxp, 4096);
        // Unset knobs stay unset so the filter string omits them, like mpv's.
        assert_eq!(cfg.ir.irnorm, Some(-1.0));
        assert_eq!(cfg.ir.gtype, None);
        let f = g6me_core::stages::afir_filter(&cfg.ir);
        assert!(f.contains("irfmt=mono"), "{f}");
        assert!(f.contains("precision=double"), "{f}");
        assert!(f.contains("irnorm=-1"), "{f}");
        assert!(!f.contains("gtype"), "{f}");
    }

    #[test]
    fn rate_flag_enables_soxr_and_reaches_the_filter() {
        let c = Cli::try_parse_from(["g6me", "-r", "44100", "--dither", "low_shibata", "x.mka"])
            .unwrap();
        let cfg = c.render_cfg().unwrap();
        assert!(cfg.resample.enable);
        assert_eq!(cfg.resample.rate, 44100);
        assert_eq!(cfg.out.dither, Dither::LowShibata);
        assert_eq!(
            cfg.out.resample, None,
            "resampling happens before the downmix only"
        );
    }

    /// The help text promises "Defaults to 8". This is the test that makes it true.
    ///
    /// Both frontends used `unwrap_or_default()` for the limiter's ratio, and `Ratio::default`
    /// is the meter's x4, so every shipped render ran the side chain a whole step shallower
    /// than the manual, the code comment and `Cfg::default` each claimed. `guard_db` went
    /// from 0.04 dB to 0.62 dB, which is most of the 0.6 dB a -3 dBTP ceiling undershot by.
    #[test]
    fn the_limiter_ratio_default_is_the_limiters_not_the_meters() {
        assert_eq!(Ratio::default(), Ratio::X4, "the meter's default moved");
        assert_eq!(DEFAULT_RATIO, Ratio::X8);
        let c = Cli::try_parse_from(["g6me", "x.mka"]).unwrap();
        assert_eq!(
            c.render_cfg().unwrap().limiter.ratio,
            DEFAULT_RATIO,
            "the CLI fell back to the meter's ratio, so its own help text is lying"
        );
        let c = Cli::try_parse_from(["g6me", "--limit-ratio", "16", "x.mka"]).unwrap();
        assert_eq!(c.render_cfg().unwrap().limiter.ratio, Ratio::X16);
    }

    #[test]
    fn loss_summary_says_what_runs() {
        let c = Cli::try_parse_from(["g6me", "x.mka"]).unwrap();
        let s = c.render_cfg().unwrap().loss_summary();
        assert!(s.contains("decode"), "{s}");
        assert!(!s.contains("soxr"), "{s}");
        assert!(s.contains("FLAC"), "{s}");
        let c = Cli::try_parse_from(["g6me", "-r", "48000", "-f", "wav", "--depth", "16", "x"])
            .unwrap();
        let s = c.render_cfg().unwrap().loss_summary();
        assert!(s.contains("soxr"), "{s}");
        assert!(s.contains("WAV"), "{s}");
    }

    #[test]
    fn bad_combos_are_rejected_before_any_render() {
        // Float FLAC is not a thing.
        let c = Cli::try_parse_from(["g6me", "-f", "flac", "--depth", "f32", "x"]).unwrap();
        let cfg = c.render_cfg().unwrap();
        assert!(cfg.out.validate(&Caps::default()).is_err());
        // 16-bit FLAC is fine.
        let c = Cli::try_parse_from(["g6me", "-f", "flac", "--depth", "16", "x"]).unwrap();
        assert!(c
            .render_cfg()
            .unwrap()
            .out
            .validate(&Caps::default())
            .is_ok());
    }

    /// The bound is ffmpeg's FLAC encoder's own: on n9.0.2 levels 0..=12 encode
    /// (12 packs smallest) and 13 fails at encoder init with "invalid compression
    /// level". The reference `flac` CLI's -0..-8 presets are a different scale and
    /// are not what G6ME drives, so 9 is legal here even though flac reserves it.
    #[test]
    fn level_is_bounded_like_ffmpeg_flac() {
        assert!(Cli::try_parse_from(["g6me", "--level", "9", "x"]).is_ok());
        assert!(Cli::try_parse_from(["g6me", "--level", "12", "x"]).is_ok());
        assert!(Cli::try_parse_from(["g6me", "--level", "13", "x"]).is_err());
        assert!(Cli::try_parse_from(["g6me", "--level", "0", "x"]).is_ok());
        assert!(Cli::try_parse_from(["g6me", "--level", "-1", "x"]).is_err());
    }

    #[test]
    fn numeric_arguments_reject_nonsense_with_a_readable_message() {
        assert_eq!(parse_level("8").unwrap(), 8);
        assert_eq!(parse_rate("48_000").unwrap(), 48000);
        assert!(parse_rate("4800").is_err(), "below the floor");
        assert!(parse_rate("9999999").is_err(), "above the ceiling");
        assert!(parse_block("63").is_err());
        assert!(parse_queue("0").is_err());
        assert!(parse_part("128").is_err());
        assert!(parse_precision_bits("16").is_err());
        assert_eq!(parse_precision_bits("28").unwrap(), 28.0);
        assert_eq!(parse_ir_norm("-1").unwrap(), -1.0);
        assert!(parse_ir_norm("2").is_err());
        // The message must name the bound, not just "invalid value".
        let e = parse_rate("4800").unwrap_err();
        assert!(e.contains("8000..=384000"), "{e}");
    }

    /// `afir`'s `length` is a fraction, and a value outside 0..=1 kills the filter graph
    /// per file rather than being clamped.
    #[test]
    fn ir_length_is_a_fraction_and_is_checked_as_one() {
        let c = Cli::try_parse_from(["g6me", "a.mka", "--ir-length", "0.25"]).unwrap();
        assert_eq!(c.ir_length, Some(0.25));
        for bad in ["1.5", "2", "200", "-0.1"] {
            let e = Cli::try_parse_from(["g6me", "a.mka", "--ir-length", bad]).unwrap_err();
            let msg = e.to_string();
            assert!(
                msg.contains("fraction"),
                "'{bad}' must be refused by name and explained as a fraction: {msg}"
            );
        }
    }

    /// The documented meaning of `--ir-length 0` is the whole IR, and the whole IR is what
    /// `afir` does when the option is absent. Passing `length=0` would instead convolve a
    /// fraction of zero of the response.
    #[test]
    fn a_zero_ir_length_means_the_whole_ir() {
        let c = Cli::try_parse_from(["g6me", "a.mka", "--ir-length", "0"]).unwrap();
        let cfg = c.render_cfg().unwrap();
        assert_eq!(
            cfg.ir.length, None,
            "0 asks for the whole IR, which is afir's default"
        );
        let half = Cli::try_parse_from(["g6me", "a.mka", "--ir-length", "0.5"])
            .unwrap()
            .render_cfg()
            .unwrap();
        assert_eq!(half.ir.length, Some(0.5));
    }

    #[test]
    fn many_inputs_are_accepted() {
        let c = Cli::try_parse_from(["g6me", "a.mka", "b.mka", "/music/jazz"]).unwrap();
        assert_eq!(c.input.len(), 3);
    }

    /// A monitor run walks a folder, and the device rate is only known once.
    ///
    /// Sabotage check: replace `base.clone()` with a config shared and mutated across the
    /// loop and the second assertion fails — the 48 kHz file gets resampled by a decision
    /// that belonged to the 44.1 kHz one before it, with no message about it.
    #[test]
    fn a_monitored_file_does_not_inherit_the_previous_files_resample() {
        let base = RenderCfg::with_defaults();
        assert!(!base.resample.enable, "default path does not resample");
        let off_rate = monitor_cfg_for(&base, 44_100, 48_000, None);
        assert!(off_rate.resample.enable, "44.1 kHz on a 48 kHz device");
        assert_eq!(
            off_rate.resample.rate, 48_000,
            "to the device, not anywhere"
        );
        let matched = monitor_cfg_for(&base, 48_000, 48_000, None);
        assert!(
            !matched.resample.enable,
            "a file already at the device rate must not be resampled because some other \
             file needed it"
        );
        assert_eq!(
            base,
            RenderCfg::with_defaults(),
            "the caller's config is untouched"
        );
    }

    #[test]
    fn an_explicit_rate_outranks_the_device_in_a_monitor_run() {
        let base = RenderCfg::with_defaults();
        let cfg = monitor_cfg_for(&base, 44_100, 48_000, Some(96_000));
        assert_eq!(
            cfg, base,
            "--rate was given, so the device rate is not layered on top of it"
        );
    }

    trait StrErr {
        fn is_str_err(&self) -> bool;
    }
    impl<T, E: std::fmt::Debug> StrErr for Result<T, E> {
        fn is_str_err(&self) -> bool {
            self.is_err()
        }
    }
}
