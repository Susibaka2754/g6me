//! Peak meters with the behaviour of the VST2 delivery's parameters, so a level
//! read here matches what a host used to read from the plugin — with one
//! deliberate correction at the -60 dBFS floor, explained on [`level_to_param`].
//! The meter *curve* (instant attack, `exp(-n / (0.30 * sr))` release) is kept
//! identical, because that is what makes screenshots and log levels comparable
//! with the plugin.
//!
//! From `g6-re/windows/plugin/g6ct_vst.c`:
//!
//! ```c
//! decay_for(n)   = exp(-n / (0.30 * sample_rate))     /* 0.30 s peak-hold release */
//! hold_level(p)  = peak > prev*decay ? peak : prev*decay  /* instant attack */
//! level_to_param = (clamp(20*log10(v), -60, 0) + 60) / 60
//! ```
//!
//! Meters are the ten the plugin exposed: eight inputs mapped to the DSP's slot
//! table `{0,1,2,3,4,5,9,10}` plus the two outputs. Single writer (the audio
//! thread), many readers (GUI), so each meter is one `AtomicU32` holding `f32`
//! bits — no lock, and a torn read is impossible.

use crate::dsp::Real;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrd};

/// Peak-hold release time constant, seconds. Matches `METER_FALL_S`.
pub const METER_FALL_S: f64 = 0.30;

/// Input meters: VST channel -> DSP slot, matching the plugin's `g_slot`.
pub const IN_SLOTS: [usize; 8] = [0, 1, 2, 3, 4, 5, 9, 10];

pub const IN_CHANNELS: usize = 8;
pub const OUT_CHANNELS: usize = 2;
pub const TOTAL: usize = IN_CHANNELS + OUT_CHANNELS;

pub const IN_LABELS: [&str; IN_CHANNELS] = [
    "In1 FL", "In2 FR", "In3 FC", "In4 LFE", "In5 BL", "In6 BR", "In7 SL", "In8 SR",
];
pub const OUT_LABELS: [&str; OUT_CHANNELS] = ["OutL", "OutR"];

/// Live meter state, shared between the audio/callback thread and the UI.
#[derive(Debug, Default)]
pub struct Meters {
    values: [AtomicU32; TOTAL],
    /// `f32` sample rate and last block size, packed as bits, for `tick()`.
    rate_bits: AtomicU32,
    block: AtomicU64,
    frames: AtomicU64,
    xruns: AtomicU64,
}

impl Meters {
    pub fn new() -> Self {
        Self {
            values: std::array::from_fn(|_| AtomicU32::new(0)),
            rate_bits: AtomicU32::new(0),
            block: AtomicU64::new(0),
            frames: AtomicU64::new(0),
            xruns: AtomicU64::new(0),
        }
    }

    /// Tell the meters the working rate and the block size they are fed, which
    /// is what sets the decay per call.
    pub fn set_format(&self, sample_rate: f32, block: usize) {
        self.rate_bits
            .store(sample_rate.to_bits(), AtomicOrd::Relaxed);
        self.block.store(block as u64, AtomicOrd::Relaxed);
    }

    fn decay(&self) -> f32 {
        let sr = f32::from_bits(self.rate_bits.load(AtomicOrd::Relaxed));
        let sr = if sr > 0.0 { sr as f64 } else { 48000.0 };
        let n = self.block.load(AtomicOrd::Relaxed).max(1) as f64;
        (-(n / (METER_FALL_S * sr))).exp() as f32
    }

    fn hold(&self, index: usize, peak: f32) {
        let prev = f32::from_bits(self.values[index].load(AtomicOrd::Relaxed));
        // A non-finite previous value is decayed to silence rather than multiplied: inf ×
        // 0.99 is inf, so one bad sample used to latch the bar at full scale for the rest
        // of the session, and only reset() would free it.
        let fall = if prev.is_finite() {
            prev * self.decay()
        } else {
            0.0
        };
        let peak = if peak.is_finite() { peak } else { f32::MAX };
        let v = if peak > fall { peak } else { fall };
        self.values[index].store(v.to_bits(), AtomicOrd::Relaxed);
    }

    /// Peak of one buffer, in the `f32` domain exactly as the plugin measured it
    /// (`(float)fabs((double)x[i])`).
    fn peak_of<T: Real>(buf: &[T], n: usize) -> f32 {
        let mut peak = 0.0f32;
        for &x in buf.iter().take(n) {
            // Clamped, not cast. The plugin this reproduces stored the peak in a float too,
            // but its samples came from a float buffer, so it could not meet an f64 above
            // f32::MAX; a bare `as f32` of one yields +inf, which [`Self::hold`] would then
            // latch. A non-finite sample is skipped, not clamped: `f64::min` returns the
            // other operand when one is NaN, so clamping turned a NaN into f32::MAX.
            let a = x.to_f64().abs();
            if !a.is_finite() {
                continue;
            }
            let v = a.min(f64::from(f32::MAX)) as f32;
            if v > peak {
                peak = v;
            }
        }
        peak
    }

    /// Feed one processed block: `planar` is indexed by DSP slot, `n` frames.
    pub fn observe_block<T: Real>(
        &self,
        planar: &[Option<&[T]>; crate::dsp::SLOTS],
        l: &[T],
        r: &[T],
        n: usize,
    ) {
        for (i, &slot) in IN_SLOTS.iter().enumerate() {
            if let Some(buf) = planar[slot] {
                self.hold(i, Self::peak_of(buf, n));
            }
        }
        self.hold(IN_CHANNELS, Self::peak_of(l, n));
        self.hold(IN_CHANNELS + 1, Self::peak_of(r, n));
        self.frames.fetch_add(n as u64, AtomicOrd::Relaxed);
    }

    /// Release-only tick, for a source that went silent (or between files).
    pub fn decay_once(&self) {
        for i in 0..TOTAL {
            self.hold(i, 0.0);
        }
    }

    /// Linear peak, 0.0 .. >1.0 (above 1.0 means the encoder will clip).
    pub fn peak(&self, index: usize) -> f32 {
        self.values
            .get(index)
            .map(|v| f32::from_bits(v.load(AtomicOrd::Relaxed)))
            .unwrap_or(0.0)
    }

    pub fn peaks(&self) -> [f32; TOTAL] {
        std::array::from_fn(|i| self.peak(i))
    }

    /// Host-visible parameter, `-60..0 dB` mapped onto `0..1`.
    pub fn param(&self, index: usize) -> f32 {
        level_to_param(self.peak(index))
    }

    pub fn frames(&self) -> u64 {
        self.frames.load(AtomicOrd::Relaxed)
    }

    pub fn xruns(&self) -> u64 {
        self.xruns.load(AtomicOrd::Relaxed)
    }

    pub fn note_xrun(&self, n: u64) {
        self.xruns.fetch_add(n, AtomicOrd::Relaxed);
    }

    pub fn reset(&self) {
        for v in &self.values {
            v.store(0, AtomicOrd::Relaxed);
        }
        self.frames.store(0, AtomicOrd::Relaxed);
        self.xruns.store(0, AtomicOrd::Relaxed);
    }
}

/// Meter value -> normalised dB parameter: `-60..0 dB` mapped onto `0..1`.
///
/// The VST plugin computed this as `20*log10(v)` and clamped the result to
/// `[-60, 0]`. That has a defect worth not inheriting: `log10(0.001)` is
/// `-2.9999999999999996`, so a signal sitting exactly at the -60 dB floor read
/// `6.9e-9` instead of `0`, and the floor was therefore unreachable — a meter
/// that never quite rests. The floor and ceiling are defined here by what they
/// *mean* (`-60 dBFS` is exactly `0.001`, `0 dBFS` is exactly `1.0`), which
/// reaches the same endpoints exactly and is monotone between them.
///
/// Only the meter scale changed. Nothing in [`crate::dsp`] is adjusted this way:
/// reproducing the module's transfer function is the point of the port.
pub fn level_to_param(v: f32) -> f32 {
    // Written as a failed `>` rather than `<=` so that NaN takes this branch: both
    // comparisons with NaN are false, and a NaN reaching egui's geometry panics on the
    // first Rect built from it. A meter has no honest answer for NaN; the floor is the
    // safe one, and the value is still visible in the numeric report.
    // NaN has no ordering, and `partial_cmp` says so instead of quietly answering `false`
    // twice. A meter has no honest reading for NaN; the floor is the safe one, and the
    // value is still visible in the numeric report.
    if v.partial_cmp(&0.001).is_none_or(|o| o.is_le()) {
        return 0.0; // at or below the -60 dBFS floor, and anything unmeasurable
    }
    if v >= 1.0 {
        return 1.0; // full scale and above
    }
    let db = 20.0 * (v as f64).log10();
    ((db + 60.0) / 60.0) as f32
}

/// Inverse, for axis labels.
pub fn param_to_db(p: f32) -> f64 {
    (p as f64) * 60.0 - 60.0
}

/// A [`SampleTap`](crate::pipeline::SampleTap) that only meters — no device, no
/// output, no `cpal`.
///
/// This is what keeps a GUI's levels alive during a batch. A render is the only
/// thing that can say what the downmixer produced, so the meters are fed by the
/// render itself rather than by a parallel "just measure it" path that would
/// drift from the real one the first time a filter changed.
///
/// One of these per file, not one per batch: two files sharing a meter show a
/// third level that belongs to neither.
pub struct MeterTap {
    meters: std::sync::Arc<Meters>,
    rate: AtomicU32,
}

impl MeterTap {
    pub fn new(meters: std::sync::Arc<Meters>) -> Self {
        Self {
            meters,
            // Until `will_render` says otherwise, assume the rate the plugin ran
            // at; a wrong guess only moves the release constant, never a sample.
            rate: AtomicU32::new(48_000),
        }
    }

    pub fn meters(&self) -> &std::sync::Arc<Meters> {
        &self.meters
    }

    /// The rate this tap last rendered at, for the meter caption.
    pub fn rate(&self) -> u32 {
        self.rate.load(AtomicOrd::Relaxed)
    }
}

impl crate::pipeline::SampleTap for MeterTap {
    fn will_render(&self, sample_rate: u32) {
        if sample_rate > 0 {
            self.rate.store(sample_rate, AtomicOrd::Relaxed);
        }
    }

    fn tap(&self, planar: &[Option<&[f64]>; crate::dsp::SLOTS], l: &[f64], r: &[f64], n: usize) {
        // Block sizes vary along a file (the queue delivers what the decoder
        // produced), so the release constant is recomputed per block — that is
        // what keeps `exp(-n / (0.30 * sr))` exact instead of approximated.
        self.meters.set_format(self.rate() as f32, n);
        self.meters.observe_block(planar, l, r, n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_matches_the_vst_formula() {
        let m = Meters::new();
        m.set_format(48000.0, 1024);
        // exp(-1024 / (0.30 * 48000))
        let want = (-(1024.0f64 / (METER_FALL_S * 48000.0))).exp() as f32;
        assert_eq!(m.decay().to_bits(), want.to_bits());
    }

    #[test]
    fn attack_is_instant_release_is_exponential() {
        let m = Meters::new();
        m.set_format(48000.0, 480);
        m.hold(0, 0.5);
        assert_eq!(m.peak(0), 0.5);
        // A quieter block must not pull the meter down instantly: it falls by
        // exactly exp(-n / (0.30 * sr)) per block, which at 480 frames is 3.3 %.
        m.hold(0, 0.1);
        let one_step = 0.5 * (-(480.0f64 / (METER_FALL_S * 48000.0))).exp() as f32;
        assert_eq!(
            m.peak(0).to_bits(),
            one_step.to_bits(),
            "release must follow the VST curve"
        );
        m.hold(0, 0.9); // above -> instant
        assert_eq!(m.peak(0), 0.9);
    }

    #[test]
    fn param_scale_matches_plugin() {
        assert_eq!(level_to_param(1.0), 1.0);
        assert_eq!(level_to_param(0.0), 0.0);
        // level_to_param(v) = (clamp(20*log10(v), -60, 0) + 60) / 60, so half
        // scale (-6.0206 dB) sits at 0.8997, not near zero.
        assert!(
            (level_to_param(0.5) - 0.8997).abs() < 0.0002,
            "{}",
            level_to_param(0.5)
        );
        // The floor is reachable and exact, which is the behaviour the plugin
        // intended and could not express.
        assert_eq!(level_to_param(0.001), 0.0, "-60 dBFS reads zero");
        assert_eq!(level_to_param(0.0000001), 0.0);
        assert_eq!(level_to_param(0.0), 0.0);
        // overshoot past full scale clamps, like the plugin's parameter
        assert_eq!(level_to_param(2.0), 1.0);
        assert!(level_to_param(0.9) < level_to_param(0.95), "monotone");
        // the inverse of the -60..0 dB scale
        assert_eq!(param_to_db(0.5), -30.0);
        assert_eq!(param_to_db(0.0), -60.0);
        assert_eq!(param_to_db(1.0), 0.0);
    }

    #[test]
    fn block_observation_uses_the_slot_map() {
        let m = Meters::new();
        m.set_format(48000.0, 4);
        let loud = [-0.25f64, 0.75, 0.0, 0.0];
        let quiet = [0.05f64, 0.0, 0.0, 0.0];
        let mut planar: [Option<&[f64]>; crate::dsp::SLOTS] = [None; crate::dsp::SLOTS];
        planar[0] = Some(&loud); // FL -> In1
        planar[10] = Some(&quiet); // SR -> In8
        m.observe_block(&planar, &loud, &quiet, 4);
        assert_eq!(m.peak(0), 0.75);
        assert_eq!(m.peak(7), 0.05);
        assert_eq!(m.peak(8), 0.75);
        assert_eq!(m.peak(9), 0.05);
        assert_eq!(m.frames(), 4);
    }

    /// A non-finite sample is not a level: it must neither become the peak nor hide
    /// the finite samples around it.
    #[test]
    fn non_finite_samples_are_skipped() {
        let m = Meters::new();
        m.set_format(48000.0, 4);
        let l = [f64::NAN, 0.25, 0.0, 0.0];
        let r = [0.5f64, f64::INFINITY, f64::NEG_INFINITY, 0.0];
        let planar: [Option<&[f64]>; crate::dsp::SLOTS] = [None; crate::dsp::SLOTS];
        m.observe_block(&planar, &l, &r, 4);
        assert_eq!(m.peak(IN_CHANNELS), 0.25);
        assert_eq!(m.peak(IN_CHANNELS + 1), 0.5);
        // An f64 beyond f32 range is still clamped rather than skipped.
        m.reset();
        let big = [1e300f64, 0.0, 0.0, 0.0];
        m.observe_block(&planar, &big, &big, 4);
        assert_eq!(m.peak(IN_CHANNELS), f32::MAX);
    }

    #[test]
    fn meter_tap_measures_without_a_device() {
        use crate::pipeline::SampleTap;
        let m = std::sync::Arc::new(Meters::new());
        let tap = MeterTap::new(m.clone());
        assert_eq!(
            tap.rate(),
            48_000,
            "a tap that has not rendered yet still meters"
        );

        let loud = [0.5f64, -0.75, 0.0, 0.0];
        let zero = [0.0f64; 4];
        let mut planar: [Option<&[f64]>; crate::dsp::SLOTS] = [None; crate::dsp::SLOTS];
        planar[3] = Some(&loud); // LFE -> In4
        tap.tap(&planar, &loud, &zero, 4);
        assert_eq!(m.peak(3), 0.75);
        assert_eq!(m.peak(8), 0.75, "the pair follows the input it was given");
        assert_eq!(m.peak(9), 0.0);
    }

    #[test]
    fn meter_tap_releases_at_the_rate_it_was_told() {
        use crate::pipeline::SampleTap;
        let m = std::sync::Arc::new(Meters::new());
        let tap = MeterTap::new(m.clone());
        tap.will_render(44_100);
        assert_eq!(tap.rate(), 44_100);

        let buf = [0.8f64; 8];
        let mut planar: [Option<&[f64]>; crate::dsp::SLOTS] = [None; crate::dsp::SLOTS];
        planar[0] = Some(&buf);
        tap.tap(&planar, &buf, &buf, 8);
        assert_eq!(m.peak(0), 0.8);

        // Silence next: the fall must use 44 100, not the 48 000 default.
        let quiet = [0.0f64; 8];
        planar[0] = Some(&quiet);
        tap.tap(&planar, &quiet, &quiet, 8);
        let at_441 = (0.8 * (-(8.0 / (METER_FALL_S * 44_100.0))).exp()) as f32;
        let at_480 = (0.8 * (-(8.0 / (METER_FALL_S * 48_000.0))).exp()) as f32;
        assert_eq!(m.peak(0).to_bits(), at_441.to_bits());
        assert_ne!(
            at_441.to_bits(),
            at_480.to_bits(),
            "the two are meant to differ"
        );
    }

    #[test]
    fn zero_rate_is_ignored_not_stored() {
        use crate::pipeline::SampleTap;
        let tap = MeterTap::new(std::sync::Arc::new(Meters::new()));
        tap.will_render(0);
        assert_eq!(
            tap.rate(),
            48_000,
            "a zero rate would divide the decay by zero"
        );
    }
}
