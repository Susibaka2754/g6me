//! True-peak ceiling limiter, and the metering of its own effect.
//!
//! ## Where it sits
//!
//! Immediately after the downmixer and before ffmpeg receives a byte, so it is
//! applied to the same `f64` samples the matrix produced and *before* any
//! quantisation to s16/s24/f32. A ceiling applied after quantisation would be
//! applied to a signal that has already been clipped by the format.
//!
//! ## Shape of the gain computer
//!
//! The side chain is the [`crate::tp`] detector: a zero-phase-equivalent linear-phase
//! FIR oversampled peak estimate. There is deliberately **no IIR in the side chain** —
//! no RMS detector, no feedback, no one-pole in the peak path — because an IIR has a
//! group delay of its own that the limiter cannot see ahead of, and a limiter whose
//! idea of "now" is smeared cannot honour a hard ceiling.
//!
//! Gain is channel-linked by degree: [`Cfg::link_transient`] and [`Cfg::link_release`]
//! blend each channel's own gain with the group's, the fast stage fully linked by default
//! and the slow stage at 75 %. Fully independent gains would change the level
//! relationships the matrix exists to create, which makes the processor change the thing
//! the product is.
//!
//! ```text
//!  input[i] ──► per-channel TP detector ──► p[i] ──► window minimum over the
//!            (group max, linked)                      lookahead ──► target gain
//!        │                                                          │
//!        └──► delay line, L = lookahead + group delay ──► × gain ◄──┘
//! ```
//!
//! The audio is delayed by `L` so the gain applied to a sample was computed from a
//! peak that had already been measured. That is the whole content of "lookahead":
//! without the delay the gain can only react to a peak after the sample that caused
//! it has been emitted.
//!
//! ## Why the lookahead has a floor
//!
//! Three things have to fit inside the lookahead, or the ceiling is a suggestion:
//!
//! 1. the interpolator's group delay ([`Ratio::group_delay`]) — a peak is not
//!    measured until its energy has emerged from the FIR;
//! 2. the attack ramp ([`ATTACK_MS`]) — a gain step is a discontinuity in the output
//!    and is ramped linearly to avoid clicking, and a ramp has to be finished before
//!    the peak it is protecting arrives;
//! 3. one sample, so "arriving before" means strictly before.
//!
//! [`Cfg::lookahead_samples`] applies that floor, and [`Cfg::effective_lookahead_ms`]
//! reports what was actually used: a request below the floor is raised, never
//! silently honoured short. At 48 kHz and x8 (the default) the floor is 29 samples,
//! 0.60 ms, so the 2 ms default fits and a 0.2 ms request is raised to 0.60 ms; the
//! Kaiser banks have a 25-sample group delay because their transition band is narrow
//! (see [`crate::tp::BAND_EDGE`]). The report says so whenever it happens.
//!
//! ## When it is off
//!
//! [`Limiter::disabled`] returns a limiter that never attenuates and adds no delay, and
//! the pipeline skips the stage entirely, so the byte stream is the one G6ME produced
//! before this stage existed. Nothing in the disabled path touches a sample.

use crate::tp::{Detector, Ratio, TruePeak};

/// Linear attack time. Short enough to catch a single-cycle transient, long enough
/// that the gain change is not a step in the output.
pub const ATTACK_MS: f64 = 0.5;

/// The limiter's own share of the ceiling guard band, in dB.
///
/// Not a fudge factor, and not the whole guard. A limiter multiplies a *neighbourhood* of
/// samples by a gain that is itself moving, so the interpolated peak of the output is not
/// simply the input peak times the gain: a peak sitting on a rising gain ends up a few
/// ten-thousandths to a few hundred-thousandths of a dB above the target, and how much
/// depends on the slope the lookahead allows. Aiming slightly low buys that difference
/// back with room to spare and costs a level change no listener has ever measured.
///
/// The rest of the guard belongs to the meter: [`Ratio::worst_underread_db`] is how much
/// a peak the side chain cannot see can be worth, and no amount of lookahead recovers it.
/// [`Cfg::guard_db`] adds the two, so a declared ceiling means the same thing at every
/// ratio instead of silently meaning "the ceiling as x2 sees it".
pub const GAIN_SLEW_GUARD_DB: f64 = 0.02;

/// Lowest ceiling offered. Below this the processor is a attenuator, not a limiter,
/// and `--gain` already exists for that.
pub const MIN_CEILING_DB: f64 = -24.0;

/// Highest ceiling: digital full scale, i.e. "make sure nothing exceeds 0 dBTP".
pub const MAX_CEILING_DB: f64 = 0.0;

/// Shortest gain ramp that is a ramp rather than a step, in frames.
pub const MIN_RAMP_FRAMES: usize = 2;

/// How close a one-pole follower has to be to its target before it is told to arrive.
///
/// `v += (target - v) * k` converges in real arithmetic and does not in IEEE 754: once
/// `target - v` is small enough that the product rounds below half an ULP of `v`, `v +`
/// that product is `v` again, exactly, forever. At 48 kHz with a 900 ms release that
/// stall lands at 0.99999999999760192 — 2.1e-11 dB below unity, inaudible in the audio and
/// fatal to any test that asks whether the gain came back.
///
/// The dead band is written as a difference rather than as a floor on the increment
/// because its job is to say "this is the target, for every purpose this processor has".
pub const GAIN_DEADBAND: f64 = 1e-9;

/// Smallest gain reduction that counts as the limiter working.
///
/// `engaged_frames` and `release_frames` are counts of frames the processor changed the
/// signal on, and a frame changed by 1e-11 dB was not changed in any sense a listener, a
/// diff, or a second render would agree on. Without this the counters answer the question
/// "has a peak ever happened" — which stays true for the rest of the file.
pub const ENGAGED_EPS: f64 = 1e-4; // ~0.001 dB

/// Smallest lookahead the UI and CLI offer, in ms.
pub const MIN_LOOKAHEAD_MS: f64 = 0.2;

/// Largest lookahead offered. Past this the processor is audible as pre-ringing of
/// the gain envelope rather than as protection.
pub const MAX_LOOKAHEAD_MS: f64 = 2.0;

/// Lookahead the UI and CLI start with. Two ms is the top of the range on purpose:
/// with a 26-frame-deep x8 side chain there is nothing to gain from less, and the
/// slower the arrival, the less the gain movement itself makes new inter-sample peaks.
pub const DEFAULT_LOOKAHEAD_MS: f64 = 2.0;

/// Ceiling the UI and CLI start with, dBTP.
pub const DEFAULT_CEILING_DB: f64 = -3.0;

/// Side-chain over-sampling ratio the UI and CLI start with.
///
/// This is deliberately *not* [`Ratio::default()`], which is x4 and belongs to the meter.
/// A limiter that runs at the meter's ratio gives away 0.60 dB of its own ceiling — the
/// printed ITU bank reads up to that much low on a crest landing between grid points — and
/// `guard_db` has to be paid back out of the headroom the user asked to keep. Both frontends
/// used to say `unwrap_or_default()` and so silently meant x4, while `Cfg::default`, the CLI
/// help and the README all said 8x.
pub const DEFAULT_RATIO: Ratio = Ratio::X8;

/// Shortest attack for the slow stage. Below about this the "slow" stage is catching
/// peaks, which is the fast stage's job, and the two stop meaning anything.
pub const MIN_ATTACK_MS: f64 = 1.0;
/// Longest attack for the slow stage.
pub const MAX_ATTACK_MS: f64 = 200.0;
/// Attack the UI and CLI start with, ms.
pub const DEFAULT_ATTACK_MS: f64 = 60.0;
/// Shortest release for the slow stage, ms.
pub const MIN_RELEASE_MS: f64 = 10.0;
/// Longest release for the slow stage, ms.
pub const MAX_RELEASE_MS: f64 = 5_000.0;
/// Release the UI and CLI start with, ms.
pub const DEFAULT_RELEASE_MS: f64 = 900.0;
/// Channel linking of the fast stage at the default. Full linking means a short peak
/// in one channel ducks both; the honest reason to turn it down is that a peak only in
/// one channel is nearly inaudible if it is removed only in that channel.
pub const DEFAULT_LINK_TRANSIENT: f64 = 1.0;
/// Channel linking of the slow stage at the default. The slow stage moves level over
/// seconds, so it is the one that can wreck a stereo image, and 75 % leaves a little of
/// each channel's own dynamics standing.
pub const DEFAULT_LINK_RELEASE: f64 = 0.75;

/// Window the slow stage measures its level over, in ms, and not a control.
///
/// It is what separates "a peak" from "a level": below this the slow stage is a second
/// peak limiter, above it the stage starts arriving late to the thing it is meant to be
/// smoothing. Exposing it would let the user set attack and release against a reference
/// that also moves, so it is fixed and stated here.
pub const RELEASE_WINDOW_MS: f64 = 5.0;

/// How fast the fast stage is allowed to let go, in ms.
///
/// The scheduler already decides *when* the gain has to be back up; this is only what
/// stops that return from being a step. It is short because the slow stage exists: level
/// recovery is no longer this stage's job, so holding on for 120 ms after every peak
/// would be the slow stage's behaviour charged twice.
pub const TRANSIENT_RELEASE_MS: f64 = 25.0;

/// Settings for the true-peak limiter.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Cfg {
    /// Off by default: the whole point of G6ME is that nothing happens unless asked.
    pub enable: bool,
    /// Output ceiling in dBTP.
    pub ceiling_db: f64,
    /// Over-sampling ratio of the side chain and of the post-limiter measurement.
    pub ratio: Ratio,
    /// Requested lookahead in ms, clamped to [`MIN_LOOKAHEAD_MS`]..=[`MAX_LOOKAHEAD_MS`].
    ///
    /// This is the *transient* stage's lookahead, and only its: a peak is not protected by
    /// the slow stage no matter how long its attack is.
    pub lookahead_ms: f64,
    /// How fast the slow stage's gain reduction arrives, ms.
    ///
    /// Short attack means the slow stage sets in sooner and takes more of the level; long
    /// attack means it arrives after the thing it was reacting to, which is another way of
    /// saying it does less. Nothing about this protects a peak — see [`Cfg::lookahead_ms`].
    pub attack_ms: f64,
    /// How fast the slow stage lets go, ms.
    pub release_ms: f64,
    /// Channel linking of the transient stage, 0.0 to 1.0.
    ///
    /// 1.0 is one gain for every channel, 0.0 is each channel for itself. Between the two
    /// the two gains are mixed in the dB domain, so the control does what it says at every
    /// value rather than only at the ends.
    pub link_transient: f64,
    /// Channel linking of the slow stage, 0.0 to 1.0. Same blending as
    /// [`Cfg::link_transient`], and the one worth turning down first, because a slow gain
    /// movement shared across the image moves the image.
    pub link_release: f64,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            // On by default, unlike every other stage in here, because a limiter with a
            // ceiling below full scale cannot make a file louder than it was and the
            // default output is a delivery. The cost is stated in the README: a default
            // render is not bit-transparent, and it is `latency_samples` longer than the
            // source. `enable: false` still gives the bit-exact path, hash and all.
            enable: true,
            // -3 dBTP: below the -1 that a delivery spec asks for, with room for the
            // ~0.1 dB of new inter-sample peak that the gain movement itself creates.
            ceiling_db: DEFAULT_CEILING_DB,
            // x8, not the meter's x4: the printed ITU bank has no transition band and no
            // peak refinement, and reads up to 0.6 dB low on a crest between grid points.
            // See the table in `crate::tp`.
            ratio: DEFAULT_RATIO,
            lookahead_ms: DEFAULT_LOOKAHEAD_MS,
            attack_ms: DEFAULT_ATTACK_MS,
            release_ms: DEFAULT_RELEASE_MS,
            link_transient: DEFAULT_LINK_TRANSIENT,
            link_release: DEFAULT_LINK_RELEASE,
        }
    }
}

impl Cfg {
    /// Validate, with an error that says what to type instead.
    pub fn validate(&self) -> Result<(), String> {
        if !self.enable {
            return Ok(());
        }
        if !(self.ceiling_db.is_finite()
            && self.ceiling_db >= MIN_CEILING_DB
            && self.ceiling_db <= MAX_CEILING_DB)
        {
            return Err(format!(
                "true-peak ceiling must be between {MIN_CEILING_DB} and \
                 {MAX_CEILING_DB} dBTP (got {})",
                self.ceiling_db
            ));
        }
        if !(self.lookahead_ms.is_finite()
            && self.lookahead_ms >= MIN_LOOKAHEAD_MS
            && self.lookahead_ms <= MAX_LOOKAHEAD_MS)
        {
            return Err(format!(
                "lookahead must be between {MIN_LOOKAHEAD_MS} and {MAX_LOOKAHEAD_MS} ms \
                 (got {self})",
                self = self.lookahead_ms
            ));
        }
        for (name, v, lo, hi) in [
            ("attack", self.attack_ms, MIN_ATTACK_MS, MAX_ATTACK_MS),
            ("release", self.release_ms, MIN_RELEASE_MS, MAX_RELEASE_MS),
        ] {
            if !(v.is_finite() && v >= lo && v <= hi) {
                return Err(format!("{name} must be between {lo} and {hi} ms (got {v})"));
            }
        }
        for (name, v) in [
            ("transient link", self.link_transient),
            ("release link", self.link_release),
        ] {
            if !(v.is_finite() && (0.0..=1.0).contains(&v)) {
                return Err(format!(
                    "{name} must be between 0 and 1 (got {v}; it is a fraction, so 75 % is 0.75)"
                ));
            }
        }
        Ok(())
    }

    /// One-pole coefficient for a time constant of `ms` at `rate`.
    ///
    /// Written as `exp(-1/tau_samples)` and applied as `v += (target - v) * (1 - coef)`,
    /// which is the form that reaches `target` instead of creeping towards it, and whose
    /// step response is the stated time constant rather than something near it.
    #[must_use]
    pub fn coef(&self, ms: f64, rate: u32) -> f64 {
        (-1.0 / (ms * 1e-3 * rate as f64)).exp()
    }

    /// Frames of the [`RELEASE_WINDOW_MS`] peak-hold the slow stage measures over.
    #[must_use]
    pub fn release_window(&self, rate: u32) -> usize {
        (RELEASE_WINDOW_MS * 1e-3 * rate as f64).round() as usize
    }

    /// Lookahead in samples before the floor is applied.
    #[must_use]
    pub fn requested_lookahead_samples(&self, rate: u32) -> usize {
        (self.lookahead_ms * 1e-3 * rate as f64).round() as usize
    }

    /// Floor: the interpolator's memory plus the shortest ramp that is not a step.
    ///
    /// Two things have to fit in the lookahead before the gain is even allowed to try
    /// for the ceiling:
    ///
    /// * the interpolation filter's own memory (`delta`) — a peak's interpolated value
    ///   is a weighted sum over about that many input samples either side, and none of
    ///   it is measurable until the filter has produced it;
    /// * a ramp of at least two frames, because a one-frame gain change is a step and
    ///   a step in the gain is a click in the output.
    ///
    /// [`Cfg::ramp_frames`] then takes whatever is left, so the schedule can always
    /// reach the gain it needs in time — that is the arithmetic the ceiling guarantee
    /// rests on, not a hope.
    #[must_use]
    pub fn min_lookahead_samples(&self) -> usize {
        self.ratio.report_delay() + MIN_RAMP_FRAMES + 1
    }

    /// Lookahead actually used, after the floor.
    #[must_use]
    pub fn lookahead_samples(&self, rate: u32) -> usize {
        self.requested_lookahead_samples(rate)
            .max(self.min_lookahead_samples())
            .max(1)
    }

    /// How far below the requested ceiling the gain aims: what the side chain's filter
    /// can fail to see, plus what a moving gain adds on top. See
    /// [`GAIN_SLEW_GUARD_DB`] for both halves and the table of what each ratio costs.
    #[must_use]
    pub fn guard_db(&self) -> f64 {
        self.ratio.worst_underread_db() + GAIN_SLEW_GUARD_DB
    }

    /// How much further past the ceiling a file can land when its content is **not**
    /// band-limited to [`crate::tp::BAND_EDGE`]: the full-band under-read minus the part
    /// [`Cfg::guard_db`] already covers.
    ///
    /// The guard is built from the error measured for content up to 0.45 fs, which is all
    /// of real programme material (a 48 kHz file with energy above 21.6 kHz is noise or a
    /// test signal). Between 0.45 fs and fs/2 no finite meter is accurate, and at fs/2
    /// the samples do not determine the peak, so a ceiling is honest only as
    /// `ceiling (+ caveat for such content)`. The report states the number rather than
    /// letting `post_dbtp <= ceiling` imply more than it can know.
    #[must_use]
    pub fn band_caveat_db(&self) -> f64 {
        (self.ratio.fullband_underread_db() - self.ratio.worst_underread_db()).max(0.0)
    }

    /// Frames over which the gain is allowed to fall, and the reason lookahead is
    /// audible: the ramp is whatever the lookahead can spare after the interpolator's
    /// memory, capped at [`ATTACK_MS`].
    ///
    /// `lookahead - delta` is exactly the room the schedule has to bring the gain down
    /// before a peak's protection window opens, so using all of it is both the
    /// smoothest choice and the only one that is always fast enough.
    #[must_use]
    pub fn ramp_frames(&self, rate: u32) -> usize {
        let la = self.lookahead_samples(rate);
        // One frame less than the room, and the reason is a proof rather than a
        // preference: the schedule for an emitted frame is exact only if every
        // neighbourhood value that can still deepen it is already known, and that
        // leaves `lookahead - delta - 1` frames of ramp.
        let room = la
            .saturating_sub(self.ratio.report_delay())
            .saturating_sub(1);
        let preferred = (ATTACK_MS * 1e-3 * rate as f64).round() as usize;
        room.max(MIN_RAMP_FRAMES)
            .min(preferred.max(MIN_RAMP_FRAMES))
    }

    /// The lookahead the processor really has, in ms. Differs from the request when
    /// the interpolator or the attack ramp needed more room, and `--dry-run` prints
    /// both so the difference is never hidden.
    #[must_use]
    pub fn effective_lookahead_ms(&self, rate: u32) -> f64 {
        self.lookahead_samples(rate) as f64 / rate as f64 * 1e3
    }

    /// True when the requested lookahead had to be raised.
    #[must_use]
    pub fn lookahead_was_raised(&self, rate: u32) -> bool {
        self.lookahead_samples(rate) > self.requested_lookahead_samples(rate)
    }

    /// Total added latency in samples: lookahead plus the interpolator's delay,
    /// because the gain for a sample comes from a peak the detector emits late.
    #[must_use]
    pub fn latency_samples(&self, rate: u32) -> usize {
        self.lookahead_samples(rate) + self.ratio.report_delay()
    }
}

/// What one file's limiter pass measured.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Report {
    /// True peak at the limiter input, dBTP.
    pub pre_dbtp: f64,
    /// True peak at the limiter output, dBTP — measured, not computed.
    pub post_dbtp: f64,
    /// Largest gain reduction applied, in dB (positive number, 0 when untouched).
    pub max_gain_reduction_db: f64,
    /// Ceiling in force, dBTP.
    pub ceiling_db: f64,
    /// How far below that ceiling the gain aims, and why: meter error plus gain-slew
    /// error, per [`GAIN_SLEW_GUARD_DB`]. A file limited at -1 dBTP with `guard_db` of
    /// 0.72 is expected to measure around -1.72, and that is the point, not a bug.
    pub guard_db: f64,
    /// Lookahead in force, ms.
    pub effective_lookahead_ms: f64,
    /// Lookahead the user asked for, ms.
    pub requested_lookahead_ms: f64,
    /// Added latency, ms.
    pub latency_ms: f64,
    /// Added latency, samples.
    pub latency_samples: usize,
    /// Frames the gain ramp takes in force.
    pub ramp_frames: usize,
    /// Slow stage attack in force, ms.
    #[serde(default)]
    pub attack_ms: f64,
    /// Slow stage release in force, ms.
    #[serde(default)]
    pub release_ms: f64,
    /// Transient stage channel linking in force, 0..=1.
    #[serde(default)]
    pub link_transient: f64,
    /// Slow stage channel linking in force, 0..=1.
    #[serde(default)]
    pub link_release: f64,
    /// Largest reduction taken by the fast stage on its own, dB.
    #[serde(default)]
    pub transient_reduction_db: f64,
    /// Largest reduction taken by the slow stage on its own, dB.
    #[serde(default)]
    pub release_reduction_db: f64,
    /// Frames the slow stage was holding gain below unity — the number that says how
    /// much of this file was level-managed rather than peak-protected.
    #[serde(default)]
    pub release_frames: u64,
    /// Side-chain ratio.
    pub ratio: Ratio,
    /// Frames of gain history observed (0 when the limiter never engaged).
    pub engaged_frames: u64,
    /// Extra dB by which full-band content (energy above 0.45 fs) can exceed the ceiling.
    /// See [`Cfg::band_caveat_db`].
    #[serde(default)]
    pub band_caveat_db: f64,
}

impl Report {
    /// Did the limiter's **own detector** read its output at or below the ceiling?
    ///
    /// This is the control loop grading itself: the gain was chosen with this detector and
    /// `post_dbtp` is measured with it, so a `true` here can only confirm the limiter's
    /// belief. It is a unit-test oracle and a sanity check, not evidence. The delivered
    /// stream and file are measured by independent meters, reported as
    /// `pipeline::Report::true_peak_dbtp` and `pipeline::Report::delivered`.
    #[must_use]
    pub fn within_ceiling(&self) -> bool {
        self.post_dbtp <= self.ceiling_db
    }
}

/// Streaming true-peak limiter over planar blocks of one group.
///
/// The gain schedule is computed in three passes over a bounded ring, so memory does
/// not grow with file length and the result does not depend on how the input is cut
/// into blocks:
///
/// 1. **required gain** per input frame, from the side-chain peak the detector reports
///    for that frame;
/// 2. **neighbourhood minimum** over `±delta` frames, because a peak's interpolated
///    value is built from that many samples either side and every one of them is
///    scaled by the gain of its own frame;
/// 3. **backward ramp pass** `g[j] = min(g_a[j], g[j + 1] + slope)`, which is what
///    makes the gain *arrive* at the required depth instead of stepping there, and is
///    the reason no overshoot is possible rather than merely unlikely. It is evaluated
///    at emission as its closed form over the ramp horizon,
///    `g[j] = min over r in 0..=ramp of (g_a[j + r] + r * slope)`, because the frames that
///    constrain `g[j]` are ahead of it and are finalised one frame per input frame — a
///    running accumulator inside the scheduler could not see them. See
///    [`Limiter::schedule_to`] for what that cost when it was tried the other way.
///
/// A one-pole release follower runs forward at emission, and only ever pulls the gain
/// *down* toward the schedule, so it cannot break the ceiling.
pub struct Limiter {
    cfg: Cfg,
    rate: u32,
    /// Side-chain detector per channel.
    side: Vec<Detector>,
    /// Post-limiter measurement, same ratio as the side chain.
    out_meter: TruePeak,
    /// Audio delay lines, one per channel, length `latency`.
    lines: Vec<Vec<f64>>,
    /// Write/read cursor in the delay lines.
    pos: usize,
    /// Total latency: lookahead + interpolator memory.
    latency: usize,
    /// Interpolator memory in frames: how far either side of a peak has to be covered.
    delta: usize,
    /// Frames per slot ring.
    ring_len: usize,
    /// One frame of scratch, reused so `process` allocates nothing per block.
    scratch: Vec<f64>,
    /// Required transient gain per input frame per channel, and the scheduled gain, on
    /// the same ring.
    req_ring: Vec<Vec<f64>>,
    sched_ring: Vec<Vec<f64>>,
    /// Slow stage gain per input frame, on the same ring, so the frame that is emitted
    /// gets the slow gain that belonged to it rather than the one in force now.
    gr_ring: Vec<Vec<f64>>,
    /// Frames fed to the side chain, and the first input frame not yet scheduled.
    fed: u64,
    scheduled: u64,
    /// Per-channel transient gain, after the return-to-unity follower.
    gain: Vec<f64>,
    /// Per-channel slow gain, following its own attack and release.
    g_rel: Vec<f64>,
    /// Slow gain for the group, before linking.
    g_rel_group: f64,
    /// Slow stage level detector: the peak-hold, and frames left in its window.
    hold: Vec<f64>,
    hold_left: usize,
    /// What the level detector demands of the slow stage, before attack and release.
    g_target: Vec<f64>,
    g_target_group: f64,
    /// Linked slow gain of the frame being fed, reused so the step allocates nothing.
    gr_now: Vec<f64>,
    /// One-pole coefficient for the transient stage's return to unity.
    release_coef: f64,
    /// Slow stage coefficients, and the hold window length.
    atk_coef: f64,
    slow_rel_coef: f64,
    release_window: usize,
    /// Scratch for the per-frame peaks, reused so the step allocates nothing.
    peaks: Vec<f64>,
    /// Fastest gain change allowed per frame, in linear terms.
    slope: f64,
    /// Frames the ramp above is allowed to reach forward over. Kept next to the slope it
    /// is the reciprocal of, so the two cannot disagree about how long the ramp is.
    ramp: usize,
    /// Per channel: the ramped schedule target for the frame being emitted.
    ramp_target: Vec<f64>,
    /// Amplitude the gain aims at — the ceiling less [`Cfg::guard_db`].
    ceiling_lin: f64,
    report: Report,
    engaged: bool,
}

/// Mix two gains in the dB domain: `w` of `b`, the remainder of `a`.
///
/// Mixed in the log domain so that 0.5 means half the *reduction* rather than half the
/// amplitude, and so that 1.0 is exactly `b` with no residue of `a` left in it. Both
/// inputs are at most 1.0, so the result never asks for a boost.
fn blend(a: f64, b: f64, w: f64) -> f64 {
    let la = a.max(f64::MIN_POSITIVE).ln();
    let lb = b.max(f64::MIN_POSITIVE).ln();
    (la + (lb - la) * w).exp()
}

impl Limiter {
    /// A limiter that does nothing and adds no delay. Cheaper than configuring one
    /// off, and it is what the pipeline uses when `enable` is false.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            // Explicitly off, not `Cfg::default()`: the default is on now, and a limiter
            // built to do nothing has to say so in its own config too, because `process`
            // and `flush` both read `enable` before they touch state that was never sized.
            cfg: Cfg {
                enable: false,
                ..Cfg::default()
            },
            rate: 48_000,
            side: Vec::new(),
            out_meter: TruePeak::new(Ratio::X4, 0),
            lines: Vec::new(),
            pos: 0,
            latency: 0,
            delta: 0,
            ring_len: 0,
            scratch: Vec::new(),
            ramp: 0,
            ramp_target: Vec::new(),
            req_ring: Vec::new(),
            sched_ring: Vec::new(),
            gr_ring: Vec::new(),
            fed: 0,
            scheduled: 0,
            gain: Vec::new(),
            g_rel: Vec::new(),
            g_rel_group: 1.0,
            hold: Vec::new(),
            hold_left: 0,
            g_target: Vec::new(),
            g_target_group: 1.0,
            gr_now: Vec::new(),
            peaks: Vec::new(),
            atk_coef: 0.5,
            slow_rel_coef: 0.5,
            release_window: 1,
            release_coef: 1.0,
            slope: 1.0,
            ceiling_lin: 1.0,
            report: Report::default(),
            engaged: false,
        }
    }

    #[must_use]
    pub fn enabled(cfg: Cfg, rate: u32, channels: usize) -> Self {
        let delta = cfg.ratio.report_delay();
        let _lookahead = cfg.lookahead_samples(rate);
        let latency = cfg.latency_samples(rate);
        let ramp = cfg.ramp_frames(rate);
        // Every live index needs a slot: the emission cursor, the scheduling cursor
        // `delta + ramp` behind it, and the neighbourhood read `delta` ahead of the
        // scheduling cursor. Rounded up with room to spare.
        let ring_len = (latency + 2 * delta + ramp + 8).max(16);
        Self {
            cfg,
            rate,
            side: (0..channels).map(|_| Detector::new(cfg.ratio)).collect(),
            out_meter: TruePeak::new(cfg.ratio, channels),
            lines: vec![vec![0.0; latency.max(1)]; channels.max(1)],
            pos: 0,
            latency,
            delta,
            ring_len,
            scratch: Vec::with_capacity(channels),
            req_ring: vec![vec![1.0; ring_len]; channels.max(1)],
            sched_ring: vec![vec![1.0; ring_len]; channels.max(1)],
            gr_ring: vec![vec![1.0; ring_len]; channels.max(1)],
            fed: 0,
            scheduled: 0,
            gain: vec![1.0; channels.max(1)],
            g_rel: vec![1.0; channels.max(1)],
            g_rel_group: 1.0,
            hold: vec![0.0; channels.max(1)],
            hold_left: cfg.release_window(rate).max(1),
            g_target: vec![1.0; channels.max(1)],
            g_target_group: 1.0,
            gr_now: vec![1.0; channels.max(1)],
            peaks: vec![0.0; channels.max(1)],
            atk_coef: cfg.coef(cfg.attack_ms, rate),
            slow_rel_coef: cfg.coef(cfg.release_ms, rate),
            release_window: cfg.release_window(rate).max(1),
            // One-pole return to unity for the fast stage: g += (1 - g) * (1 - coef).
            release_coef: cfg.coef(TRANSIENT_RELEASE_MS, rate),
            slope: 1.0 / ramp as f64,
            ramp,
            ramp_target: vec![1.0; channels.max(1)],
            ceiling_lin: 10f64.powf((cfg.ceiling_db - cfg.guard_db()) / 20.0),
            report: Report {
                ceiling_db: cfg.ceiling_db,
                guard_db: cfg.guard_db(),
                effective_lookahead_ms: cfg.effective_lookahead_ms(rate),
                requested_lookahead_ms: cfg.lookahead_ms,
                latency_ms: latency as f64 / rate as f64 * 1e3,
                latency_samples: latency,
                ramp_frames: ramp,
                attack_ms: cfg.attack_ms,
                release_ms: cfg.release_ms,
                link_transient: cfg.link_transient,
                link_release: cfg.link_release,
                ratio: cfg.ratio,
                pre_dbtp: f64::NEG_INFINITY,
                post_dbtp: f64::NEG_INFINITY,
                max_gain_reduction_db: 0.0,
                transient_reduction_db: 0.0,
                release_reduction_db: 0.0,
                release_frames: 0,
                engaged_frames: 0,
                band_caveat_db: cfg.band_caveat_db(),
            },
            engaged: false,
        }
    }

    /// Sample rate the limiter was built for.
    #[must_use]
    pub fn rate(&self) -> u32 {
        self.rate
    }

    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.cfg.enable
    }

    /// Added latency in samples, for the report and the GUI.
    #[must_use]
    pub fn latency_samples(&self) -> usize {
        self.latency
    }

    /// Post-limiter true peak per channel, dBTP.
    ///
    /// [`Report::post_dbtp`] is the group maximum, because that is what the ceiling is
    /// judged against — the ceiling holds for every channel, so one number has to hold.
    /// A meter panel wants the breakdown, and the difference is not cosmetic: on 5.1
    /// material the group maximum and the loudest single channel can disagree by several
    /// dB once the LFE is in the group.
    #[must_use]
    pub fn per_channel_post_dbtp(&self) -> Vec<f64> {
        self.out_meter.per_channel_dbtp()
    }

    #[must_use]
    pub fn report(&self) -> Report {
        let mut r = self.report;
        r.pre_dbtp =
            crate::tp::db_from_peak(self.side.iter().map(Detector::peak).fold(0.0, f64::max));
        r.post_dbtp = self.out_meter.dbtp();
        r
    }

    /// Transient gain channel `c` needs for the peak just reported, after the slow stage.
    fn required(&self, c: usize) -> f64 {
        let after = (self.peaks[c] * self.gr_now[c]).max(f64::MIN_POSITIVE);
        if after > self.ceiling_lin {
            self.ceiling_lin / after
        } else {
            1.0
        }
    }

    /// Slot for an input-frame index on the ring.
    fn slot(&self, j: u64) -> usize {
        (j % self.ring_len as u64) as usize
    }

    /// Passes 2 and 3 for every input frame up to (but excluding) `limit`.
    ///
    /// Pass 3 is a *backwards* recurrence — `g[j] = min(g_a[j], g[j + 1] + slope)` —
    /// so the segment is walked right to left and seeded with the value already
    /// scheduled at `limit`, which was finalised on an earlier call. Scheduling a
    /// segment in increasing order would read a slot that has not been written yet,
    /// which is exactly the kind of error a ring buffer makes easy and a test for
    /// chunk-size invariance makes impossible to keep.
    fn schedule_to(&mut self, limit: u64) {
        if limit <= self.scheduled {
            return;
        }
        let delta = self.delta as u64;
        for c in 0..self.req_ring.len() {
            for m in (self.scheduled..limit).rev() {
                // Pass 2: the gain has to sit at the required depth across the whole
                // neighbourhood the peak is built from, not just at its own frame.
                let lo = m.saturating_sub(delta);
                // The slow gain at this frame, against which every neighbour's demand is
                // re-based. See the comment on the loop below.
                let gr_m = self.gr_ring[c][self.slot(m)].max(f64::MIN_POSITIVE);
                let mut ga = 1.0f64;
                for k in lo..=(m + delta) {
                    let kslot = self.slot(k);
                    // A crest is not one frame. The reading at `m` is built from the samples
                    // in `[m-delta, m+delta]`, and each of those samples is multiplied by
                    // *its own frame's* combined gain, so a demand computed against frame `k`
                    // only protects `k` if every neighbour is scaled no harder than `m` is.
                    // Where the slow stage is higher than it is at `k` — earlier frames during
                    // a fast attack, later ones during a fast release — the fast stage has to
                    // make up the difference itself, by exactly this ratio. At the 1 ms attack
                    // this module accepts the slow stage can travel 3.27 dB across the window,
                    // which no guard band worth having would cover; widening the guard instead
                    // would tax every file, including the ones where the envelope barely
                    // moves, and this costs 0.00 dB when the ratio is 1.
                    let scale = (self.gr_ring[c][kslot] / gr_m).min(1.0);
                    let v = self.req_ring[c][kslot] * scale;
                    if v < ga {
                        ga = v;
                    }
                }
                // This is pass 2's answer, stored as the demand for the frame. The slope
                // is NOT applied here: this routine is called once per input frame and
                // finalises exactly one frame per call, so a running `acc` seeded at 1.0 on
                // entry could only ever compare the demand against `1.0 + slope` — which
                // every demand in [0, 1] loses to, and the pass silently became the identity
                // while still writing a value that looked like a ramp. The ramp is applied
                // where it has to be: at emission, looking forward into the schedule, in
                // [`Limiter::advance_gain`]'s caller.
                let slot = self.slot(m);
                self.sched_ring[c][slot] = ga;
                if ga < 1.0 - ENGAGED_EPS {
                    let gr = -20.0 * ga.max(f64::MIN_POSITIVE).log10();
                    if gr > self.report.transient_reduction_db {
                        self.report.transient_reduction_db = gr;
                    }
                }
            }
        }
        self.scheduled = limit;
    }

    /// The release follower: fall instantly onto the schedule (which is itself
    /// slope-limited, so this is not a step), rise one-pole toward unity.
    /// Pass 3: the ramped schedule target for the frame about to be emitted.
    ///
    /// The gain at frame `j` may not fall faster than `slope` per frame, and the only values
    /// that know that are the demands at `j+1 … j+ramp`: a depth three frames ahead has to
    /// be arrived at from behind, one slope at a time. Reading forward is legal because of
    /// an inequality the config proves rather than hopes — `latency = lookahead + report
    /// delay`, and the ramp is what the lookahead has left after the delay
    /// ([`Cfg::ramp_frames`]) — so every slot reached here was finalised by
    /// [`Self::schedule_to`] some frames ago.
    ///
    /// A demand past the last finalised frame is "nothing", which is what it means at end of
    /// file: there is no input left to protect. Deepening the gain for a future that never
    /// arrives is the mistake that closes.
    fn ramp_schedule(&mut self, j: u64, live: bool) {
        let ramp = self.ramp as u64;
        let slope = self.slope;
        let ring_len = self.ring_len as u64;
        let scheduled = self.scheduled;
        for c in 0..self.ramp_target.len() {
            let mut target = 1.0f64;
            if live {
                for r in 0..=ramp {
                    let m = j + r;
                    let demand = if m < scheduled {
                        self.sched_ring[c][(m % ring_len) as usize]
                    } else {
                        1.0
                    };
                    let v = demand + (r as f64) * slope;
                    if v < target {
                        target = v;
                    }
                }
            }
            self.ramp_target[c] = target.clamp(0.0, 1.0);
        }
    }

    fn advance_gain(&mut self, c: usize, target: f64) {
        let g = self.gain[c];
        if target < g {
            self.gain[c] = target;
            return;
        }
        let next = (g + (1.0 - g) * (1.0 - self.release_coef)).min(target);
        // Same arrival rule as the slow stage: a release that creeps to within 2e-11 dB of
        // unity and stops there is a release that never ended, and every counter downstream
        // of it says "still limiting" for the rest of the file.
        self.gain[c] = if target - next < GAIN_DEADBAND {
            target
        } else {
            next
        };
    }

    /// One frame in, one (delayed) frame out. `frame` holds one input sample per channel
    /// and is overwritten with the output sample per channel.
    ///
    /// This is the whole per-frame algorithm. [`Limiter::process`] calls it once per input
    /// frame and [`Limiter::flush`] calls it once per frame of tail with a zero input, so
    /// the end of a file takes exactly the path the middle of it does. The previous
    /// `flush` re-implemented the delay line by hand and never pushed the last `delta`
    /// input frames through the side chain, so a peak in the final few milliseconds was
    /// never scheduled for and could leave the file above the ceiling.
    /// One frame through the side chain and the delay line.
    ///
    /// Two stages, as every serious limiter has: a fast one that has to be at depth by the
    /// time a peak arrives, and a slow one that responds to level over tenths of a second.
    /// They multiply, and the fast one is handed the *residual* need — what the slow stage
    /// has not already taken — so a loud passage is managed once instead of twice.
    fn step(&mut self, frame: &mut [f64]) {
        let nch = frame.len();
        let latency = self.latency;

        // --- pass 1: per-channel true peak of the frame the side chain reports ---------
        // Measured per channel, because the two link controls exist to decide how much of
        // one channel's trouble becomes the others'.
        for (side, (peak, x)) in self
            .side
            .iter_mut()
            .zip(self.peaks.iter_mut().zip(frame.iter()))
        {
            *peak = side.push(*x);
        }
        self.fed += 1;

        // --- slow stage: a peak-hold window, then attack and release on the gain -------
        // The window is what makes this a level detector rather than a second peak
        // detector. The followers run every frame and not once per window, so the time
        // constants on the panel are the time constants rather than those divided by the
        // window length.
        for c in 0..nch {
            if self.peaks[c] > self.hold[c] {
                self.hold[c] = self.peaks[c];
            }
        }
        self.hold_left = self.hold_left.saturating_sub(1);
        if self.hold_left == 0 {
            self.hold_left = self.release_window.max(1);
            let aim_db = 20.0 * self.ceiling_lin.max(f64::MIN_POSITIVE).log10();
            let mut group = f64::NEG_INFINITY;
            for c in 0..nch {
                let lvl = 20.0 * self.hold[c].max(f64::MIN_POSITIVE).log10();
                if lvl > group {
                    group = lvl;
                }
                // Reduction the level demands, never a boost: the ceiling is a limit, and
                // a slow stage that could raise the level is an expander wearing its clothes.
                self.g_target[c] = 10f64.powf((aim_db - lvl).min(0.0) / 20.0);
                self.hold[c] = 0.0;
            }
            self.g_target_group = 10f64.powf((aim_db - group).min(0.0) / 20.0);
        }
        let atk = 1.0 - self.atk_coef;
        let rel = 1.0 - self.slow_rel_coef;
        for c in 0..nch {
            let t = self.g_target[c];
            let g = self.g_rel[c];
            let next = if t < g {
                g + (t - g) * atk
            } else {
                g + (t - g) * rel
            };
            // Snap on arrival — see [`GAIN_DEADBAND`]. Applied to the follower and not to
            // the arithmetic it converges towards, so a target the follower is merely
            // approaching is untouched.
            self.g_rel[c] = if (t - next).abs() < GAIN_DEADBAND {
                t
            } else {
                next
            };
        }
        let tg = self.g_target_group;
        let gg = self.g_rel_group;
        let next = if tg < gg {
            gg + (tg - gg) * atk
        } else {
            gg + (tg - gg) * rel
        };
        self.g_rel_group = if (tg - next).abs() < GAIN_DEADBAND {
            tg
        } else {
            next
        };
        let mut slow_min = 1.0f64;
        for c in 0..nch {
            let gr = blend(self.g_rel[c], self.g_rel_group, self.cfg.link_release);
            self.gr_now[c] = gr;
            if gr < slow_min {
                slow_min = gr;
            }
        }
        if slow_min < 1.0 - ENGAGED_EPS {
            self.report.release_frames += 1;
            let db = -20.0 * slow_min.max(f64::MIN_POSITIVE).log10();
            if db > self.report.release_reduction_db {
                self.report.release_reduction_db = db;
            }
        }

        // --- required transient gain, on the frame the peaks actually describe ----------
        // The detector reports a peak `delta` frames after the input frame it describes.
        // The first `delta` feeds have no frame of their own, but they already read the
        // opening samples (a reading depends on the last `2 * delta` inputs at most), so
        // they are folded into frame 0's demand, whose neighbourhood spans all of them.
        // Computed *after* the slow stage and against what is left of the peak once that
        // stage has had it.
        let m = self.fed.saturating_sub(1 + self.delta as u64);
        let slot = self.slot(m);
        let mut group = 1.0f64;
        for c in 0..nch {
            let req = self.required(c);
            if req < group {
                group = req;
            }
        }
        for c in 0..nch {
            let mut req = self.required(c);
            if self.cfg.link_transient > 0.0 {
                req = blend(req, group, self.cfg.link_transient);
            }
            // Frame 0's slot still holds its construction value of 1.0 on the first feed.
            if m == 0 {
                req = req.min(self.req_ring[c][slot]);
            }
            self.req_ring[c][slot] = req;
            self.gr_ring[c][slot] = self.gr_now[c];
        }

        // A frame's schedule can still be deepened by input arriving up to `2 * delta`
        // frames later (its own neighbourhood reaches `delta` ahead, and the peak it
        // protects is reported `delta` late), so it is final only once that input is in.
        let limit = self.fed.saturating_sub(self.delta as u64 * 2);
        self.schedule_to(limit);

        // --- emit one frame: the delay line, scaled by transient gain times slow gain ---
        // The frame coming out is input frame `fed - 1 - latency`, and its schedule is
        // already final because the scheduling cursor trails the feed by `2 * delta` while
        // the emission cursor trails it by `latency`, larger by construction.
        let j = self.fed.saturating_sub(latency as u64 + 1);
        let jslot = self.slot(j);
        let live = self.fed > latency as u64;
        // Pass 3, the backward ramp, evaluated here. The gain at frame j may not fall
        // faster than `slope` per frame, and the only values that know that are the demands
        // at j+1 … j+ramp: a depth two frames ahead has to be arrived at from behind, one
        // slope at a time. Reading forward is legal because of an inequality the config
        // proves rather than hopes — `latency = lookahead + report delay`, and the ramp is
        // what the lookahead has left after the delay ([`Cfg::ramp_frames`]), so every slot
        // reached here has already been finalised by `schedule_to`.
        //
        // A demand beyond the last finalised frame is treated as "nothing", which is what
        // it means at end of file: there is no input left to protect. Deepening the gain
        // for a future that never arrives is the mistake this closes.
        self.ramp_schedule(j, live);
        for c in 0..nch {
            // Read into a local first: the value is Copy, and holding a borrow of the
            // vector across `advance_gain` would make the two halves of this loop fight
            // over `self` for no reason at all.
            let t = self.ramp_target[c];
            self.advance_gain(c, t);
        }
        let slot = self.pos;
        let mut deepest = 1.0f64;
        // Zipped rather than indexed, but still strictly ascending in channel order: the
        // order is not an optimisation detail here, it is how the linked gains were
        // combined a few lines above, and the two have to agree.
        let Self {
            gain,
            gr_ring,
            lines,
            ..
        } = self;
        for ((g, ring), (line, buf)) in gain
            .iter()
            .zip(gr_ring.iter())
            .zip(lines.iter_mut().zip(frame.iter_mut()))
        {
            let g = *g * ring[jslot];
            let old = line[slot];
            line[slot] = *buf;
            *buf = old * g;
            if g < deepest {
                deepest = g;
            }
        }
        if deepest < 1.0 - ENGAGED_EPS {
            if !self.engaged {
                self.engaged = true;
            }
            self.report.engaged_frames += 1;
            let db = -20.0 * deepest.max(f64::MIN_POSITIVE).log10();
            if db > self.report.max_gain_reduction_db {
                self.report.max_gain_reduction_db = db;
            }
        }
        self.pos = if slot + 1 >= latency.max(1) {
            0
        } else {
            slot + 1
        };
    }

    /// Process one block of planar samples in place. Every channel in `planar` must be
    /// the same length.
    ///
    /// One mutable slice per channel, all the same length, processed in place. Slices
    /// rather than `Vec`s because the render loop hands over a window of buffers it
    /// already owns; making it copy a block to satisfy the signature would be pure cost
    /// on the one path that has to stay ahead of a live device.
    pub fn process(&mut self, planar: &mut [&mut [f64]]) {
        if !self.cfg.enable || planar.is_empty() {
            return;
        }
        let n = planar[0].len();
        debug_assert!(
            planar.iter().all(|p| p.len() == n),
            "ragged block into the limiter"
        );
        let nch = self.side.len().min(planar.len());
        let mut frame = std::mem::take(&mut self.scratch);
        frame.clear();
        frame.resize(nch, 0.0);
        for i in 0..n {
            for (f, buf) in frame.iter_mut().zip(planar.iter()) {
                *f = buf[i];
            }
            self.step(&mut frame);
            for (f, buf) in frame.iter().zip(planar.iter_mut()) {
                buf[i] = *f;
            }
        }
        self.scratch = frame;
        // Measure what came out, with the detector the ceiling was judged by. This is the
        // control loop's own belief about its output; the pipeline measures the delivered
        // stream with a different instrument (see `pipeline::PeakStage`).
        for (c, buf) in planar[..nch].iter().enumerate() {
            self.out_meter.push_ch(c, &buf[..n]);
        }
    }

    /// Drain the delay line at end of file, returning the trailing frames. They are
    /// real output — the last `latency` samples of a limited file live here — and
    /// dropping them would shorten every limited render.
    ///
    /// Runs [`Limiter::step`] on silence for `latency` frames, which pushes the last real
    /// frames through the side chain (so their peaks are scheduled for) and emits them
    /// through the same delay line and gain follower as every other frame.
    pub fn flush(&mut self) -> Vec<Vec<f64>> {
        if !self.cfg.enable {
            return Vec::new();
        }
        let nch = self.lines.len().min(self.side.len());
        let mut frame = vec![0.0f64; nch];
        let mut out = vec![Vec::with_capacity(self.latency); nch];
        for _ in 0..self.latency {
            frame.fill(0.0);
            self.step(&mut frame);
            for (o, &v) in out.iter_mut().zip(frame.iter()) {
                o.push(v);
            }
        }
        // The tail is programme, not debris: a peak can absolutely live in it, so it goes
        // through the same detector before the report is allowed to be read.
        for (c, buf) in out.iter().enumerate() {
            self.out_meter.push_ch(c, buf);
        }
        // And that detector's own memory is run out too. When the limiter's ratio equals
        // the render's meter ratio this meter *is* the file's measurement, so a crest in the
        // last frames would otherwise be missing from `post_dbtp` — the one number the
        // ceiling claim rests on.
        self.out_meter.drain();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    fn cfg(ceiling: f64, ratio: Ratio, lookahead: f64) -> Cfg {
        Cfg {
            enable: true,
            ceiling_db: ceiling,
            ratio,
            lookahead_ms: lookahead,
            ..Cfg::default()
        }
    }

    /// The ceiling is a promise, and a promise is only worth what it holds for at the
    /// settings most likely to break it: a deep cut (a lot of gain reduction to slew), the
    /// shortest lookahead the API accepts (no time to arrange the reduction in advance),
    /// and the ratios that make each decibel of overage demand the most reduction.
    ///
    /// This is the answer to "a short lookahead needs a wider guard band". It does not:
    /// every cell of this grid came out under its ceiling, so `Cfg::guard_db()` already
    /// covers the short-lookahead case and inflating it further would only make the meter
    /// pessimistic about a risk that is not there.
    ///
    /// Sabotage check: set `GAIN_SLEW_GUARD_DB` to 0 and this fails — measured at
    /// +0.0035 dB over the ceiling with lookahead 0.2 ms, ratio x32, ceiling -12 dBTP —
    /// so the guard is doing real work and 0.02 dB is enough of it. That is the whole
    /// answer to asking for a wider band: the margin is earned at the worst cell in the
    /// grid, not guessed at.
    #[test]
    fn the_ceiling_holds_across_ratios_lookahead_and_depth() {
        let mut worst = f64::NEG_INFINITY;
        let mut at = String::new();
        let mut cells = 0usize;
        for la in [MIN_LOOKAHEAD_MS, 1.0, MAX_LOOKAHEAD_MS] {
            for ratio in [Ratio::X2, Ratio::X4, Ratio::X8, Ratio::X16, Ratio::X32] {
                for &ceil in &[-1.0, -3.0, -6.0, -12.0, -18.0] {
                    let c = cfg(ceil, ratio, la);
                    for (i, m) in probe_materials().into_iter().enumerate() {
                        let (_, r) = run(c, &[m], 333);
                        cells += 1;
                        let margin = r.post_dbtp - ceil;
                        if margin > worst {
                            worst = margin;
                            at = format!(
                                "lookahead {la} ms, {ratio:?}, ceiling {ceil} dBTP, material #{i}"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            worst <= 0.0,
            "post-limiter true peak sat {worst:+.4} dB over its ceiling at {at} \
             ({cells} configurations)"
        );
    }

    fn burst(f: f64, amp: f64, n: usize, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * std::f64::consts::PI * f * i as f64 / FS + phase).sin())
            .collect()
    }

    /// Material designed to break a limiter: full-scale bursts at frequencies whose
    /// crests fall between samples, plus single-sample impulses whose interpolated
    /// neighbours are the actual peak.
    fn worst_case(n: usize) -> Vec<f64> {
        let mut v = vec![0.0f64; n];
        let tones = [1_000.0f64, 6_000.0, 11_000.0, 12_000.0, 15_500.0, 23_000.0];
        for (k, &f) in tones.iter().enumerate() {
            let seg = n / tones.len();
            let start = k * seg;
            let end = ((k + 1) * seg).min(n);
            let x = burst(f, 1.0, end - start, k as f64 * 0.37);
            v[start..end].copy_from_slice(&x);
        }
        // Impulses: a single sample at full scale has interpolated neighbours that
        // a sample-peak-only limiter never sees.
        for i in (0..n).step_by(997) {
            v[i] = 1.0;
        }
        v
    }

    fn run(cfg: Cfg, planar: &[Vec<f64>], chunk: usize) -> (Vec<Vec<f64>>, Report) {
        let mut lim = Limiter::enabled(cfg, FS as u32, planar.len());
        let mut out = vec![Vec::new(); planar.len()];
        // The limiter works in place, so the fixture keeps its own copy to feed it.
        let mut data: Vec<Vec<f64>> = planar.to_vec();
        let mut i = 0usize;
        while i < data[0].len() {
            let end = (i + chunk).min(data[0].len());
            let mut block: Vec<&mut [f64]> = data.iter_mut().map(|d| &mut d[i..end]).collect();
            lim.process(&mut block);
            for (c, b) in block.iter().enumerate() {
                out[c].extend_from_slice(b);
            }
            i = end;
        }
        let tail = lim.flush();
        for (c, t) in tail.iter().enumerate() {
            out[c].extend_from_slice(t);
        }
        let r = lim.report();
        (out, r)
    }

    /// Materials that move the slow stage hard.
    ///
    /// The slew that #15 is about only exists while the release envelope is climbing, so the
    /// fixtures are a level step with crests arriving *through* the climb, not a steady
    /// level. A file at one loudness gives the slow stage nothing to move, which is how a
    /// worst case of 3.27 dB across the kernel window can sit in a green suite.
    fn probe_materials() -> Vec<Vec<f64>> {
        // Loud, then quiet with a full-scale crest every 8 frames: the release is letting
        // gain back exactly when the crests arrive.
        let mut step_down = vec![0.9f64; 2000];
        step_down.extend(std::iter::repeat_n(0.0, 2600));
        for i in (2000..2600).step_by(8) {
            step_down[i] = 1.0;
        }
        // The other direction: silence, then level with crests on top of it, so the attack
        // is digging while the same crests are being scheduled.
        let mut step_up = vec![0.0f64; 4096];
        for s in step_up.iter_mut().skip(2000) {
            *s = 0.9;
        }
        for i in (2000..2600).step_by(8) {
            step_up[i] = 1.0;
        }
        vec![worst_case(4096), step_down, step_up]
    }

    /// The fastest attack and release the interface accepts still hold the ceiling.
    ///
    /// A crest is not one sample: the interpolated value at frame `m` is built from the
    /// samples in `[m-delta, m+delta]`, and each of those samples is multiplied by *its own
    /// frame's* combined gain. So the slow stage only has to be steady across about 26 frames
    /// for the ceiling to mean anything — and at the 1 ms attack and 10 ms release this module
    /// accepts, it travels 3.27 dB in that window, which is more than any guard band worth
    /// having. The fix is not a wider guard: that taxes every file, including the ones where
    /// the slow stage barely moves. The demand for a frame is scaled by the slow gain of the
    /// frames around it, which costs nothing when they are equal and exactly what is needed
    /// when they are not.
    /// The meter and the limiter start at different ratios on purpose, and nothing may
    /// quietly make them the same again: the difference is 0.58 dB of ceiling.
    #[test]
    fn the_default_side_chain_is_deeper_than_the_default_meter() {
        assert_eq!(Ratio::default(), Ratio::X4, "the meter's default moved");
        assert_eq!(DEFAULT_RATIO, Ratio::X8);
        assert_eq!(Cfg::default().ratio, DEFAULT_RATIO);

        // 0.02 dB the meter cannot see at 8x, plus 0.02 dB of gain slew. At the meter's 4x
        // the same expression is 0.62 dB, and a file limited to -3 dBTP would land near
        // -3.62 instead of -3.04: the same ceiling, asked for, half a dB further down.
        let g = Cfg::default().guard_db();
        assert!(
            (g - 0.04).abs() < 1e-12,
            "the default guard is {g} dB, not 0.04 — the README quotes that number"
        );

        let lat = Cfg::default().latency_samples(48_000);
        assert_eq!(lat, 96 + Ratio::X8.report_delay());
        assert!(lat > 96, "the interpolator's own delay went missing");
    }

    #[test]
    fn the_fastest_envelope_the_interface_allows_still_holds_the_ceiling() {
        for &(at, rel) in &[
            (MIN_ATTACK_MS, MIN_RELEASE_MS),
            (MIN_ATTACK_MS, DEFAULT_RELEASE_MS),
            (DEFAULT_ATTACK_MS, MIN_RELEASE_MS),
        ] {
            let mut c = cfg(-3.0, Ratio::X8, MIN_LOOKAHEAD_MS);
            c.attack_ms = at;
            c.release_ms = rel;
            let mut worst = (f64::NEG_INFINITY, 0.0f64, 0.0f64);
            for x in probe_materials() {
                let (_, r) = run(c, &[x], 333);
                if r.post_dbtp > worst.0 {
                    worst = (
                        r.post_dbtp,
                        r.release_reduction_db,
                        r.transient_reduction_db,
                    );
                }
            }
            assert!(
                worst.0 <= -3.0,
                "attack {at} ms / release {rel} ms delivered a file measuring {:+.3} dBTP \
                 against a -3.00 dBTP ceiling (slow stage held {} dB, transient {} dB): the \
                 slow stage moved inside the window the crest is built from, and nothing \
                 compensated for it",
                worst.0,
                worst.1,
                worst.2,
            );
        }
    }

    /// A peak that arrives in the first frames of a file still cannot beat the ceiling.
    ///
    /// Nothing precedes a crest at frame 0, so there is no history in the bank to smooth it
    /// and no earlier gain to lean on; the limiter meets it with whatever it can schedule
    /// from a cold start. The detector reads frame 0 before it has a frame to report it
    /// against, which is why frame 0 is tested separately from frame 1. The burst is
    /// Nyquist-alternating because an interpolating lowpass is in its own transition band
    /// there, which leaves the sample-peak floor as the only thing the meter can see — the
    /// hardest thing to catch, at the worst moment to catch it. The single sample is the
    /// opposite case: a crest the interpolator sees in full.
    ///
    /// Measured with a second meter rather than `r.post_dbtp`, because the limiter's own
    /// optics are the thing under test and would agree with whatever they did wrong. The
    /// allowance is the limiter's guard band plus this meter's own over-read: a ceiling is a
    /// statement about the file, and both instruments standing between the two are measured
    /// numbers, not fudge.
    #[test]
    fn a_peak_inside_the_filter_depth_is_limited_where_it_lands() {
        let mut worst = f64::NEG_INFINITY;
        let mut detail = String::new();
        for start in [0usize, 1] {
            let mut burst = vec![0.0f64; 4096];
            for (i, slot) in burst.iter_mut().enumerate().skip(start).take(8) {
                *slot = if i % 2 == 0 { -4.0 } else { 4.0 };
            }
            let mut impulse = vec![0.0f64; 4096];
            impulse[start] = 4.0;
            for (name, x) in [("burst", &burst), ("impulse", &impulse)] {
                for &ratio in Ratio::ALL.iter() {
                    for lookahead in [MIN_LOOKAHEAD_MS, DEFAULT_LOOKAHEAD_MS] {
                        let (out, r) =
                            run(cfg(-3.0, ratio, lookahead), std::slice::from_ref(x), 777);
                        let mut meter = TruePeak::new(Ratio::X8, 1);
                        meter.push_ch(0, &out[0]);
                        meter.drain();
                        let got = meter.dbtp();
                        // What the ceiling permits, in reading terms: the limiter may
                        // under-read the crest by up to its guard band and still have done
                        // its duty, and the meter doing this measurement has an error of its
                        // own in the other direction.
                        let allowed = -3.0 + r.guard_db + Ratio::X8.worst_overread_db();
                        if got - allowed > worst {
                            worst = got - allowed;
                            detail = format!(
                                "{ratio} {name} at frame {start}, lookahead {lookahead} ms: \
                                 measured {got:+.3} dBTP, allowed {allowed:+.3} (guard {:.3} \
                                 + meter over-read {:.3}), reduction {:.3} dB",
                                r.guard_db,
                                Ratio::X8.worst_overread_db(),
                                r.max_gain_reduction_db
                            );
                        }
                    }
                }
            }
        }
        assert!(
            worst <= 0.0,
            "delivered file exceeded the ceiling by {:+.3} dB: {}",
            worst,
            detail
        );
    }

    /// The headline claim: measured output true peak never exceeds the ceiling, for
    /// every ratio, ceiling and lookahead on deliberately hostile material.
    ///
    /// Every case runs even when one fails, and the message names the worst offender,
    /// because "the first case that failed" is a much less useful number than "the
    /// largest overshoot anywhere in the matrix".
    #[test]
    fn output_true_peak_never_exceeds_the_ceiling() {
        let x = worst_case(4096);
        let mut worst = (f64::NEG_INFINITY, String::new());
        // (how far below the ceiling the best case landed, that case's guard, its name)
        let mut landing = (f64::INFINITY, 0.0f64, String::new());
        let mut cases = 0usize;
        for &ratio in Ratio::ALL.iter() {
            for &ceiling in &[-1.0f64, -3.0, -5.0, -12.0] {
                for &la in &[MIN_LOOKAHEAD_MS, 1.0, MAX_LOOKAHEAD_MS] {
                    let (out, r) = run(cfg(ceiling, ratio, la), std::slice::from_ref(&x), 333);
                    cases += 1;
                    let excess = r.post_dbtp - ceiling;
                    if excess < landing.0 {
                        landing = (excess, r.guard_db, format!("{ratio} at {ceiling} dBTP"));
                    }
                    if excess > worst.0 {
                        worst = (
                            excess,
                            format!(
                                "{ratio} at {ceiling} dBTP, {la} ms requested / {:.3} ms \
                                 effective, gain reduction {:.3} dB",
                                r.effective_lookahead_ms, r.max_gain_reduction_db
                            ),
                        );
                    }
                    assert_eq!(out.len(), 1, "channel count changed: {r:?}");
                    assert_eq!(
                        out[0].len(),
                        x.len() + r.latency_samples,
                        "output length must be input plus latency, {ratio} {la} ms"
                    );
                }
            }
        }
        assert!(
            worst.0 <= 0.0,
            "{cases} cases; worst overshoot {:+.6} dB over the ceiling, at {}",
            worst.0,
            worst.1
        );
        // The ceiling is a ceiling, but it is not supposed to be a vague one: a limiter
        // that lands 6 dB below the target is a volume knob with extra steps. Each case
        // has to come within 1 dB above its own guard band, which is what the ratio's
        // optics cost, and the worst case across the matrix is printed so the number is
        // visible rather than assumed.
        assert!(
            landing.0 > -(landing.1 + 1.0),
            "worst landing {:+.3} dB below the ceiling, at {}, where the guard band is \
             {:.3} dB: that is squashing, not limiting",
            landing.0,
            landing.1,
            landing.2
        );
    }

    /// A limiter that never engages must leave the signal alone apart from the delay.
    #[test]
    fn quiet_input_passes_through_at_unity_gain() {
        let x = burst(440.0, 0.01, 4096, 0.0);
        let (_, r) = run(cfg(-1.0, Ratio::X4, 1.0), std::slice::from_ref(&x), 512);
        assert_eq!(r.max_gain_reduction_db, 0.0, "{r:?}");
        assert_eq!(r.engaged_frames, 0, "{r:?}");
        assert_eq!(r.post_dbtp, r.pre_dbtp, "gain moved the signal: {r:?}");
    }

    /// Chunk-size invariance: the same bytes out for any block sequence, including
    /// one sample at a time.
    #[test]
    fn limiter_is_chunk_size_invariant() {
        let x = worst_case(2048);
        let (a, ra) = run(cfg(-3.0, Ratio::X8, 0.7), std::slice::from_ref(&x), 4096);
        for chunk in [1usize, 7, 512, 1021] {
            let (b, rb) = run(cfg(-3.0, Ratio::X8, 0.7), std::slice::from_ref(&x), chunk);
            assert_eq!(a[0].len(), b[0].len(), "chunk {chunk}");
            let diff = a[0]
                .iter()
                .zip(b[0].iter())
                .fold(0.0f64, |m, (&u, &v)| m.max((u - v).abs()));
            assert!(diff == 0.0, "chunk {chunk} differs by {diff}");
            assert_eq!(ra.max_gain_reduction_db, rb.max_gain_reduction_db);
        }
    }

    /// The readings the detector makes before frame 0 has a report of its own are folded
    /// into frame 0, so that fold must not depend on where the blocks are cut either.
    #[test]
    fn a_crest_on_frame_zero_is_chunk_size_invariant() {
        let mut l = vec![0.0f64; 2048];
        let mut r = vec![0.0f64; 2048];
        l[0] = 4.0;
        for (i, slot) in r.iter_mut().enumerate().take(8) {
            *slot = if i % 2 == 0 { -2.0 } else { 2.0 };
        }
        let planar = [l, r];
        for &ratio in Ratio::ALL.iter() {
            let c = Cfg {
                ratio,
                ..Cfg::default()
            };
            let (a, ra) = run(c, &planar, 4096);
            for chunk in [1usize, 2, 29, 441] {
                let (b, rb) = run(c, &planar, chunk);
                assert_eq!(a, b, "{ratio}: chunk {chunk} changed the output");
                assert_eq!(ra, rb, "{ratio}: chunk {chunk} changed the report");
            }
        }
    }

    /// A gain step is a discontinuity in the output, so the attack is a ramp. Fed a
    /// flat-top block of 1.0 samples, the output *is* the gain, so the descent can be
    /// measured directly off the samples.
    #[test]
    fn gain_moves_on_a_ramp_not_a_step() {
        let c = cfg(-6.0, Ratio::X4, 1.0);
        let ramp = c.ramp_frames(48_000) as f64;
        let mut x = vec![0.0f64; 2048];
        for v in x.iter_mut().skip(1024) {
            *v = 1.0;
        }
        let (out, r) = run(c, &[x], 256);
        let lin = 10f64.powf(-6.0 / 20.0);
        assert!(r.max_gain_reduction_db > 5.0, "{r:?}");
        assert!(r.post_dbtp <= -6.0, "{r:?}");
        let o = &out[0];
        let first = o.iter().position(|&v| v.abs() > 1e-9).expect("no output");
        // Nothing above the ceiling, anywhere.
        assert!(
            o[first..].iter().all(|&v| v.abs() <= lin + 1e-12),
            "output exceeds the ceiling"
        );
        // The gain arrives at its destination by ramping: the largest single
        // frame-to-frame change is bounded by the slope, and the descent takes more
        // than one frame.
        let mut biggest = 0.0f64;
        let mut steps = 0usize;
        for w in o[first..first + 400].windows(2) {
            let d = (w[1] - w[0]).abs();
            if d > 1e-12 {
                steps += 1;
            }
            biggest = biggest.max(d);
        }
        assert!(
            biggest <= 1.0 / ramp + 1e-9,
            "gain moved {biggest} in one frame; slope allows {}",
            1.0 / ramp
        );
        assert!(steps >= 2, "gain moved in {steps} step, not a ramp");
    }

    /// The ramp where a ramp is actually visible: under signal.
    ///
    /// The test above feeds silence and then a block of 1.0, so the gain is free to fall
    /// while there is nothing to fall on — the descent happens across zeros and the output
    /// looks identical whether the schedule ramps or steps. That is how a pass that had
    /// become the identity survived a green suite. Here the material in front of the
    /// transient is a held 0.2, well under the ceiling so the slow stage keeps its hands
    /// off, and the output across the descent *is* the gain times 0.2.
    #[test]
    fn the_ramp_is_real_when_there_is_signal_under_it() {
        let c = cfg(-6.0, Ratio::X4, 2.0);
        let ramp = c.ramp_frames(48_000);
        let mut x = vec![0.2f64; 2048];
        for v in x.iter_mut().skip(1024) {
            *v = 1.0;
        }
        let (out, r) = run(c, &[x], 256);
        assert!(r.max_gain_reduction_db > 3.0, "{r:?}");
        assert!(r.post_dbtp <= -6.0, "post {} over the ceiling", r.post_dbtp);
        let o = &out[0];
        let began = o
            .iter()
            .position(|&v| v.abs() > 1e-9 && v < 0.2 * 0.999)
            .expect("the gain never moved before the transient");
        // The window the ramp is supposed to occupy. Measured before the input step at 1024,
        // because that step is the signal's, not the gain's.
        // Up to the frame the input step lands on and no further: that jump belongs to the
        // signal, not to the gain. The descent itself is shorter than the ramp whenever the
        // required depth is shallower than the schedule's full travel — 1.0 down to 0.44 at
        // a slope of 1/24 is 14 frames, not 24 — so the window has to end at the step rather
        // than be measured from where the descent started.
        let stop = c.latency_samples(48_000) + 1024;
        assert!(
            stop > began + 3,
            "descent starts at {began}, step lands at {stop}"
        );
        let span = &o[began..stop];
        let mut biggest = 0.0f64;
        let mut steps = 0usize;
        for w in span.windows(2) {
            let d = (w[1] - w[0]).abs();
            if d > 1e-12 {
                steps += 1;
            }
            biggest = biggest.max(d);
        }
        assert!(steps >= 3, "gain moved in {steps} frames, not a ramp");
        assert!(
            biggest <= 0.2 / ramp as f64 + 1e-9,
            "gain stepped {biggest} in one frame under a 0.2 signal; the slope allows {} \
             and a step there is a click in the file",
            0.2 / ramp as f64
        );
    }

    /// Below the floor the request is raised, reported as raised, and never silently
    /// shortened.
    #[test]
    fn lookahead_floor_is_enforced_and_reported() {
        let c = cfg(-1.0, Ratio::X32, MIN_LOOKAHEAD_MS);
        let rate = 8_000u32;
        assert!(c.lookahead_was_raised(rate), "{c:?} at {rate}");
        assert!(
            c.effective_lookahead_ms(rate) > MIN_LOOKAHEAD_MS,
            "effective {} ms should exceed the request",
            c.effective_lookahead_ms(rate)
        );
        assert_eq!(
            c.lookahead_samples(rate),
            c.min_lookahead_samples(),
            "the floor should be what is used"
        );
        // Accuracy costs delay, and the floor is where that shows up: a 96-tap-per-phase
        // 32x bank is 47.5 samples deep, so at 48 kHz it cannot honour a 0.2 ms request
        // and says so. Every ratio must still get at least its floor, and the effective
        // value is what the UI prints — a slider that silently did something other than
        // what it shows is worse than one with a narrower range.
        for &r in Ratio::ALL.iter() {
            for &rt in &[8_000u32, 48_000, 96_000, 192_000] {
                for &la in &[MIN_LOOKAHEAD_MS, 1.0, MAX_LOOKAHEAD_MS] {
                    let k = cfg(-1.0, r, la);
                    assert!(
                        k.lookahead_samples(rt) >= k.min_lookahead_samples(),
                        "{r} at {rt}: {la} ms gives {} frames, the filter needs {}",
                        k.lookahead_samples(rt),
                        k.min_lookahead_samples()
                    );
                    assert!(
                        k.latency_samples(rt) >= k.min_lookahead_samples(),
                        "{r} at {rt}: latency must at least cover the filter's memory"
                    );
                    // The claim `lookahead_was_raised` makes, checked against the frames.
                    assert_eq!(
                        k.lookahead_was_raised(rt),
                        k.requested_lookahead_samples(rt) < k.min_lookahead_samples(),
                        "{r} at {rt} with {la} ms"
                    );
                }
            }
        }
        // Which ratios can honour the bottom of the range at a normal rate, and which
        // cannot. This is the honest shape of the control, and it is set by filter depth:
        // the printed 12-tap x4 bank is 5.875 samples deep, so 0.2 ms (9 frames at
        // 48 kHz) is real for it, while every derived Kaiser bank is 51 taps and 25
        // samples deep, so its floor is 0.52 ms at 48 kHz and the UI has to say so rather
        // than pretend the slider reaches 0.2 ms for every setting. Depth is what bought
        // the accuracy in `tp::accuracy`; it is paid for here.
        let min_at_48k = |r: Ratio| cfg(-1.0, r, MIN_LOOKAHEAD_MS).lookahead_was_raised(48_000);
        assert!(
            !min_at_48k(Ratio::X4),
            "x4 should fit in {} ms",
            MIN_LOOKAHEAD_MS
        );
        for r in [Ratio::X2, Ratio::X8, Ratio::X16, Ratio::X32] {
            assert!(min_at_48k(r), "{r} is deeper than its memory");
            // The number that decides it is the bank's own depth, and the delay line is
            // that plus whatever lookahead was asked for. Peak refinement costs one more
            // frame, because the parabola needs the sample after the crest.
            assert_eq!(
                r.report_delay(),
                r.group_delay().ceil() as usize + usize::from(r.refines()),
                "{r} should report the Kaiser bank's own depth plus its refinement"
            );
            assert!(
                r.report_delay() > (MIN_LOOKAHEAD_MS * 48.0) as usize,
                "{r} at 48 kHz: {} frames of filter, {} frames for {} ms",
                r.report_delay(),
                (MIN_LOOKAHEAD_MS * 48.0) as usize,
                MIN_LOOKAHEAD_MS
            );
            let k = cfg(-1.0, r, MIN_LOOKAHEAD_MS);
            assert_eq!(
                k.latency_samples(48_000),
                k.lookahead_samples(48_000) + r.report_delay(),
                "{r} should delay by the lookahead it got plus the meter's reporting lag"
            );
        }
        assert_eq!(Ratio::X4.report_delay(), 6, "the printed bank is 5.875");
        // The top of the range stays usable where the sample rate is normal.
        assert!(!cfg(-1.0, Ratio::X2, MAX_LOOKAHEAD_MS).lookahead_was_raised(48_000));
        assert!(!cfg(-1.0, Ratio::X4, MAX_LOOKAHEAD_MS).lookahead_was_raised(192_000));
        assert!(!cfg(-1.0, Ratio::X32, MAX_LOOKAHEAD_MS).lookahead_was_raised(192_000));
    }

    /// A burst whose crest is in the file's last frames must still be inside the
    /// ceiling. Those frames leave through `flush()`, and `flush()` used to pop the delay
    /// line without feeding the side chain, so the gain schedule for the tail was never
    /// finalised and the last `latency` frames went out at whatever gain was current when
    /// the file ended. The bug is invisible to `report()`, because the same unfinalised
    /// schedule decides what the limiter believes it delivered, so the check is done with
    /// a x32 instrument on the emitted bytes instead.
    #[test]
    fn a_burst_that_ends_the_file_is_limited_not_escaped() {
        const PI: f64 = std::f64::consts::PI;
        const TAU: f64 = std::f64::consts::TAU;
        let c = cfg(-1.0, Ratio::X4, 1.0);
        let n = 4_000usize;
        let onset = n - 300;
        let mut l = vec![0.0f64; n];
        let mut r = vec![0.0f64; n];
        for i in 0..n {
            // A quiet bed and then a hot tone, faded in over 100 frames so the onset is
            // programme rather than a step, running to the file's very last frame.
            let env = if i < onset {
                0.0
            } else {
                let k = ((i - onset) as f64 / 100.0).min(1.0);
                0.5 * (1.0 - (PI * k).cos())
            };
            let t = i as f64;
            l[i] = env * (TAU * 0.3 * t).sin();
            r[i] = env * (TAU * 0.3 * (t - 0.25)).sin();
        }
        let pre = {
            let mut d = Detector::new(Ratio::X32);
            d.push_block(&l);
            d.push_block(&r);
            d.dbtp()
        };
        assert!(pre > 0.0, "the fixture needs to start above the ceiling");

        let (out, rep) = run(c, &[l, r], 777);
        let mut d = Detector::new(Ratio::X32);
        for ch in &out {
            d.push_block(ch);
        }
        let measured = d.dbtp();
        // 0.10 of tolerance on a -1.0 dBTP ceiling: x4 aims 0.62 dB low by its own
        // stated error, so a correct limiter lands near -1.6 and this cannot pass by
        // luck, while the escaped tail this test exists for reads about -0.9.
        assert!(
            measured <= -1.0 + 0.10,
            "delivered file measures {measured:+.3} dBTP against a -1.0 dBTP ceiling \
             (pre-limiter {pre:+.3}, limiter's own report {:+.3})",
            rep.post_dbtp
        );
        // And it limited rather than flattening: a whole-file gain of 1.0 would leave the
        // quiet bed untouched, so the tone has to be the thing that came down.
        let bed: Vec<f64> = out[0].iter().take(onset).copied().collect();
        let mut bd = Detector::new(Ratio::X32);
        bd.push_block(&bed);
        assert!(
            bd.dbtp() < -30.0,
            "the quiet bed should have passed through, measures {:+.1} dBTP",
            bd.dbtp()
        );
    }

    /// Two runs must produce identical bytes.
    #[test]
    fn limiter_is_deterministic() {
        let x = worst_case(1500);
        let (a, _) = run(cfg(-2.0, Ratio::X16, 0.4), std::slice::from_ref(&x), 222);
        let (b, _) = run(cfg(-2.0, Ratio::X16, 0.4), &[x], 222);
        assert_eq!(a[0], b[0]);
    }

    /// Linked gain: the channel balance is preserved, because a limiter that
    /// attenuated channels independently would rewrite the matrix's output.
    #[test]
    fn gain_is_linked_across_channels() {
        let loud = burst(12_000.0, 1.0, 2048, std::f64::consts::FRAC_PI_4);
        let quiet = burst(12_000.0, 0.001, 2048, std::f64::consts::FRAC_PI_4);
        let (out, _) = run(
            cfg(-6.0, Ratio::X4, 1.0),
            &[loud.clone(), quiet.clone()],
            512,
        );
        let lat = cfg(-6.0, Ratio::X4, 1.0).latency_samples(48_000);
        // Compare the tail where both channels are limited: the ratio between them
        // must be the ratio between their inputs (0.001/1.0), to within the release
        // envelope's own movement.
        let n = out[0].len();
        let mut worst = 0.0f64;
        for i in (lat + 600..n).step_by(7) {
            if out[0][i].abs() > 1e-6 && out[1][i].abs() > 1e-9 {
                let want = quiet[i.saturating_sub(lat)] / loud[i.saturating_sub(lat)];
                let got = out[1][i] / out[0][i];
                if want.abs() > 1e-9 {
                    worst = worst.max((got / want - 1.0).abs());
                }
            }
        }
        assert!(worst < 0.25, "channel balance moved by {worst}");
    }

    /// With the limiter disabled nothing is delayed and nothing is changed.
    #[test]
    fn disabled_limiter_is_an_identity() {
        let mut x = worst_case(1000);
        let before = x.clone();
        let mut lim = Limiter::disabled();
        let mut block: [&mut [f64]; 1] = [x.as_mut_slice()];
        lim.process(&mut block);
        assert_eq!(x, before, "disabled limiter changed samples");
        assert_eq!(lim.flush().len(), 0, "disabled limiter added frames");
        assert_eq!(lim.latency_samples(), 0);
        assert!(!lim.is_enabled());
    }

    #[test]
    fn config_rejects_silly_values_with_an_actionable_message() {
        assert!(cfg(-1.0, Ratio::X4, 1.0).validate().is_ok());
        assert!(cfg(1.0, Ratio::X4, 1.0).validate().is_err());
        assert!(cfg(-25.0, Ratio::X4, 1.0).validate().is_err());
        assert!(cfg(-1.0, Ratio::X4, 0.05).validate().is_err());
        assert!(cfg(-1.0, Ratio::X4, 3.0).validate().is_err());
        assert!(cfg(f64::NAN, Ratio::X4, 1.0).validate().is_err());
        // Disabled, nothing is checked: an unset ceiling must not block a render.
        let off = Cfg {
            enable: false,
            ..cfg(f64::NAN, Ratio::X4, 0.05)
        };
        assert!(off.validate().is_ok());
    }

    /// The end of the file is inside the ceiling claim, not outside it.
    #[test]
    fn a_peak_in_the_flushed_tail_is_measured() {
        let lat = cfg(-1.0, Ratio::X4, 1.0).latency_samples(48_000);
        let mut lim = Limiter::enabled(cfg(-1.0, Ratio::X4, 1.0), 48_000, 1);
        // Silence, then a full-scale crest in the last few samples: those frames leave
        // through flush(), so a report that metered only process() would call this file
        // silent.
        let mut buf = vec![0.0f64; lat * 2];
        let last = buf.len() - 1;
        buf[last] = 0.9;
        let mut block: [&mut [f64]; 1] = [buf.as_mut_slice()];
        lim.process(&mut block);
        let tail = lim.flush();
        let r = lim.report();
        assert!(!tail[0].is_empty(), "flush returned nothing");
        let tail_peak = tail[0].iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        assert!(tail_peak > 0.5, "the crest did not reach the tail");
        assert!(
            r.post_dbtp > -2.0,
            "post-limiter true peak {:+.2} dBTP does not know about the tail crest \
             at {:+.2} dBFS",
            r.post_dbtp,
            crate::tp::db_from_peak(tail_peak)
        );
    }

    /// Trailing frames are output, not debris.
    #[test]
    fn flush_returns_the_delayed_tail() {
        let lat = cfg(-1.0, Ratio::X4, 1.0).latency_samples(48_000);
        let mut lim = Limiter::enabled(cfg(-1.0, Ratio::X4, 1.0), 48_000, 1);
        let mut buf = vec![0.5f64; 100];
        let mut block: [&mut [f64]; 1] = [buf.as_mut_slice()];
        lim.process(&mut block);
        let tail = lim.flush();
        assert_eq!(tail[0].len(), lat);
        assert_eq!(block[0].len(), 100, "process must not change block length");
        // The whole render is input plus latency: nothing lost, nothing invented.
        let all: usize = block[0].iter().chain(tail[0].iter()).count();
        assert_eq!(all, 100 + lat);
    }

    /// Multichannel: every channel is measured, and the report's post reading is the
    /// group maximum.
    #[test]
    fn multichannel_output_is_measured_as_a_group() {
        let a = burst(12_000.0, 1.0, 2048, 0.4);
        let b = burst(12_000.0, 0.9, 2048, 1.2);
        let c = burst(12_000.0, 0.5, 2048, 2.0);
        let (_, r) = run(cfg(-3.0, Ratio::X8, 1.0), &[a, b, c], 512);
        assert!(r.post_dbtp <= -3.0 + 1e-6, "{r:?}");
        assert!(r.pre_dbtp > 0.0, "input was over the ceiling: {r:?}");
        assert!(r.max_gain_reduction_db > 2.0, "{r:?}");
        assert!(r.engaged_frames > 0);
    }
}

#[cfg(test)]
mod diag {
    use super::*;
    use std::f64::consts::PI;

    /// Flatness of each ratio's magnitude response, plus its response to a single
    /// impulse. A peak-hold meter is only honest if the passband is flat: a cascade
    /// that reads +2 dB on an impulse will read high on click-heavy material too.
    #[test]
    fn response_by_class() {
        for &r in Ratio::ALL.iter() {
            // impulse
            let mut d = Detector::new(r);
            let mut p_imp = 0.0f64;
            for i in 0..400 {
                p_imp = p_imp.max(d.push(if i == 200 { 1.0 } else { 0.0 }));
            }
            // tones: settle, then measure the steady-state peak
            let mut worst_low = 0.0f64;
            let mut worst_hi = 0.0f64;
            let mut f = 0.01f64;
            while f <= 0.5001 {
                let mut lo = (0.0f64, 0.0f64);
                for off in 0..16 {
                    let mut d = Detector::new(r);
                    d.clear();
                    for i in 0..400 {
                        let _ = d.push((2.0 * PI * f * i as f64 + off as f64).sin());
                    }
                    let mut p = 0.0f64;
                    for i in 0..2000 {
                        let v = (2.0 * PI * f * (i + 400) as f64 / 1.0 + off as f64).sin();
                        p = p.max(d.push(v).abs());
                    }
                    lo = (lo.0.max(p), lo.1);
                }
                if f < 0.2 {
                    worst_low = worst_low.max(lo.0);
                } else {
                    worst_hi = worst_hi.max(lo.0);
                }
                f += 0.01;
            }
            println!(
                "{r:?}  impulse {:+.3} dB   tone passband lo {:+.3} dB  hi {:+.3} dB",
                20.0 * p_imp.log10(),
                20.0 * worst_low.log10(),
                20.0 * worst_hi.log10()
            );
        }
    }
}
