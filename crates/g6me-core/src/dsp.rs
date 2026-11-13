//! The G6ME downmixer — bit-exact Rust port of `vendor/g6ct/g6ct.c`.
//!
//! ```text
//! frontL = FL + 0.707*FC + 0.707*LFE
//! frontR = FR + 0.707*FC + 0.707*LFE
//! surL   = 0.786*BL + 0.619*BR + 0.875*SL + 0.485*SR   (+0.707*BC)
//! surR   = 0.619*BL + 0.786*BR + 0.485*SL + 0.875*SR   (+0.707*BC)
//! outL   = B(A(frontL)) + D(C(surL))
//! outR   = B(A(frontR)) - D(C(surR))
//! ```
//!
//! `A..D` are second-order all-pass sections (`b0 == a2`, `b1 == a1`,
//! `b2 == 1`), so both branches are phase-only: the matrix carries the level
//! information, the all-pass cascade carries the surround image, and the minus
//! sign on the right surround branch is the out-of-phase cross-feed that puts
//! that image behind the listener.
//!
//! # Numerical contract — read before optimizing
//!
//! The float32 build of the C reference is bit-identical to the shipping
//! `KSUSBAPO64.dll`, and the float64 build is that same transfer function with
//! the arithmetic noise floor moved down. Both properties hold only because
//! every operation here is a single, explicitly ordered IEEE-754 `+`, `-` or
//! `*`. The following are therefore forbidden in this module, and
//! the `asm` group of `g6me-verify` (`crates/g6me-verify/src/checks.rs`) fails if
//! they appear in the generated assembly:
//!
//! * **FMA / `fma()`** — fusing `b0*x` into the following subtraction changes
//!   which roundings happen. In a recursive (IIR) section that error does not
//!   stay put: it is fed back through `y1`/`y2` on every later sample, so the
//!   all-pass group delay stops matching the reference and the surround image
//!   drifts. This is the one optimization that would actually break the phase,
//!   and it is why the C reference is built `-ffp-contract=off -fno-fast-math`
//!   and why no `+fma` / `-C target-cpu=native` is added in this workspace.
//! * **Reassociation** — `fl + t1 + t2` stays left-associated as in C;
//!   `(fl + t2) + t1` is a different number.
//! * **fast-math / flush-to-zero** — each section adds `1e-10` to its input as
//!   the DLL's denormal guard ([`Ap2::run`]); a build that flushes denormals or
//!   folds that add is not this DSP.
//!
//! Widening `f32 -> f64` is exact, and narrowing with the default
//! round-to-nearest-even is deterministic, so the `f32` I/O stages around the
//! DSP (device callbacks, host-style float buffers) are safe. Everything
//! between those stages stays in `T` in the order written below.
//!
//! # Why there is no SIMD here
//!
//! The four section chains are serial recursions, and the matrix is ~10 flops
//! per sample, so a vectorised version must either process different *frames*
//! in parallel (impossible: the recursion is frame-serial) or different
//! *channels* (4 lanes, two more of them serially dependent). Throughput in
//! this app therefore comes from processing several **files** concurrently,
//! each with its own single-threaded [`Dsp`] — see [`crate::batch`].

use core::ops::{Add, AddAssign, Mul, Sub, SubAssign};

// ---------------------------------------------------------------------------
// Channel mask bits (WAVEFORMATEXTENSIBLE / KSAUDIO_SPEAKER_*)
// ---------------------------------------------------------------------------

pub const CH_FL: u32 = 0x0001;
pub const CH_FR: u32 = 0x0002;
pub const CH_FC: u32 = 0x0004;
pub const CH_LFE: u32 = 0x0008;
pub const CH_BL: u32 = 0x0010;
pub const CH_BR: u32 = 0x0020;
pub const CH_FLC: u32 = 0x0040;
pub const CH_FRC: u32 = 0x0080;
pub const CH_BC: u32 = 0x0100;
pub const CH_SL: u32 = 0x0200;
pub const CH_SR: u32 = 0x0400;
pub const CH_TC: u32 = 0x0800;

/// `FL FR FC LFE BL BR SL SR` — the mask the G6 endpoint receives, and what a
/// TrueHD/Atmos 7.1 bed decodes to.
pub const MASK_7_1: u32 = 0x063f;
/// `FL FR FC LFE BL BR FLC FRC`
pub const MASK_8CH: u32 = 0x00ff;
/// `FL FR FC LFE BL BR` — DVD-style 5.1.
pub const MASK_5_1: u32 = 0x003f;
/// `FL FR` — the single mask the module passes through untouched, see
/// [`Dsp::is_bypass`].
pub const MASK_STEREO: u32 = 0x0003;

/// Width of the module's internal channel table. Slots are indexed by
/// WAVEFORMATEXTENSIBLE channel-bit position, exactly like the C reference's
/// `in[12]`:
/// `0 FL, 1 FR, 2 FC, 3 LFE, 4 BL, 5 BR, 6 FLC, 7 FRC, 8 BC, 9 SL, 10 SR, 11 TC`.
pub const SLOTS: usize = 12;

/// Slot index for a mask bit (`CH_FL -> 0`, `CH_SL -> 9`, ...).
pub const fn slot_of(bit: u32) -> usize {
    bit.trailing_zeros() as usize
}

// ---------------------------------------------------------------------------
// Coefficients — the exact float bit patterns written by the DLL constructor,
// transcribed from vendor/g6ct/g6ct.c lines 38..53.
// ---------------------------------------------------------------------------

const AP_A_B0: u32 = 0x3f72_5269;
const AP_A_B1: u32 = 0xbff9_2104;
const AP_B_B0: u32 = 0x3d82_d384;
const AP_B_B1: u32 = 0xbf56_7621;
const AP_C_B0: u32 = 0xbe85_6085;
const AP_C_B1: u32 = 0x3cd2_73d6;
const AP_D_B0: u32 = 0x3f5e_e547;
const AP_D_B1: u32 = 0xbfef_2b02;
const AP_B2: u32 = 0x3f80_0000; // 1.0
/// 1.0000000134e-10 — the DLL's denormal guard, added to every section input.
const EPSILON: u32 = 0x2edb_e6ff;

const K_707: u32 = 0x3f34_fdf4; // 0.7070000171661377
const K_485: u32 = 0x3ef8_51ec; // 0.48500001430511475
const K_619: u32 = 0x3f1e_76c9; // 0.6190000176429749
const K_786: u32 = 0x3f49_374c; // 0.7860000133514404
const K_875: u32 = 0x3f60_0000; // 0.875

/// Compile-time proof that the transcription matches the C reference, so a
/// typo here cannot silently pass the runtime golden test on an unlucky input.
macro_rules! assert_bits {
    ($name:ident, $got:expr, $want:literal) => {
        const _: () = assert!(
            $got == $want,
            concat!("coefficient drift: ", stringify!($name))
        );
    };
}
// Names + values as documented in the C header comment block.
assert_bits!(AP_A_B0, AP_A_B0, 0x3F725269);
assert_bits!(AP_A_B1, AP_A_B1, 0xBFF92104);
assert_bits!(AP_B_B0, AP_B_B0, 0x3D82D384);
assert_bits!(AP_B_B1, AP_B_B1, 0xBF567621);
assert_bits!(AP_C_B0, AP_C_B0, 0xBE856085);
assert_bits!(AP_C_B1, AP_C_B1, 0x3CD273D6);
assert_bits!(AP_D_B0, AP_D_B0, 0x3F5EE547);
assert_bits!(AP_D_B1, AP_D_B1, 0xBFEF2B02);

/// The working sample type. `f32` reproduces the DLL bit for bit; `f64` is the
/// same transfer function with a lower arithmetic noise floor.
pub trait Real:
    Copy
    + Default
    + Send
    + Sync
    + core::fmt::Debug
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + AddAssign
    + SubAssign
{
    const ZERO: Self;
    /// Matches the C reference's `g6ct_precision()`.
    const PRECISION: &'static str;

    /// Coefficient literal: an `f32` bit pattern taken verbatim from the
    /// module's constructor, promoted to the working precision. For `f64` the
    /// promotion is exact, which is what lets one source serve both builds —
    /// this is the C `u2r()`.
    fn from_f32_bits(bits: u32) -> Self;

    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;

    /// Raw bit pattern, for bit-exact comparison against the C reference.
    fn bits(self) -> u64;
    fn from_bits(bits: u64) -> Self;

    fn abs(self) -> Self;
}

impl Real for f32 {
    const ZERO: f32 = 0.0;
    const PRECISION: &'static str = "float32";
    #[inline]
    fn from_f32_bits(bits: u32) -> Self {
        f32::from_bits(bits)
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        f32::from_bits(bits as u32)
    }
    #[inline]
    fn abs(self) -> Self {
        f32::abs(self)
    }
}

impl Real for f64 {
    const ZERO: f64 = 0.0;
    const PRECISION: &'static str = "float64";
    #[inline]
    fn from_f32_bits(bits: u32) -> Self {
        // Exact: every f32 is representable in f64.
        f32::from_bits(bits) as f64
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn bits(self) -> u64 {
        self.to_bits()
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        f64::from_bits(bits)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
}

/// One 2nd-order all-pass section, direct form I, with the module's state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ap2<T> {
    /// feed-forward (`b2` is always 1.0)
    pub b0: T,
    pub b1: T,
    pub b2: T,
    /// feed-back, subtracted
    pub a1: T,
    pub a2: T,
    /// input history
    pub x1: T,
    pub x2: T,
    /// output history
    pub y1: T,
    pub y2: T,
}

impl<T: Real> Ap2<T> {
    /// All-pass section from its two distinct coefficients: `b0 == a2` and
    /// `b1 == a1`, which is what makes `H(z)` phase-only.
    pub fn allpass(b0: T, b1: T) -> Self {
        Self {
            b0,
            b1,
            b2: T::from_f32_bits(AP_B2),
            a1: b1,
            a2: b0,
            x1: T::ZERO,
            x2: T::ZERO,
            y1: T::ZERO,
            y2: T::ZERO,
        }
    }

    /// Clear the filter history, keep the coefficients.
    #[inline]
    pub fn reset(&mut self) {
        self.x1 = T::ZERO;
        self.x2 = T::ZERO;
        self.y1 = T::ZERO;
        self.y2 = T::ZERO;
    }

    /// `y = b0*in + b1*x1 + b2*x2 - a1*y1 - a2*y2`, with `in = x + 1e-10`.
    ///
    /// The parenthesisation reproduces the C source's left-to-right association
    /// and is part of the bit-identical result — see the module-level numerical
    /// contract.
    #[inline]
    pub fn run(&mut self, x: T) -> T {
        let inp = x + T::from_f32_bits(EPSILON);
        let t0 = self.b0 * inp;
        let t1 = self.b1 * self.x1;
        let t2 = self.b2 * self.x2;
        let fwd = (t0 + t1) + t2;
        let y = (fwd - (self.a1 * self.y1)) - (self.a2 * self.y2);

        self.x2 = self.x1;
        self.x1 = inp;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// `|H(e^jw)|` for `H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)`.
    ///
    /// Equals 1.0 for a well-formed all-pass. Analysis only — not in the sample
    /// path — which is why it may use `sin`/`cos`.
    pub fn magnitude(&self, w: f64) -> f64 {
        let (nr, ni) = self.eval_num(w);
        let (dr, di) = self.eval_den(w);
        let num = nr * nr + ni * ni;
        let den = dr * dr + di * di;
        if den > 0.0 {
            (num / den).sqrt()
        } else {
            0.0
        }
    }

    /// Numerator on the unit circle.
    pub fn eval_num(&self, w: f64) -> (f64, f64) {
        let (c1, s1) = (w.cos(), w.sin());
        let (c2, s2) = ((2.0 * w).cos(), (2.0 * w).sin());
        let b0 = self.b0.to_f64();
        let b1 = self.b1.to_f64();
        let b2 = self.b2.to_f64();
        (b0 + (b1 * c1) + (b2 * c2), -((b1 * s1) + (b2 * s2)))
    }

    /// Denominator `1 + a1 z^-1 + a2 z^-2` on the unit circle.
    pub fn eval_den(&self, w: f64) -> (f64, f64) {
        let (c1, s1) = (w.cos(), w.sin());
        let (c2, s2) = ((2.0 * w).cos(), (2.0 * w).sin());
        let a1 = self.a1.to_f64();
        let a2 = self.a2.to_f64();
        (1.0 + (a1 * c1) + (a2 * c2), -((a1 * s1) + (a2 * s2)))
    }

    /// Unwrapped phase of the section at `w`, in radians.
    pub fn phase(&self, w: f64) -> f64 {
        let (nr, ni) = self.eval_num(w);
        let (dr, di) = self.eval_den(w);
        ni.atan2(nr) - di.atan2(dr)
    }

    /// Group delay `-(d/dw) arg H(e^jw)` in samples, by central difference.
    pub fn group_delay(&self, w: f64) -> f64 {
        let h = 1e-5;
        let pi = std::f64::consts::PI;
        let (a, b) = (self.phase((w + h).min(pi)), self.phase((w - h).max(0.0)));
        let mut d = a - b;
        // unwrap across the +-pi boundary
        while d > pi {
            d -= 2.0 * pi;
        }
        while d < -pi {
            d += 2.0 * pi;
        }
        -d / (2.0 * h)
    }

    /// Roots of the denominator `z^2 + a1 z + a2`; both must satisfy `|z| < 1`
    /// for the section to be stable. Returns the two real parts, or `+-radius`
    /// for a complex-conjugate pair.
    pub fn poles(&self) -> (f64, f64) {
        let a1 = self.a1.to_f64();
        let a2 = self.a2.to_f64();
        let disc = a1 * a1 - 4.0 * a2;
        if disc >= 0.0 {
            let r = disc.sqrt();
            ((-a1 + r) / 2.0, (-a1 - r) / 2.0)
        } else {
            let rad = a2.max(0.0).sqrt();
            (rad, -rad)
        }
    }

    /// Largest pole radius; must be `< 1.0`.
    pub fn pole_radius(&self) -> f64 {
        let (p, q) = self.poles();
        p.abs().max(q.abs())
    }
}

/// The downmixer. One instance per stream; a worker thread owns its own, which
/// is what keeps the DSP single-threaded per file.
#[derive(Clone, Debug)]
pub struct Dsp<T> {
    /// A, B — front path, left
    front_l: [Ap2<T>; 2],
    /// C, D — surround path, left
    sur_l: [Ap2<T>; 2],
    /// A, B — front path, right
    front_r: [Ap2<T>; 2],
    /// C, D — surround path, right
    sur_r: [Ap2<T>; 2],
    mask: u32,
}

impl<T: Real> Default for Dsp<T> {
    fn default() -> Self {
        Self::new(MASK_7_1)
    }
}

impl<T: Real> Dsp<T> {
    /// Initialise coefficients and zero state for an input channel mask.
    pub fn new(mask: u32) -> Self {
        let front = || {
            [
                Ap2::allpass(T::from_f32_bits(AP_A_B0), T::from_f32_bits(AP_A_B1)),
                Ap2::allpass(T::from_f32_bits(AP_B_B0), T::from_f32_bits(AP_B_B1)),
            ]
        };
        let sur = || {
            [
                Ap2::allpass(T::from_f32_bits(AP_C_B0), T::from_f32_bits(AP_C_B1)),
                Ap2::allpass(T::from_f32_bits(AP_D_B0), T::from_f32_bits(AP_D_B1)),
            ]
        };
        Self {
            front_l: front(),
            sur_l: sur(),
            front_r: front(),
            sur_r: sur(),
            mask,
        }
    }

    /// Drop all filter history, keep coefficients and mask.
    pub fn reset(&mut self) {
        for s in self.sections_mut() {
            s.reset();
        }
    }

    pub fn mask(&self) -> u32 {
        self.mask
    }

    /// Arithmetic precision of this instantiation, e.g. `"float64"`.
    pub fn precision(&self) -> &'static str {
        T::PRECISION
    }

    /// The module has a stereo-bypass branch: an input mask of exactly `FL|FR`
    /// returns the front pair untouched (verified bit-for-bit against the DLL).
    /// Mask 0, and masks with any other bit set, run the downmixer.
    pub fn is_bypass(&self) -> bool {
        self.mask == MASK_STEREO
    }

    /// `false` once any section state has gone non-finite. The recursion is
    /// stable for the shipped coefficients (all poles well inside the unit
    /// circle), so this only trips on non-finite *input*; the batch engine
    /// checks it to report a corrupt source instead of writing NaN files.
    pub fn is_finite(&self) -> bool {
        self.sections().iter().all(|s| {
            s.x1.to_f64().is_finite()
                && s.x2.to_f64().is_finite()
                && s.y1.to_f64().is_finite()
                && s.y2.to_f64().is_finite()
        })
    }

    /// Sections in a fixed order, for verification:
    /// `[front_l A, front_l B, sur_l C, sur_l D, front_r A, front_r B, sur_r C, sur_r D]`.
    pub fn sections(&self) -> [&Ap2<T>; 8] {
        [
            &self.front_l[0],
            &self.front_l[1],
            &self.sur_l[0],
            &self.sur_l[1],
            &self.front_r[0],
            &self.front_r[1],
            &self.sur_r[0],
            &self.sur_r[1],
        ]
    }

    /// The eight sections in signal-flow order, mutably.
    ///
    /// `each_mut` splits each pair into two disjoint `&mut` before the array
    /// literal is built, which is how the borrow checker accepts eight live
    /// borrows of one struct. The order matches [`Dsp::sections`].
    fn sections_mut(&mut self) -> [&mut Ap2<T>; 8] {
        let Self {
            front_l,
            front_r,
            sur_l,
            sur_r,
            mask: _,
        } = self;
        let [fl0, fl1] = front_l.each_mut();
        let [fr0, fr1] = front_r.each_mut();
        let [sl0, sl1] = sur_l.each_mut();
        let [sr0, sr1] = sur_r.each_mut();
        [fl0, fl1, sl0, sl1, fr0, fr1, sr0, sr1]
    }

    /// Process `n` frames.
    ///
    /// `ch` is indexed by WAVEFORMATEXTENSIBLE channel-bit position (see
    /// [`SLOTS`]); `None` means the stream carries no such channel and is read
    /// as silence, exactly like a `NULL` pointer in the C reference. `out_l`
    /// and `out_r` must each hold at least `n` frames.
    pub fn process(
        &mut self,
        ch: &[Option<&[T]>; SLOTS],
        out_l: &mut [T],
        out_r: &mut [T],
        n: usize,
    ) {
        debug_assert!(out_l.len() >= n && out_r.len() >= n);
        debug_assert!(ch.iter().all(|c| match c {
            Some(s) => s.len() >= n,
            None => true,
        }));

        // Stereo bypass, verified against the DLL: mask FL|FR is a pure wire.
        if self.is_bypass() {
            for i in 0..n {
                out_l[i] = ch[0].map_or(T::ZERO, |c| c[i]);
                out_r[i] = ch[1].map_or(T::ZERO, |c| c[i]);
            }
            return;
        }

        let m = self.mask;
        let k707 = T::from_f32_bits(K_707);
        let k485 = T::from_f32_bits(K_485);
        let k619 = T::from_f32_bits(K_619);
        let k786 = T::from_f32_bits(K_786);
        let k875 = T::from_f32_bits(K_875);

        let has_fc = (m & CH_FC) != 0;
        let has_lfe = (m & CH_LFE) != 0;
        let has_bc = (m & CH_BC) != 0;
        let has_ss = (m & CH_SL) != 0 && (m & CH_SR) != 0;
        let has_bs = (m & CH_BL) != 0 && (m & CH_BR) != 0;

        let fl_i = ch[slot_of(CH_FL)];
        let fr_i = ch[slot_of(CH_FR)];
        let fc_i = ch[slot_of(CH_FC)];
        let lfe_i = ch[slot_of(CH_LFE)];
        let bl_i = ch[slot_of(CH_BL)];
        let br_i = ch[slot_of(CH_BR)];
        let bc_i = ch[slot_of(CH_BC)];
        let sl_i = ch[slot_of(CH_SL)];
        let sr_i = ch[slot_of(CH_SR)];

        for i in 0..n {
            let mut fl = fl_i.map_or(T::ZERO, |c| c[i]);
            let mut fr = fr_i.map_or(T::ZERO, |c| c[i]);
            let mut sl = T::ZERO;
            let mut sr = T::ZERO;

            if has_fc {
                let t = fc_i.map_or(T::ZERO, |c| c[i]) * k707;
                fl += t;
                fr += t;
            }
            if has_lfe {
                let t = lfe_i.map_or(T::ZERO, |c| c[i]) * k707;
                fl += t;
                fr += t;
            }
            if has_bc {
                let t = bc_i.map_or(T::ZERO, |c| c[i]) * k707;
                sl = t;
                sr = t;
            }
            if has_ss {
                let a = sl_i.map_or(T::ZERO, |c| c[i]);
                let b = sr_i.map_or(T::ZERO, |c| c[i]);
                sl += (k875 * a) + (k485 * b);
                sr += (k485 * a) + (k875 * b);
            }
            if has_bs {
                let a = bl_i.map_or(T::ZERO, |c| c[i]);
                let b = br_i.map_or(T::ZERO, |c| c[i]);
                sl += (k619 * b) + (k786 * a);
                sr += (k619 * a) + (k786 * b);
            }

            let mut l = self.front_l[0].run(fl);
            l = self.front_l[1].run(l);
            let mut t = self.sur_l[0].run(sl);
            t = self.sur_l[1].run(t);
            out_l[i] = l + t;

            let mut r = self.front_r[0].run(fr);
            r = self.front_r[1].run(r);
            let mut t = self.sur_r[0].run(sr);
            t = self.sur_r[1].run(t);
            out_r[i] = r - t;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stereo mask is the DLL's bypass branch, so a DC test on it measures a wire.
    ///
    /// This is the fact that made an earlier version of this test worthless: it passed,
    /// and it proved nothing, because `mask == FL|FR` skips the whole all-pass cascade
    /// (`Dsp::process`, "Stereo bypass, verified against the DLL"). Anything that claims
    /// to measure what the filters do has to use a mask that reaches them.
    #[test]
    fn the_stereo_mask_is_a_wire_not_a_filter_chain() {
        let n = 4096;
        let sig: Vec<f64> = (0..n)
            .map(|k| 0.5 * (2.0 * std::f64::consts::PI * k as f64 / 512.0).sin() + 0.25)
            .collect();
        let mut ch: [Option<&[f64]>; SLOTS] = [None; SLOTS];
        ch[0] = Some(&sig);
        ch[1] = Some(&sig);
        let mut l = vec![0.0f64; n];
        let mut r = vec![0.0f64; n];
        let mut d = Dsp::<f64>::new(MASK_STEREO);
        assert!(d.is_bypass(), "FL|FR must select the bypass branch");
        d.process(&ch, &mut l, &mut r, n);
        assert_eq!(l, sig, "the bypass branch changed the left channel");
        assert_eq!(r, sig, "the bypass branch changed the right channel");
    }

    /// The filtered path's DC, measured where it is actually settled.
    ///
    /// Three things had to be true before any of these numbers meant anything.
    ///
    /// *The mask has to reach the filters.* `MASK_7_1`, not the stereo bypass.
    ///
    /// *The slots have to be the ones the mask names.* `SLOTS` is indexed by
    /// `WAVEFORMATEXTENSIBLE` bit position, so 7.1 is FL(0) FR(1) FC(2) LFE(3) BL(4)
    /// BR(5) SL(9) SR(10) — slots 6..8 are FLC/FRC/BC and are not in the mask. Filling
    /// `[0..8]` drives two channels the file does not have and leaves two it does
    /// silent, which is how a first attempt here measured a matrix gain of 0.955 where
    /// the real one is 1.2948.
    ///
    /// *The window has to start after the transient and span whole periods.* The slowest
    /// pole in the cascade is section A's at p = 0.99485016, tau = 193.7 samples, so
    /// 16_384 samples is 85 tau — the cold-start response is long gone — and
    /// `N - START` is 96 whole periods of the 512-sample test tone, so the tone
    /// contributes exactly nothing to a mean taken there. Measured from sample 0,
    /// instead, the same file reads +2.29e-4: not DC, the fixed area of the response
    /// that a cold start never emits, divided by the window length. It shrinks as the
    /// window grows, which is how to tell the two apart — real DC does not care how
    /// long you listen.
    ///
    /// The matrix's own output, pinned as bytes, on every platform we ship.
    ///
    /// The comparison this project rests on is compiled from `g6ct.c` by gcc, so it can only
    /// run where gcc and the vendored objects are — which leaves the Windows binary's
    /// arithmetic asserted by nobody. This closes that gap at the level actually under our
    /// control. What the file finally holds is ffmpeg's encoder's business; these numbers are
    /// ours, produced by `+ - * /` on IEEE-64 with no fused multiply-add anywhere in the path,
    /// so there is no legitimate platform excuse for a single bit of them to move. Pinned from
    /// a Linux run: if another target disagrees, that disagreement is the finding.
    #[test]
    fn the_matrix_output_is_the_same_bytes_on_every_platform() {
        use crate::pcm::Sha256;
        const N: usize = 4096;
        let mask = CH_FL | CH_FR | CH_FC | CH_LFE | CH_BL | CH_BR | CH_SL | CH_SR;
        let mut dsp = Dsp::new(mask);
        assert!(
            !dsp.is_bypass(),
            "this mask must reach the filters, not the wire"
        );

        // Deterministic programme-like material: one linear congruential run per channel,
        // offset so no two branches see the same sequence, scaled to a level that exercises
        // the filters without pinning against full scale.
        let chans: Vec<Vec<f64>> = (0..8)
            .map(|c| {
                let mut s = 0x1234_5678_u64 ^ (c as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                (0..N)
                    .map(|_| {
                        s = s
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((s >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.6
                    })
                    .collect()
            })
            .collect();

        // Slots are WAVEFORMATEXTENSIBLE bit positions, so 7.1 is not the first eight: slots
        // 6, 7 and 8 are FLC, FRC and BC, and feeding a 7.1 file into them is the mistake this
        // file's own history contains.
        const SLOTS_7_1: [usize; 8] = [0, 1, 2, 3, 4, 5, 9, 10];
        let mut ch: [Option<&[f64]>; SLOTS] = [None; SLOTS];
        for (i, slot) in SLOTS_7_1.iter().enumerate() {
            ch[*slot] = Some(&chans[i]);
        }

        let mut l = vec![0.0f64; N];
        let mut r = vec![0.0f64; N];
        dsp.process(&ch, &mut l, &mut r, N);
        assert!(dsp.is_finite(), "the recursion went non-finite");
        // The tail matters as much as the head: the state carried between blocks is the part
        // most likely to differ if anything ever contracts or reassociates.
        for v in l.iter().chain(r.iter()) {
            assert!(v.is_finite());
        }

        let mut h = Sha256::new();
        h.update(dsp.precision().as_bytes());
        for v in l.iter().chain(r.iter()) {
            h.update(&v.to_le_bytes());
        }
        assert_eq!(
            h.finish_hex(),
            "1a3858ad6bb9cb458ba405a1a84e5d8db2bf99b23d9ab553ab71bfa8aeb44570",
            "the matrix's output is not the bytes the reference platform produced"
        );
    }

    /// What is left is the denormal guard. Every section input gets `+EPSILON`
    /// (`coefs.inc`, the DLL's own anti-denormal bias), and each all-pass section has
    /// `H(1) = (b0+b1+1)/(1+a1+a2) = 1` exactly, because its numerator is its
    /// denominator reversed: the all-pass zeros at 1/p cancel any DC rise its poles
    /// would give. So the guard is carried to the output at unity — two sections per
    /// branch, front and surround, summed for left and differenced for right:
    /// left gets `+4*EPSILON` (−188 dBFS, 0.0034 of a 24-bit step) and right cancels to
    /// exactly zero. Nothing in the port accumulates DC; a x4 oversampled true-peak
    /// reading is not going to drift up a file's baseline over its length.
    ///
    /// Real input DC is a different matter, and this measures that too: it passes at the
    /// matrix's DC sums, `1 + 2*k707` for the front pair and `k786 + k619 + k875 + k485`
    /// for the surround — 5.179 on left, -0.351 on right. A quarter of DC in arrives as
    /// +1.295 on left, which is +14.3 dB of *offset*, and the limiter downstream cannot
    /// tell that from level. That is the reason DC is metered now, and the reason a
    /// blocker exists as an opt-in rather than a rewrite of this file: `Dsp` is the
    /// bit-exact port and must stay that way.
    #[test]
    fn filtered_path_dc_steady_state_and_matrix_dc_gain() {
        const SLOTS_7_1: [usize; 8] = [0, 1, 2, 3, 4, 5, 9, 10];
        const PERIOD: usize = 512;
        const W: f64 = 2.0 * std::f64::consts::PI / (PERIOD as f64);
        const START: usize = 16_384;
        const N: usize = 65_536;

        fn run_7_1(gen: fn(usize) -> f64) -> (f64, f64) {
            let sig: Vec<f64> = (0..N).map(gen).collect();
            let mut ch: [Option<&[f64]>; SLOTS] = [None; SLOTS];
            for &slot in &SLOTS_7_1 {
                ch[slot] = Some(&sig);
            }
            let mut l = vec![0.0f64; N];
            let mut r = vec![0.0f64; N];
            let mut dsp = Dsp::<f64>::new(MASK_7_1);
            assert!(!dsp.is_bypass(), "MASK_7_1 must reach the all-pass cascade");
            dsp.process(&ch, &mut l, &mut r, N);
            let win = (N - START) as f64;
            (
                l[START..].iter().sum::<f64>() / win,
                r[START..].iter().sum::<f64>() / win,
            )
        }

        let eps = f64::from_f32_bits(EPSILON);
        for (name, gen) in [
            ("sin", (|k| 0.5 * (W * k as f64).sin()) as fn(usize) -> f64),
            ("cos", (|k| 0.5 * (W * k as f64).cos()) as fn(usize) -> f64),
        ] {
            let (ml, mr) = run_7_1(gen);
            assert!(
                (ml - 4.0 * eps).abs() < 1e-12,
                "{name}: settled left mean should be 4*EPSILON ({:+.6e}), got {:+.6e}",
                4.0 * eps,
                ml
            );
            assert!(
                mr.abs() < 1e-12,
                "{name}: settled right mean should cancel to 0, got {:+.6e}",
                mr
            );
        }

        let front = 1.0 + 2.0 * f64::from_f32_bits(K_707);
        let sur = f64::from_f32_bits(K_786)
            + f64::from_f32_bits(K_619)
            + f64::from_f32_bits(K_875)
            + f64::from_f32_bits(K_485);
        let want_l = 0.25 * (front + sur) + 4.0 * eps;
        let want_r = 0.25 * (front - sur);
        let (ml, mr) = run_7_1(|k| 0.5 * (W * k as f64).sin() + 0.25);
        assert!(
            (ml - want_l).abs() < 1e-11,
            "left DC gain: want {want_l:.9}, got {ml:.9}"
        );
        assert!(
            (mr - want_r).abs() < 1e-11,
            "right DC gain: want {want_r:.9}, got {mr:.9}"
        );
    }
}
