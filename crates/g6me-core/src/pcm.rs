//! Sample-format vocabulary shared by the ffmpeg stages, plus render statistics.

use serde::{Deserialize, Serialize};

/// PCM format used on the wire between a stage and ffmpeg.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum BitDepth {
    S16,
    S24,
    S32,
    F32,
    F64,
}

impl BitDepth {
    pub const ALL: [BitDepth; 5] = [
        BitDepth::S16,
        BitDepth::S24,
        BitDepth::S32,
        BitDepth::F32,
        BitDepth::F64,
    ];

    pub fn sample_bytes(self) -> usize {
        match self {
            BitDepth::S16 => 2,
            BitDepth::S24 => 3,
            BitDepth::S32 | BitDepth::F32 => 4,
            BitDepth::F64 => 8,
        }
    }

    /// ffmpeg decoder/muxer name for raw PCM of this format.
    pub fn codec(self) -> &'static str {
        match self {
            BitDepth::S16 => "pcm_s16le",
            BitDepth::S24 => "pcm_s24le",
            BitDepth::S32 => "pcm_s32le",
            BitDepth::F32 => "pcm_f32le",
            BitDepth::F64 => "pcm_f64le",
        }
    }

    /// ffmpeg `-f` raw muxer name.
    pub fn raw_fmt(self) -> &'static str {
        match self {
            BitDepth::S16 => "s16le",
            BitDepth::S24 => "s24le",
            BitDepth::S32 => "s32le",
            BitDepth::F32 => "f32le",
            BitDepth::F64 => "f64le",
        }
    }

    /// `aresample`/`aformat` sample-format token.
    pub fn swr_fmt(self) -> &'static str {
        match self {
            BitDepth::S16 => "s16",
            BitDepth::S24 => "s32", // swr has no s24; see `quantize_note`
            BitDepth::S32 => "s32",
            BitDepth::F32 => "flt",
            BitDepth::F64 => "dbl",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, BitDepth::F32 | BitDepth::F64)
    }

    pub fn label(self) -> &'static str {
        match self {
            BitDepth::S16 => "16-bit int",
            BitDepth::S24 => "24-bit int",
            BitDepth::S32 => "32-bit int",
            BitDepth::F32 => "32-bit float",
            BitDepth::F64 => "64-bit float",
        }
    }

    /// Why 24-bit is quantized here and not by a filter: libswresample has no `s24`
    /// sample format (`S24` maps to `s32` above), and ffmpeg's 24-bit encoders keep the
    /// top 24 bits of that `s32` -- a floor, half an LSB low on average, dither
    /// included. [`Grid24`] rounds onto the 24-bit grid first, so the encoder's
    /// truncation drops only zeros.
    pub fn quantize_note() -> &'static str {
        "24-bit is rounded (and dithered) by g6me before the encoder; ffmpeg's own s32->s24 \
         step truncates"
    }
}

/// Rounds a stereo stream onto the 24-bit grid, optionally with dither, before it
/// reaches an encoder that would otherwise truncate it (see [`BitDepth::quantize_note`]).
///
/// Only the plain dithers are offered: ffmpeg's noise-shaped methods are not
/// reproduced here, and at 24 bits its own versions diverge.
#[derive(Clone, Debug)]
pub struct Grid24 {
    dither: Dither,
    rng: u64,
    /// Previous uniform draw per channel, for the high-passed triangular dither.
    prev: [f64; 2],
}

impl Grid24 {
    const SCALE: f64 = 8_388_608.0; // 2^23
    const MAX: f64 = 8_388_607.0;

    /// Dithers this quantizer can apply at 24 bits.
    pub fn supports(d: Dither) -> bool {
        matches!(
            d,
            Dither::None | Dither::Rectangular | Dither::Triangular | Dither::TriangularHp
        )
    }

    /// `None` for a dither [`Grid24::supports`] refuses.
    pub fn new(dither: Dither) -> Option<Self> {
        Self::supports(dither).then_some(Self {
            dither,
            // Fixed seed: two renders of one file stay byte-identical.
            rng: 0x9E37_79B9_7F4A_7C15,
            prev: [0.0; 2],
        })
    }

    /// Uniform in [-0.5, 0.5), xorshift64*.
    fn uniform(&mut self) -> f64 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let v = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    fn noise(&mut self, ch: usize) -> f64 {
        match self.dither {
            Dither::Rectangular => self.uniform(),
            Dither::Triangular => self.uniform() + self.uniform(),
            Dither::TriangularHp => {
                let u = self.uniform();
                let n = u - self.prev[ch];
                self.prev[ch] = u;
                n
            }
            _ => 0.0,
        }
    }

    /// Quantize both channels in place, frame by frame.
    pub fn process(&mut self, l: &mut [f64], r: &mut [f64]) {
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (da, db) = (self.noise(0), self.noise(1));
            *a = Self::snap(*a, da);
            *b = Self::snap(*b, db);
        }
    }

    fn snap(x: f64, noise: f64) -> f64 {
        (x * Self::SCALE + noise)
            .round_ties_even()
            .clamp(-Self::SCALE, Self::MAX)
            / Self::SCALE
    }
}

/// Noise-shaping / dither method, passed to ffmpeg's `aresample` as
/// `dither_method`. `None` is the default and keeps the numbers reproducible.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum Dither {
    #[default]
    None,
    Rectangular,
    Triangular,
    TriangularHp,
    Lipshitz,
    Shibata,
    LowShibata,
    HighShibata,
    FWeighted,
    ModifiedEWeighted,
    ImprovedEWeighted,
}

impl Dither {
    pub const ALL: [Dither; 11] = [
        Dither::None,
        Dither::Rectangular,
        Dither::Triangular,
        Dither::TriangularHp,
        Dither::Lipshitz,
        Dither::Shibata,
        Dither::LowShibata,
        Dither::HighShibata,
        Dither::FWeighted,
        Dither::ModifiedEWeighted,
        Dither::ImprovedEWeighted,
    ];

    pub fn ffmpeg_name(self) -> &'static str {
        match self {
            Dither::None => "none",
            Dither::Rectangular => "rectangular",
            Dither::Triangular => "triangular",
            Dither::TriangularHp => "triangular_hp",
            Dither::Lipshitz => "lipshitz",
            Dither::Shibata => "shibata",
            Dither::LowShibata => "low_shibata",
            Dither::HighShibata => "high_shibata",
            Dither::FWeighted => "f_weighted",
            Dither::ModifiedEWeighted => "modified_e_weighted",
            Dither::ImprovedEWeighted => "improved_e_weighted",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Dither::None => "none (reproducible)",
            other => other.ffmpeg_name(),
        }
    }
}

/// What one render actually did, for the log and for the GUI's result list.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Stats {
    pub frames_in: u64,
    pub frames_out: u64,
    /// Loudest |sample| seen on any input channel.
    pub peak_in: f64,
    /// Loudest |sample| produced by the downmixer, before encoding.
    pub peak_out: f64,
    /// Output samples outside [-1, 1] — i.e. what the encoder will clip.
    pub clipped: u64,
    pub seconds_wall: f64,
    pub out_bytes: u64,
    pub sha256: Option<String>,
}

impl Stats {
    /// Full-scale reference for PCM: `1.0` maps to 0 dBFS.
    pub fn dbfs(x: f64) -> f64 {
        if x > 0.0 && x.is_finite() {
            20.0 * x.log10()
        } else {
            f64::NEG_INFINITY
        }
    }

    pub fn peak_in_db(&self) -> f64 {
        Self::dbfs(self.peak_in)
    }

    pub fn peak_out_db(&self) -> f64 {
        Self::dbfs(self.peak_out)
    }
}

/// Incremental SHA-256 over rendered output, so two runs can be compared byte
/// for byte without keeping both files.
///
/// A thin wrapper so callers cannot accidentally construct one that reports a
/// digest of nothing.
pub struct Sha256(sha2::Sha256);

impl Sha256 {
    pub fn new() -> Self {
        use sha2::Digest;
        Self(sha2::Sha256::new())
    }

    pub fn update(&mut self, bytes: &[u8]) {
        use sha2::Digest;
        self.0.update(bytes);
    }

    pub fn finish_hex(self) -> String {
        use sha2::Digest;
        let out = self.0.finalize();
        let mut s = String::with_capacity(out.len() * 2);
        for b in out {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_the_known_test_vector() {
        // The empty-input and "abc" digests are the standard FIPS 180-4 vectors.
        assert_eq!(
            Sha256::new().finish_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let mut h = Sha256::new();
        h.update(b"a");
        h.update(b"bc");
        assert_eq!(
            h.finish_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn wire_sizes() {
        assert_eq!(BitDepth::S24.sample_bytes(), 3);
        assert_eq!(BitDepth::F64.sample_bytes(), 8);
        for d in BitDepth::ALL {
            assert!(!d.codec().is_empty());
            assert!(!d.raw_fmt().is_empty());
        }
    }

    #[test]
    fn dither_names_are_what_ffmpeg_parses() {
        // ffmpeg rejects unknown constants loudly, so a typo here would fail
        // every render; these are the exact spellings from its option list.
        assert_eq!(Dither::LowShibata.ffmpeg_name(), "low_shibata");
        assert_eq!(Dither::FWeighted.ffmpeg_name(), "f_weighted");
        assert_eq!(
            Dither::ModifiedEWeighted.ffmpeg_name(),
            "modified_e_weighted"
        );
        assert_eq!(Dither::default(), Dither::None);
        assert_eq!(Dither::None.ffmpeg_name(), "none");
    }

    #[test]
    fn dbfs() {
        assert_eq!(Stats::dbfs(1.0), 0.0);
        assert!(Stats::dbfs(0.0).is_nan() || Stats::dbfs(0.0) == f64::NEG_INFINITY);
        assert!((Stats::dbfs(0.5) + 6.0206).abs() < 0.01);
    }

    /// What ffmpeg's 24-bit encoders do with the `s32` they are handed.
    fn encoder_s24(x: f64) -> i64 {
        let s32 = (x * 2147483648.0)
            .round_ties_even()
            .clamp(-2147483648.0, 2147483647.0) as i64;
        s32 >> 8
    }

    fn error_after_encoder(d: Dither) -> (f64, f64) {
        let mut g = Grid24::new(d).unwrap();
        let mut seed = 1u64;
        let src: Vec<f64> = (0..20_000)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 1e-3
            })
            .collect();
        let (mut l, mut r) = (src.clone(), src.clone());
        g.process(&mut l, &mut r);
        let e: Vec<f64> = l
            .iter()
            .zip(&src)
            .map(|(&q, &x)| encoder_s24(q) as f64 - x * 8_388_608.0)
            .collect();
        let mean = e.iter().sum::<f64>() / e.len() as f64;
        let max = e.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        (mean, max)
    }

    #[test]
    fn grid24_output_survives_the_encoders_truncation_unbiased() {
        // Without the grid, the encoder alone sits half an LSB low.
        let raw: f64 = (0..1000)
            .map(|i| encoder_s24(i as f64 * 1.37e-7) as f64 - i as f64 * 1.37e-7 * 8_388_608.0)
            .sum::<f64>()
            / 1000.0;
        assert!(raw < -0.4, "the bias this exists for: {raw}");

        let (mean, max) = error_after_encoder(Dither::None);
        assert!(mean.abs() < 0.02 && max <= 0.5 + 1e-9, "none: {mean} {max}");
        for d in [
            Dither::Rectangular,
            Dither::Triangular,
            Dither::TriangularHp,
        ] {
            let (mean, max) = error_after_encoder(d);
            assert!(mean.abs() < 0.03, "{d:?} biased: {mean}");
            assert!(
                max > 0.5 && max <= 1.5 + 1e-9,
                "{d:?} noise out of range: {max}"
            );
        }
    }

    #[test]
    fn grid24_clamps_full_scale_and_refuses_noise_shaping() {
        let mut g = Grid24::new(Dither::None).unwrap();
        let (mut l, mut r) = (vec![1.0, -1.0, 2.0], vec![0.0; 3]);
        g.process(&mut l, &mut r);
        assert_eq!(
            l,
            vec![8_388_607.0 / 8_388_608.0, -1.0, 8_388_607.0 / 8_388_608.0]
        );
        assert!(Grid24::new(Dither::Shibata).is_none());
        assert!(Grid24::new(Dither::LowShibata).is_none());
    }
}
