//! The checks themselves.
//!
//! Every check answers one question with bytes rather than adjectives:
//!
//! | check | question |
//! |---|---|
//! | `bitexact` | does the Rust f64 downmixer produce **exactly** the f32 samples the reference `-DG6CT_DOUBLE` build produces? |
//! | `builds` | does that still hold for `-march=native` and `-O3`? |
//! | `width` | how far is the f32-internal build from the f64-internal one — i.e. what the f64 invariant actually buys? |
//! | `chunk` | is the result independent of the block size the file is fed in? |
//! | `continuity` | does filter state carry across block boundaries exactly? |
//! | `drift` | over a minute of material, does anything accumulate, wander, or blow up? |
//! | `phase` | do the two paths' delays, polarities and per-slot timings match sample-for-sample? |
//! | `contraction` | is any mul+add in this binary being contracted into FMA? |
//! | `asm` | does the disassembler agree? |

use crate::reference::{self, RefBuild, MASK_7_1, REF_BLOCK, SLOTS_8};
use crate::signal::{self, Diff, Kind};
use g6me_core::dsp::{Dsp, SLOTS};
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Pass,
    Fail,
    Skip,
    /// Informational: measured and reported, not pass/fail (toolchain-dependent).
    Info,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Fail => "FAIL",
            Status::Skip => "SKIP",
            Status::Info => "info",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Check {
    pub group: &'static str,
    pub name: String,
    pub status: Status,
    /// One-line evidence: counts, delays, dB figures, the first differing frame.
    pub detail: String,
    pub metric: Option<Value>,
}

pub fn pass(group: &'static str, name: impl Into<String>, detail: impl Into<String>) -> Check {
    Check {
        group,
        name: name.into(),
        status: Status::Pass,
        detail: detail.into(),
        metric: None,
    }
}

pub fn fail(group: &'static str, name: impl Into<String>, detail: impl Into<String>) -> Check {
    Check {
        group,
        name: name.into(),
        status: Status::Fail,
        detail: detail.into(),
        metric: None,
    }
}

pub fn skip(group: &'static str, name: impl Into<String>, detail: impl Into<String>) -> Check {
    Check {
        group,
        name: name.into(),
        status: Status::Skip,
        detail: detail.into(),
        metric: None,
    }
}

pub fn info(group: &'static str, name: impl Into<String>, detail: impl Into<String>) -> Check {
    Check {
        group,
        name: name.into(),
        status: Status::Info,
        detail: detail.into(),
        metric: None,
    }
}

/// Run the Rust downmixer over interleaved f32, in f64 internally, narrowing the
/// result to f32 the way the reference CLI does on write.
pub fn rust_f64(input: &[f32], channels: u16, mask: u32, block: usize) -> Vec<f32> {
    run_rust::<f64>(input, channels, mask, block)
}

/// The same through `Dsp<f32>`, for the width comparison.
pub fn rust_f32(input: &[f32], channels: u16, mask: u32, block: usize) -> Vec<f32> {
    run_rust::<f32>(input, channels, mask, block)
}

fn run_rust<T: g6me_core::dsp::Real + Copy + Default>(
    input: &[f32],
    channels: u16,
    mask: u32,
    block: usize,
) -> Vec<f32> {
    let ch = channels as usize;
    let frames = input.len() / ch;
    let slots = slot_map(channels);
    // Widen exactly once, up front: the reference CLI does the same on read, so
    // the arithmetic both sides perform afterwards starts from identical values.
    let wide: Vec<T> = input.iter().map(|&s| T::from_f64(s as f64)).collect();
    let mut planar: Vec<Vec<T>> = vec![vec![T::default(); block]; ch];
    let mut out_l = vec![T::default(); block];
    let mut out_r = vec![T::default(); block];
    let mut dsp = Dsp::<T>::new(mask);
    let mut out = Vec::with_capacity(frames * 2);
    let mut done = 0;
    while done < frames {
        let n = block.min(frames - done);
        for c in 0..ch {
            for i in 0..n {
                planar[c][i] = wide[(done + i) * ch + c];
            }
        }
        // Built fresh per block: the slot array borrows `planar`, so it cannot be
        // reused across a mutation of it.
        let mut chans: [Option<&[T]>; SLOTS] = [None; SLOTS];
        for c in 0..ch {
            chans[slots[c]] = Some(&planar[c][..n]);
        }
        dsp.process(&chans, &mut out_l[..n], &mut out_r[..n], n);
        for i in 0..n {
            out.push(out_l[i].to_f64() as f32);
            out.push(out_r[i].to_f64() as f32);
        }
        done += n;
    }
    out
}

/// Input channel order → WAVEFORMATEXTENSIBLE slot position, matching the
/// reference CLI's `slot8` / `slot12` tables.
pub fn slot_map(channels: u16) -> Vec<usize> {
    match channels {
        12 => SLOTS_12.to_vec(),
        _ => SLOTS_8.to_vec(),
    }
}

use crate::reference::SLOTS_12;

/// Where the samples go while a reference build runs.
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Self {
        let d = std::env::temp_dir().join(format!(
            "g6me-verify-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&d).expect("scratch dir");
        Self(d)
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    pub fn write_raw(&self, name: &str, v: &[f32]) -> PathBuf {
        let p = self.path(name);
        std::fs::write(&p, signal::f32_bytes(v)).expect("write raw");
        p
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Reference builds that claim the numerical contract, i.e. everything except the
/// deliberate counter-example.
pub fn contract_builds<'a>(builds: &'a [RefBuild], width: &str) -> Vec<&'a RefBuild> {
    builds
        .iter()
        .filter(|b| b.width == width && b.variant != "contract_fast")
        .collect()
}

/// The counter-example: same C source, built the way a careless build is built.
pub fn loose_build<'a>(builds: &'a [RefBuild], width: &str) -> Option<&'a RefBuild> {
    builds
        .iter()
        .find(|b| b.width == width && b.variant == "contract_fast")
}

/// Input configurations to verify. A mask selects which filters exist at all, so
/// bit-exactness on the stock 7.1 mask alone would not prove much: the 5.1 and
/// 12-channel cases run different subsets of the filter bank, and 12 channels also
/// changes the slot mapping.
pub const CASES: &[(u32, u16, &str)] = &[
    (MASK_7_1, 8, "7.1"),
    (0x003f, 8, "5.1"),
    (0x0fff, 12, "atmos-12"),
    (0x0003, 8, "front-only"),
];

/// `bitexact`: the claim the whole project rests on.
pub fn bitexact(builds: &[RefBuild], scratch: &Scratch, frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    let f64_builds = contract_builds(builds, "f64");
    if f64_builds.is_empty() {
        return vec![skip(
            "bitexact",
            "reference builds",
            "no -DG6CT_DOUBLE reference binary was built (no C compiler?)",
        )];
    }
    for (idx, &(mask, channels, case)) in CASES.iter().enumerate() {
        // The stock case gets every signal; the others get the three that exercise
        // state, transients and broadband content, to keep the suite quick.
        let kinds: Vec<Kind> = if idx == 0 {
            Kind::all().to_vec()
        } else {
            vec![Kind::Partials, Kind::Impulse, Kind::Noise]
        };
        for kind in kinds {
            let sig = signal::generate(kind, frames, channels);
            let inp = scratch.write_raw(&format!("{case}-{kind}.raw"), &sig);
            for b in &f64_builds {
                let want = match reference::run(b, &inp, &scratch.path("o.raw"), mask, channels) {
                    Ok(v) => v,
                    Err(e) => {
                        out.push(fail("bitexact", format!("{}/{case}/{kind}", b.name), e));
                        continue;
                    }
                };
                let want = signal::f32_from_bytes(&want);
                // Feed Rust at the reference's own block size first: same input, same
                // chunking, so any difference can only be arithmetic.
                let got = rust_f64(&sig, channels, mask, REF_BLOCK);
                let d = signal::compare(&got, &want);
                let name = format!("{}/{case}/{kind}", b.name);
                let where_ = format!("{channels}ch mask {mask:#05x}");
                if d.exact() {
                    out.push(with_metric(
                        pass(
                            "bitexact",
                            name,
                            format!("{where_}: {} frames, bit-identical", d.samples),
                        ),
                        json!({
                            "mask": mask, "channels": channels, "kind": kind.label(),
                            "frames": d.samples, "differing": 0,
                        }),
                    ));
                } else {
                    let zeros = if d.signed_zero > 0 {
                        format!(" ({} of them only in the sign of zero)", d.signed_zero)
                    } else {
                        String::new()
                    };
                    out.push(with_metric(
                        fail(
                            "bitexact",
                            name,
                            format!(
                                "{where_}: {} of {} frames differ; max |d|={} worst {} \
                                 relative; first at frame {:?}{zeros}",
                                d.differing,
                                d.samples,
                                fmt_f(d.max_abs),
                                fmt_f(d.max_rel),
                                d.first_frame,
                            ),
                        ),
                        diff_json(&d),
                    ));
                }
            }
        }
    }
    out
}

/// `builds`: the reference must not care about `-O3`/`-march=native` either, which
/// is what proves the numerical contract rather than a lucky codegen.
pub fn builds_agree(builds: &[RefBuild], scratch: &Scratch, frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    let f64 = contract_builds(builds, "f64");
    if f64.len() < 2 {
        return vec![skip(
            "builds",
            "reference variants",
            "only one f64 build available",
        )];
    }
    for kind in [Kind::Partials, Kind::Noise] {
        let sig = signal::generate(kind, frames, 8);
        let inp = scratch.write_raw(&format!("v-{}.raw", kind.label()), &sig);
        let mut prev: Option<(String, Vec<f32>)> = None;
        for b in &f64 {
            let got = match reference::run(b, &inp, &scratch.path("v.raw"), MASK_7_1, 8) {
                Ok(v) => signal::f32_from_bytes(&v),
                Err(e) => {
                    out.push(fail("builds", format!("{}/{kind}", b.name), e));
                    continue;
                }
            };
            if let Some((pname, pwant)) = &prev {
                let d = signal::compare(&got, pwant);
                let name = format!("{pname} vs {}", b.name);
                out.push(if d.exact() {
                    pass(
                        "builds",
                        name,
                        format!("{kind:?} output identical across optimisation levels"),
                    )
                } else {
                    fail(
                        "builds",
                        name,
                        format!(
                            "{} frames differ (max {} ULP) — the C build flags changed the numbers",
                            d.differing, d.max_ulp
                        ),
                    )
                });
            }
            prev = Some((b.name.clone(), got));
        }
    }
    out
}

/// `width`: how much the f64-internal invariant actually buys.
///
/// The f32 build (float state) and the f64 build (`G6CT_DOUBLE`) share coefficients
/// but not arithmetic. Upstream's own note is that f64 is "60 dB closer to an exact
/// render"; measuring the gap here is what stops that claim being folklore.
pub fn width_gap(builds: &[RefBuild], scratch: &Scratch, frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    let pick = |w: &str| {
        builds
            .iter()
            .find(|b| b.variant == "contract_off" && b.width == w)
    };
    let (Some(f64b), Some(f32b)) = (pick("f64"), pick("f32")) else {
        return vec![skip("width", "f32 and f64 builds", "both widths not built")];
    };
    let sig = signal::generate(Kind::Partials, frames, 8);
    let inp = scratch.write_raw("w.raw", &sig);
    let a = match reference::run(f64b, &inp, &scratch.path("a.raw"), MASK_7_1, 8) {
        Ok(v) => signal::f32_from_bytes(&v),
        Err(e) => return vec![fail("width", "f64 reference", e)],
    };
    let b = match reference::run(f32b, &inp, &scratch.path("b.raw"), MASK_7_1, 8) {
        Ok(v) => signal::f32_from_bytes(&v),
        Err(e) => return vec![fail("width", "f32 reference", e)],
    };
    let d = signal::compare(&a, &b);
    let peak = a.iter().chain(b.iter()).fold(0.0f32, |m, s| m.max(s.abs()));
    // The gap must be real (proving internal width matters) and small (proving
    // neither build is broken). "Small" is relative to signal peak, not absolute.
    let rel = if peak > 0.0 {
        d.max_abs as f32 / peak
    } else {
        0.0
    };
    let name = format!(
        "f32-internal vs f64-internal on a {} signal",
        Kind::Partials
    );
    if d.differing == 0 {
        out.push(with_metric(
            fail(
                "width",
                name,
                "the two builds are identical, so nothing here justifies f64 state — \
                 check that -DG6CT_DOUBLE is really taking effect",
            ),
            diff_json(&d),
        ));
    } else if rel > 5e-3 {
        out.push(with_metric(
            fail(
                "width",
                name,
                format!(
                    "the f32 build is {} of peak away (worst sample {} relative) — too far \
                     for the same transfer function",
                    fmt_f(rel as f64),
                    fmt_f(d.max_rel)
                ),
            ),
            diff_json(&d),
        ));
    } else {
        out.push(with_metric(
            pass(
                "width",
                name,
                format!(
                    "{} of {} frames differ; max |d|={} ({:.1e} of peak {peak:.3}), worst \
                     sample {} relative; snr {} dB — internal width is worth about {} \
                     decimal digits on this material",
                    d.differing,
                    d.samples,
                    fmt_f(d.max_abs),
                    rel,
                    fmt_f(d.max_rel),
                    d.snr_db().map(fmt_f).unwrap_or_else(|| "inf".into()),
                    (d.max_abs as f64).max(1e-30).log10().abs().round() as i32,
                ),
            ),
            diff_json(&d),
        ));
    }
    // And the Rust implementation must sit on the f64 side, exactly. Its f32 path
    // is checked against the f32 build for the same reason: if either width drifted
    // from its own reference, the width comparison above would be meaningless.
    let r = rust_f64(&sig, 8, MASK_7_1, REF_BLOCK);
    let dr = signal::compare(&r, &a);
    let df = signal::compare(&r, &b);
    let r32 = rust_f32(&sig, 8, MASK_7_1, REF_BLOCK);
    let d32 = signal::compare(&r32, &b);
    out.push(if d32.exact() {
        pass(
            "width",
            "rust f32 path tracks the f32 build",
            format!(
                "0 of {} frames differ — both widths are faithful to their own reference",
                d32.samples
            ),
        )
    } else {
        info(
            "width",
            "rust f32 path tracks the f32 build",
            format!(
                "{} of {} frames differ (max {} ULP); the f32 path is a secondary build, \
                 the shipped one is f64",
                d32.differing, d32.samples, d32.max_ulp
            ),
        )
    });
    out.push(if dr.exact() && !df.exact() {
        pass(
            "width",
            "rust tracks the f64 build",
            format!(
                "0 of {} frames differ from f64-internal; {} differ from f32-internal",
                dr.samples, df.differing
            ),
        )
    } else if dr.exact() {
        info(
            "width",
            "rust tracks the f64 build",
            "identical to f64-internal; the f32 build happened to agree on this signal",
        )
    } else {
        fail(
            "width",
            "rust tracks the f64 build",
            format!("{} frames differ from the f64 reference", dr.differing),
        )
    });
    out
}

/// `chunk`: the render feeds the DSP in blocks chosen for memory and latency, not
/// for the DSP's convenience, so the answer must not depend on it.
pub fn chunk_invariance(builds: &[RefBuild], scratch: &Scratch, frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    for kind in [Kind::Partials, Kind::Impulse, Kind::Hot] {
        let sig = signal::generate(kind, frames, 8);
        let base = rust_f64(&sig, 8, MASK_7_1, 4096);
        for b in [1usize, 7, 13, 1024, 65_536] {
            let got = rust_f64(&sig, 8, MASK_7_1, b);
            let d = signal::compare(&got, &base);
            out.push(if d.exact() {
                pass(
                    "chunk",
                    format!("{kind}/block {b}"),
                    format!("identical to block 4096 across {} frames", d.samples),
                )
            } else {
                fail(
                    "chunk",
                    format!("{kind}/block {b}"),
                    format!(
                        "block size changed the output: {} frames differ, first at {:?} — \
                         filter state is not surviving the boundary",
                        d.differing, d.first_frame
                    ),
                )
            });
        }
        // And it must still equal the reference at its own 1024-frame block.
        if let Some(rb) = contract_builds(builds, "f64").first() {
            let inp = scratch.write_raw(&format!("c-{}.raw", kind.label()), &sig);
            match reference::run(rb, &inp, &scratch.path("c.raw"), MASK_7_1, 8) {
                Ok(v) => {
                    let d = signal::compare(&base, &signal::f32_from_bytes(&v));
                    out.push(if d.exact() {
                        pass(
                            "chunk",
                            format!("{kind}/rust 4096 vs reference 1024"),
                            "bit-identical across different block sizes in each implementation",
                        )
                    } else {
                        fail(
                            "chunk",
                            format!("{kind}/rust 4096 vs reference 1024"),
                            format!("{} frames differ", d.differing),
                        )
                    });
                }
                Err(e) => out.push(fail("chunk", format!("{kind}/reference run"), e)),
            }
        }
    }
    out
}

/// `continuity`: a file rendered in two sittings must equal one rendering.
///
/// This is the property the stage-parallel pipeline depends on: the DSP sees one
/// ordered block sequence while decode and encode run beside it. A DSP that
/// ignored its history would pass the first half of this and fail the second.
pub fn state_continuity(frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    for kind in [Kind::Partials, Kind::Sweep] {
        let sig = signal::generate(kind, frames, 8);
        let one = rust_f64(&sig, 8, MASK_7_1, 4096);

        // Same DSP object, two calls: what `render_file` actually does.
        let ch = 8usize;
        let slots = slot_map(8);
        let mut dsp = Dsp::<f64>::new(MASK_7_1);
        let mut two = Vec::with_capacity(one.len());
        for (start, len) in [(0usize, frames / 2), (frames / 2, frames - frames / 2)] {
            let mut planar = vec![vec![0.0f64; len]; ch];
            for c in 0..ch {
                for i in 0..len {
                    planar[c][i] = sig[(start + i) * ch + c] as f64;
                }
            }
            let mut chans: [Option<&[f64]>; SLOTS] = [None; SLOTS];
            for c in 0..ch {
                chans[slots[c]] = Some(&planar[c]);
            }
            let mut l = vec![0.0f64; len];
            let mut r = vec![0.0f64; len];
            dsp.process(&chans, &mut l, &mut r, len);
            for i in 0..len {
                two.push(l[i] as f32);
                two.push(r[i] as f32);
            }
        }
        let d = signal::compare(&one, &two);
        out.push(if d.exact() {
            pass(
                "continuity",
                format!("{kind}/split render"),
                format!("two sittings equal one render across {} frames", d.samples),
            )
        } else {
            fail(
                "continuity",
                format!("{kind}/split render"),
                format!(
                    "{} frames differ, first at frame {:?} — state is lost at the boundary",
                    d.differing, d.first_frame
                ),
            )
        });

        // The control: throwing the state away mid-file must change the answer.
        // Without this, a DSP with no memory would pass the check above for free.
        let mut fresh = Vec::with_capacity(one.len());
        for (start, len) in [(0usize, frames / 2), (frames / 2, frames - frames / 2)] {
            let mut dsp2 = Dsp::<f64>::new(MASK_7_1);
            let mut planar = vec![vec![0.0f64; len]; ch];
            for c in 0..ch {
                for i in 0..len {
                    planar[c][i] = sig[(start + i) * ch + c] as f64;
                }
            }
            let mut chans: [Option<&[f64]>; SLOTS] = [None; SLOTS];
            for c in 0..ch {
                chans[slots[c]] = Some(&planar[c]);
            }
            let mut l = vec![0.0f64; len];
            let mut r = vec![0.0f64; len];
            dsp2.process(&chans, &mut l, &mut r, len);
            for i in 0..len {
                fresh.push(l[i] as f32);
                fresh.push(r[i] as f32);
            }
        }
        let df = signal::compare(&one, &fresh);
        out.push(if df.differing > 0 {
            pass(
                "continuity",
                format!("{kind}/state really is load-bearing"),
                format!(
                    "resetting the filters mid-file changes {} of {} frames",
                    df.differing, df.samples
                ),
            )
        } else {
            fail(
                "continuity",
                format!("{kind}/state really is load-bearing"),
                "a mid-file state reset changed nothing, so the filters are stateless \
                 and the continuity check above proves nothing",
            )
        });
    }
    out
}

/// A per-slot timing summary, computed identically for both implementations.
#[derive(Clone, Debug, Default)]
pub struct Timing {
    /// Frame of the first non-zero output sample.
    pub first: Vec<i64>,
    /// Frame of the largest-magnitude output sample.
    pub peak: Vec<i64>,
    /// Energy centroid in frames — a group-delay-ish summary that catches a
    /// dispersed path even when the peak lines up.
    pub centroid: Vec<f64>,
    /// Polarity of the peak, per slot.
    pub sign: Vec<i8>,
}

/// Extract the timing summary from a stereo output produced by an impulse input
/// with a single channel excited (per-slot routing check).
pub fn timing_of(out: &[f32], frames: usize) -> Timing {
    let mut t = Timing::default();
    let mut peak_i = 0usize;
    let mut peak_v = 0.0f32;
    let mut first = None;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for f in 0..frames {
        let l = out[f * 2];
        let r = out[f * 2 + 1];
        let m = l.abs().max(r.abs());
        if m > 0.0 && first.is_none() {
            first = Some(f as i64);
        }
        if m > peak_v {
            peak_v = m;
            peak_i = f;
        }
        let e = (l as f64 * l as f64) + (r as f64 * r as f64);
        num += e * f as f64;
        den += e;
    }
    t.first.push(first.unwrap_or(-1));
    t.peak.push(peak_i as i64);
    t.centroid.push(if den > 0.0 { num / den } else { 0.0 });
    t.sign.push(if out[peak_i * 2] >= 0.0 { 1 } else { -1 });
    t
}

/// `phase`: per-slot impulse response timing, polarity and routing.
pub fn phase(builds: &[RefBuild], scratch: &Scratch, frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    let refs = contract_builds(builds, "f64");
    let Some(rb) = refs.first() else {
        return vec![skip("phase", "reference build", "no f64 reference binary")];
    };
    // One impulse at a time, so each slot's own path is measured in isolation.
    let mut rows: Vec<String> = Vec::new();
    let mut bad = 0;
    for slot in 0..8usize {
        let mut sig = vec![0.0f32; frames * 8];
        sig[16 * 8 + slot] = 1.0;
        let inp = scratch.write_raw(&format!("p{slot}.raw"), &sig);
        let want = match reference::run(rb, &inp, &scratch.path("p.raw"), MASK_7_1, 8) {
            Ok(v) => signal::f32_from_bytes(&v),
            Err(e) => {
                out.push(fail("phase", format!("slot {slot}"), e));
                bad += 1;
                continue;
            }
        };
        let got = rust_f64(&sig, 8, MASK_7_1, REF_BLOCK);
        let d = signal::compare(&got, &want);
        let t = timing_of(&want, frames);
        rows.push(format!(
            "{} {}",
            ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"][slot],
            json!({
                "first": t.first[0],
                "peak": t.peak[0],
                "centroid": (t.centroid[0] * 100.0).round() / 100.0,
                "peak_sign": t.sign[0],
            })
        ));
        let name = format!(
            "slot {slot} ({})",
            ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"][slot]
        );
        if d.exact() {
            out.push(with_metric(
                pass(
                    "phase",
                    name,
                    "impulse response bit-identical, so every delay, \
                                      grouping and polarity matches by construction",
                ),
                json!({
                    "first_frame": t.first[0],
                    "peak_frame": t.peak[0],
                    "energy_centroid_frames": t.centroid[0],
                    "peak_sign": t.sign[0],
                }),
            ));
        } else {
            bad += 1;
            out.push(with_metric(
                fail(
                    "phase",
                    name,
                    format!(
                        "{} frames differ (max {} ULP, first at {:?})",
                        d.differing, d.max_ulp, d.first_frame
                    ),
                ),
                diff_json(&d),
            ));
        }
    }
    out.push(info(
        "phase",
        "per-slot response (reference)",
        rows.join("  "),
    ));
    if bad > 0 {
        out.push(fail("phase", "summary", format!("{bad} slots disagreed")));
    }
    out
}

/// The counter-example, run as a check in its own right.
///
/// Built from the *same C source* with `-ffp-contract=fast`, i.e. what happens when
/// nobody tells the compiler that floating point has a contract. If that build
/// diverges from the canonical one then the flags in `port/Makefile` are doing real
/// work and every bit-exact statement above depends on them; if it does not diverge,
/// this toolchain simply found nothing to fuse, which is reported rather than
/// claimed.
pub fn adversarial(builds: &[RefBuild], scratch: &Scratch, frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    let refs = contract_builds(builds, "f64");
    let Some(strict) = refs.first() else {
        return vec![skip("contraction", "canonical reference", "not built")];
    };
    let Some(loose) = loose_build(builds, "f64") else {
        return vec![skip(
            "contraction",
            "-ffp-contract=fast reference",
            "not built",
        )];
    };
    let mut diverged = 0usize;
    let mut worst = Diff::default();
    let mut notes: Vec<String> = Vec::new();
    for kind in [Kind::Partials, Kind::Silence, Kind::Impulse] {
        let sig = signal::generate(kind, frames, 8);
        let inp = scratch.write_raw(&format!("adv-{}.raw", kind.label()), &sig);
        let a = match reference::run(strict, &inp, &scratch.path("a.raw"), MASK_7_1, 8) {
            Ok(v) => signal::f32_from_bytes(&v),
            Err(e) => return vec![fail("contraction", "canonical reference run", e)],
        };
        let b = match reference::run(loose, &inp, &scratch.path("b.raw"), MASK_7_1, 8) {
            Ok(v) => signal::f32_from_bytes(&v),
            Err(e) => return vec![fail("contraction", "contract=fast reference run", e)],
        };
        let d = signal::compare(&a, &b);
        if d.differing > 0 {
            diverged += 1;
            if d.max_rel >= worst.max_rel {
                worst = d;
            }
        }
        notes.push(format!(
            "{kind}: {} frames differ (max rel {:e})",
            d.differing, d.max_rel
        ));
    }
    if diverged > 0 {
        out.push(with_metric(
            pass(
                "contraction",
                "-ffp-contract=off is load-bearing",
                format!(
                    "{diverged} of 3 signals change when the same C source is built with \
                     -ffp-contract=fast; worst max |d|={} max rel {:e} — so the DSP flags are \
                     what make a build of this module interchangeable, and why Rust must not \
                     contract either ({})",
                    fmt_f(worst.max_abs),
                    worst.max_rel,
                    notes.join("; ")
                ),
            ),
            diff_json(&worst),
        ));
    } else {
        out.push(info(
            "contraction",
            "-ffp-contract=off is load-bearing",
            format!(
                "this toolchain produced identical code either way ({}); the flag is still \
                 required, because another compiler or another -march may fuse",
                notes.join("; ")
            ),
        ));
    }
    out
}

/// `drift`: one minute of material, looking for anything that accumulates.
pub fn drift(builds: &[RefBuild], scratch: &Scratch, seconds: u64) -> Vec<Check> {
    let mut out = Vec::new();
    let refs = contract_builds(builds, "f64");
    let Some(rb) = refs.first() else {
        return vec![skip("drift", "reference build", "no f64 reference binary")];
    };
    let frames = (seconds * 48000) as usize;
    let sig = signal::generate(Kind::Partials, frames, 8);
    let inp = scratch.write_raw("drift.raw", &sig);
    let t0 = std::time::Instant::now();
    let want = match reference::run(rb, &inp, &scratch.path("drift.out.raw"), MASK_7_1, 8) {
        Ok(v) => signal::f32_from_bytes(&v),
        Err(e) => return vec![fail("drift", "reference run", e)],
    };
    let c_wall = t0.elapsed();
    let t1 = std::time::Instant::now();
    let got = rust_f64(&sig, 8, MASK_7_1, 4096);
    let r_wall = t1.elapsed();
    let d = signal::compare(&got, &want);
    out.push(with_metric(
        if d.exact() {
            pass(
                "drift",
                format!("{seconds}s bit-exact"),
                format!(
                    "{} stereo frames, 0 differing; rust {} ms vs reference {} ms \
                     (reference includes process start-up)",
                    d.samples,
                    r_wall.as_millis(),
                    c_wall.as_millis()
                ),
            )
        } else {
            fail(
                "drift",
                format!("{seconds}s bit-exact"),
                format!(
                    "{} of {} frames differ; first at frame {:?}, max {} ULP — divergence \
                     accumulates rather than staying bounded",
                    d.differing, d.samples, d.first_frame, d.max_ulp
                ),
            )
        },
        diff_json(&d),
    ));

    // Stability, independent of the reference: no NaN/inf, and the tail must not
    // be larger than the head (an unstable recursion grows).
    let finite = got.iter().all(|s| s.is_finite());
    let rms = |v: &[f32]| {
        let n = v.len().max(1) as f64;
        (v.iter().map(|s| *s as f64 * *s as f64).sum::<f64>() / n).sqrt()
    };
    let half = got.len() / 2;
    let (head, tail) = (rms(&got[..half]), rms(&got[half..]));
    out.push(if finite && tail <= head * 2.0 + 1e-6 {
        pass(
            "drift",
            "stability over the whole file",
            format!(
                "all finite; rms head {head:.5} tail {tail:.5}, peak {:.4} — no growth",
                got.iter().fold(0.0f32, |m, s| m.max(s.abs()))
            ),
        )
    } else {
        fail(
            "drift",
            "stability over the whole file",
            format!(
                "finite={finite}, rms head {head:.6} tail {tail:.6} — the recursion is growing"
            ),
        )
    });
    out
}

/// Behavioural FMA detector: no disassembler needed, and it tests the code the
/// verifier is actually running in.
///
/// `(1+2^-52)^2 - (1+2^-51)` is exactly `2^-104`. Separately rounded mul then add
/// gives exactly zero; a fused multiply-add keeps the unrounded product and gives
/// `2^-104`. So a non-zero answer means the compiler contracted — which in a
/// recursive IIR is a different filter, not a faster one of the same filter.
pub fn contraction() -> Vec<Check> {
    let out = vec![check_f64_contraction()];
    out
}

fn check_f64_contraction() -> Check {
    let a = 1.0f64 + f64::EPSILON;
    let b = 1.0f64 + f64::EPSILON;
    let c = -(1.0f64 + 2.0 * f64::EPSILON);
    // Written so the compiler cannot fold it at compile time: the values come from
    // a black_box and the result is read back.
    let (a, b, c) = (
        std::hint::black_box(a),
        std::hint::black_box(b),
        std::hint::black_box(c),
    );
    let sep = {
        let p = std::hint::black_box(a * b);
        std::hint::black_box(p + c)
    };
    // The exact answer is 2^-104; anything else means contraction happened.
    let fused = 1.0f64 / (1u128 << 104) as f64;
    if sep == 0.0 {
        pass(
            "contraction",
            "no FMA in this binary's f64 arithmetic",
            format!(
                "(1+2^-52)^2-(1+2^-51) = {sep}, as separate rounding requires; a fused \
                 multiply-add would give {fused:e}"
            ),
        )
    } else {
        fail(
            "contraction",
            "no FMA in this binary's f64 arithmetic",
            format!(
                "got {sep:e} instead of 0 — mul+add is being contracted, which changes the \
                 recursive filter's coefficients (build without -C fp-contract=fast)"
            ),
        )
    }
}

/// `asm`: agree with the disassembler, and check the reference binaries too.
pub fn asm_guard(builds: &[RefBuild]) -> Vec<Check> {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => return vec![skip("asm", "current executable", e.to_string())],
    };
    let mut out = Vec::new();
    out.push(scan_binary("rust dsp", &exe, &["g6me"], ExpectFma::No));
    for b in contract_builds(builds, "f64") {
        // The reference must show no FMA either: that is what `-ffp-contract=off`
        // is for, and it is the assumption every bit-exact result above rests on.
        let mut c = scan_binary(
            "reference",
            &b.path,
            &["g6ct_process", "g6ct_init"],
            ExpectFma::No,
        );
        c.name = format!("{} / no FMA", b.name);
        out.push(c);
    }
    // The counter-example is expected to fuse. If it does not, that is worth knowing
    // too, because it says the divergence was produced some other way.
    if let Some(loose) = loose_build(builds, "f64") {
        let mut c = scan_binary(
            "reference",
            &loose.path,
            &["g6ct_process", "g6ct_init"],
            ExpectFma::Expected,
        );
        c.name = format!("{} / fma expected", loose.name);
        out.push(c);
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExpectFma {
    /// A DSP path that must stay unfused.
    No,
    /// The deliberate counter-example.
    Expected,
}

/// Look for fused multiply/add mnemonics inside the interesting functions.
fn scan_binary(what: &str, path: &PathBuf, want_symbols: &[&str], expect: ExpectFma) -> Check {
    let Some(tool) = find_objdump() else {
        return skip(
            "asm",
            format!("{what} disassembly"),
            "no objdump available to inspect codegen",
        );
    };
    let o = std::process::Command::new(&tool)
        .arg("-d")
        .arg("--no-show-raw-insn")
        .arg(path)
        .output();
    let Ok(o) = o else {
        return skip(
            "asm",
            format!("{what} disassembly"),
            format!("{tool} could not read {}", path.display()),
        );
    };
    let txt = String::from_utf8_lossy(&o.stdout).to_string();
    if txt.trim().is_empty() {
        return skip(
            "asm",
            format!("{what} disassembly"),
            format!("{tool} produced no disassembly for {}", path.display()),
        );
    }
    // objdump prints one block per symbol:
    //   0000000000001160 <g6ct_process>:
    //   1160:  f2 0f 10 05 …   movsd 0x…(%rip),%xmm1
    // so a symbol boundary is any line containing `<name>`.
    let fma_mnemonics = [
        "vfmadd", "vfmsub", "vfnmadd", "vfnmsub", // x86 AVX-FMA
        "fmadd", "fmsub", "fnmadd", "fnmsub", // aarch64
        "fma.",   // ppc
    ];
    let mut inside = false;
    let mut hits = 0usize;
    let mut inspected = 0usize;
    let mut insides = 0usize;
    for line in txt.lines() {
        // A symbol header is `<address> <name>:` and nothing else. Instruction lines
        // carry `<sym+off>` annotations too (rip-relative loads, call targets); treating
        // those as headers skipped exactly the FMA lines that load a constant.
        if let Some(sym) = symbol_header(line) {
            inside = want_symbols.iter().any(|w| sym.contains(w));
            if inside {
                inspected += 1;
            }
            continue;
        }
        if !inside {
            continue;
        }
        insides += 1;
        if fma_mnemonics.iter().any(|m| mnemonic(line).starts_with(m)) {
            hits += 1;
        }
    }
    if inspected == 0 {
        return skip(
            "asm",
            format!("{what} disassembly"),
            format!(
                "no symbols matching {:?} in {} (stripped?)",
                want_symbols,
                path.display()
            ),
        );
    }
    match (expect, hits) {
        (ExpectFma::No, 0) => pass(
            "asm",
            format!("{what}: no fused mul-add"),
            format!(
                "{inspected} symbol block(s), {insides} instruction(s) scanned, no FMA mnemonic"
            ),
        ),
        (ExpectFma::No, n) => fail(
            "asm",
            format!("{what}: no fused mul-add"),
            format!("{n} FMA instruction(s) inside the DSP code — bit-exactness is void"),
        ),
        (ExpectFma::Expected, n) if n > 0 => pass(
            "asm",
            format!("{what}: fma present as expected"),
            format!(
                "{n} FMA instruction(s) — this is the counter-example build, so the \
                 divergence measured against it really is fusion"
            ),
        ),
        (ExpectFma::Expected, _) => info(
            "asm",
            format!("{what}: fma expected but absent"),
            "the compiler chose not to fuse here, so any divergence it caused came from a \
             different transformation",
        ),
    }
}

/// `    1194:\tvfmadd132sd 0xe6b(%rip),...` -> `vfmadd132sd`: the first word after the
/// address. (GNU objdump separates mnemonic and operands with a space, llvm-objdump
/// with a tab, so "the last tab-separated field" is the operands for one of them.)
fn mnemonic(line: &str) -> &str {
    line.split_once(':')
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .unwrap_or("")
}

/// `0000000000001190 <g6ct_process>:` -> `g6ct_process`.
fn symbol_header(line: &str) -> Option<&str> {
    let (addr, rest) = line.trim_end().split_once(' ')?;
    if addr.is_empty() || !addr.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    rest.strip_prefix('<')?.strip_suffix(">:")
}

fn find_objdump() -> Option<String> {
    for t in ["objdump", "llvm-objdump", "gobjdump"] {
        if std::process::Command::new(t)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return Some(t.to_string());
        }
    }
    None
}

fn with_metric(mut c: Check, m: Value) -> Check {
    c.metric = Some(m);
    c
}

fn diff_json(d: &Diff) -> Value {
    json!({
        "frames": d.samples,
        "differing": d.differing,
        "signed_zero": d.signed_zero,
        "max_abs": d.max_abs,
        "max_rel": d.max_rel,
        "max_ulp": d.max_ulp,
        "first_frame": d.first_frame,
        "snr_db": d.snr_db(),
    })
}

fn fmt_f(v: f64) -> String {
    if v == 0.0 {
        "0".to_string()
    } else if v.is_infinite() {
        "inf".to_string()
    } else {
        format!("{v:.3e}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disassembly_lines_are_classified_by_shape_not_by_angle_brackets() {
        // GNU objdump on a function whose FMA loads its constant rip-relative.
        let fma =
            "    1194:\tvfmadd132sd 0xe6b(%rip),%xmm1,%xmm0        # 2008 <_IO_stdin_used+0x8>";
        assert_eq!(symbol_header(fma), None);
        assert_eq!(mnemonic(fma), "vfmadd132sd");
        assert_eq!(
            symbol_header("0000000000001190 <g6ct_process>:"),
            Some("g6ct_process")
        );
        assert_eq!(symbol_header("    11a0:\tcall   1050 <memcpy@plt>"), None);
        // llvm-objdump separates mnemonic and operands with a tab.
        assert_eq!(
            mnemonic("    1194:      \tvfmadd231sd\t%xmm2, %xmm1, %xmm0"),
            "vfmadd231sd"
        );
    }

    #[test]
    fn slot_tables_match_the_reference_cli() {
        assert_eq!(slot_map(8), vec![0, 1, 2, 3, 4, 5, 9, 10]);
        assert_eq!(slot_map(12), (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn rust_run_produces_stereo_of_the_right_length() {
        let sig = signal::generate(Kind::Partials, 3000, 8);
        let out = rust_f64(&sig, 8, MASK_7_1, 1024);
        assert_eq!(out.len(), 3000 * 2);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn a_single_impulse_does_not_stay_in_the_first_frame() {
        // The filters are recursive, so energy must spread out over time; if the
        // DSP ever became a straight pass-through, this timing check would catch it.
        let mut sig = vec![0.0f32; 4096 * 8];
        sig[16 * 8] = 1.0;
        let out = rust_f64(&sig, 8, MASK_7_1, 1024);
        let t = timing_of(&out, 4096);
        assert!(t.peak[0] > 16, "peak moved to {}", t.peak[0]);
        assert!(t.centroid[0] > t.first[0] as f64);
    }

    /// Dekker's two-product: `p` is the rounded product and `e` is the exact
    /// residual, so `a*b == p + e` with no rounding. Needs no FMA, which is the
    /// point — it lets us show what a fused instruction would have kept.
    fn two_product(a: f64, b: f64) -> (f64, f64) {
        let p = a * b;
        let split = 134_217_729.0f64; // 2^27 + 1
        let ca = split * a;
        let ah = ca - (ca - a);
        let al = a - ah;
        let cb = split * b;
        let bh = cb - (cb - b);
        let bl = b - bh;
        let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
        (p, e)
    }

    #[test]
    fn contraction_guard_is_not_vacuous() {
        let a = 1.0f64 + f64::EPSILON;
        let c = -(1.0f64 + 2.0 * f64::EPSILON);
        let (p, e) = two_product(a, a);
        // The guard's expression must have a genuinely non-zero discarded residual:
        // if this were zero, both fused and unfused would answer 0 and the guard
        // would be measuring nothing.
        assert!(e > 0.0, "residual of (1+2^-52)^2 is {e}, expected 2^-104");
        // Unfused: the product is rounded to p first, and p + c cancels exactly.
        assert_eq!(p + c, 0.0, "the unfused evaluation must be exactly zero");
        // The rounded product must be exactly 1+2^-51: that is what makes `p + c`
        // cancel to zero, and therefore what makes a non-zero answer proof of fusion.
        assert_eq!(
            p,
            1.0 + 2.0 * f64::EPSILON,
            "rounded product should be exactly 1+2^-51"
        );
        let checks = contraction();
        assert_eq!(checks.len(), 1);
        assert_eq!(
            checks[0].status,
            Status::Pass,
            "guard reported {:?}",
            checks[0].detail
        );
    }
}
