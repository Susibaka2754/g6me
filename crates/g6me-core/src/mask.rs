//! Channel identity: WAVEFORMATEXTENSIBLE slots, masks, and the mapping from
//! what a container/decoder reports to the slot table the DSP expects.

use crate::dsp::*;
use serde::{Deserialize, Serialize};

/// A channel of the module's internal table.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum Slot {
    Fl,
    Fr,
    Fc,
    Lfe,
    Bl,
    Br,
    Flc,
    Frc,
    Bc,
    Sl,
    Sr,
    Tc,
}

impl Slot {
    pub const ALL: [Slot; 12] = [
        Slot::Fl,
        Slot::Fr,
        Slot::Fc,
        Slot::Lfe,
        Slot::Bl,
        Slot::Br,
        Slot::Flc,
        Slot::Frc,
        Slot::Bc,
        Slot::Sl,
        Slot::Sr,
        Slot::Tc,
    ];

    pub fn bit(self) -> u32 {
        match self {
            Slot::Fl => CH_FL,
            Slot::Fr => CH_FR,
            Slot::Fc => CH_FC,
            Slot::Lfe => CH_LFE,
            Slot::Bl => CH_BL,
            Slot::Br => CH_BR,
            Slot::Flc => CH_FLC,
            Slot::Frc => CH_FRC,
            Slot::Bc => CH_BC,
            Slot::Sl => CH_SL,
            Slot::Sr => CH_SR,
            Slot::Tc => CH_TC,
        }
    }

    /// Index into the DSP's `ch[SLOTS]`.
    pub fn index(self) -> usize {
        slot_of(self.bit())
    }

    pub fn name(self) -> &'static str {
        match self {
            Slot::Fl => "FL",
            Slot::Fr => "FR",
            Slot::Fc => "FC",
            Slot::Lfe => "LFE",
            Slot::Bl => "BL",
            Slot::Br => "BR",
            Slot::Flc => "FLC",
            Slot::Frc => "FRC",
            Slot::Bc => "BC",
            Slot::Sl => "SL",
            Slot::Sr => "SR",
            Slot::Tc => "TC",
        }
    }

    pub fn parse(s: &str) -> Option<Slot> {
        let up = s.trim().to_ascii_uppercase();
        Slot::ALL.into_iter().find(|v| v.name() == up)
    }
}

/// Channel mask implied by a stream order.
pub fn mask_of(order: &[Slot]) -> u32 {
    order.iter().fold(0u32, |m, s| m | s.bit())
}

/// A stream can be downmixed only if it carries a front pair; everything else
/// is refused rather than silently folded.
pub fn is_downmixable(order: &[Slot]) -> bool {
    let m = mask_of(order);
    m & CH_FL != 0 && m & CH_FR != 0
}

/// Pretty-print an order, e.g. `FL FR FC LFE BL BR SL SR`.
pub fn format_order(order: &[Slot]) -> String {
    order.iter().map(|s| s.name()).collect::<Vec<_>>().join(" ")
}

/// Mask as the hex literal ffmpeg's `-channel_layout` accepts, e.g. `0x63f`.
pub fn format_mask_hex(mask: u32) -> String {
    format!("0x{mask:x}")
}

/// Well-known orders, in the interleaved order ffmpeg/AVFoundation emit them.
pub const ORDERS: &[(&str, &[Slot])] = &[
    (
        "7.1 (FL FR FC LFE BL BR SL SR)",
        &[
            Slot::Fl,
            Slot::Fr,
            Slot::Fc,
            Slot::Lfe,
            Slot::Bl,
            Slot::Br,
            Slot::Sl,
            Slot::Sr,
        ],
    ),
    (
        "7.1 front-wide (FL FR FC LFE BL BR FLC FRC)",
        &[
            Slot::Fl,
            Slot::Fr,
            Slot::Fc,
            Slot::Lfe,
            Slot::Bl,
            Slot::Br,
            Slot::Flc,
            Slot::Frc,
        ],
    ),
    (
        "5.1 (FL FR FC LFE BL BR)",
        &[Slot::Fl, Slot::Fr, Slot::Fc, Slot::Lfe, Slot::Bl, Slot::Br],
    ),
    (
        "5.1 side (FL FR FC LFE SL SR)",
        &[Slot::Fl, Slot::Fr, Slot::Fc, Slot::Lfe, Slot::Sl, Slot::Sr],
    ),
    (
        "6.1 (FL FR FC LFE BL BC BR)",
        &[
            Slot::Fl,
            Slot::Fr,
            Slot::Fc,
            Slot::Lfe,
            Slot::Bl,
            Slot::Bc,
            Slot::Br,
        ],
    ),
    (
        "quad (FL FR BL BR)",
        &[Slot::Fl, Slot::Fr, Slot::Bl, Slot::Br],
    ),
    (
        "quad side (FL FR SL SR)",
        &[Slot::Fl, Slot::Fr, Slot::Sl, Slot::Sr],
    ),
    ("stereo (FL FR)", &[Slot::Fl, Slot::Fr]),
];

/// Map a decoder-reported layout name (and channel count) to a stream order.
///
/// ffmpeg's names are matched case-insensitively. `None` layout with a
/// channel count we can reason about falls back to the convention that
/// containers actually use (8ch => 7.1, 6ch => 5.1, 2ch => stereo); anything
/// else is reported as unsupported so the caller can ask for an explicit order
/// instead of guessing and folding a surround channel into the wrong ear.
pub fn order_from_layout(layout: Option<&str>, channels: usize) -> Option<Vec<Slot>> {
    use Slot::*;
    // Exactly ffmpeg's standard layouts (`ffmpeg -layouts`), each in the native order the
    // decoder emits its planes: ascending channel id, which is ascending slot bit here.
    // Membership matters as much as order: the decode chain relabels the planes with this
    // order, so a name read as the wrong channel set routes a speaker to the wrong slot.
    let named: &[(&str, &[Slot])] = &[
        ("mono", &[Fc]),
        ("stereo", &[Fl, Fr]),
        ("2.1", &[Fl, Fr, Lfe]),
        ("3.0", &[Fl, Fr, Fc]),
        ("3.0(back)", &[Fl, Fr, Bc]),
        ("4.0", &[Fl, Fr, Fc, Bc]),
        ("quad", &[Fl, Fr, Bl, Br]),
        ("quad(side)", &[Fl, Fr, Sl, Sr]),
        ("3.1", &[Fl, Fr, Fc, Lfe]),
        ("5.0", &[Fl, Fr, Fc, Bl, Br]),
        ("5.0(side)", &[Fl, Fr, Fc, Sl, Sr]),
        ("4.1", &[Fl, Fr, Fc, Lfe, Bc]),
        ("5.1", &[Fl, Fr, Fc, Lfe, Bl, Br]),
        ("5.1(side)", &[Fl, Fr, Fc, Lfe, Sl, Sr]),
        ("6.0", &[Fl, Fr, Fc, Bc, Sl, Sr]),
        ("6.0(front)", &[Fl, Fr, Flc, Frc, Sl, Sr]),
        ("hexagonal", &[Fl, Fr, Fc, Bl, Br, Bc]),
        ("6.1", &[Fl, Fr, Fc, Lfe, Bc, Sl, Sr]),
        ("6.1(back)", &[Fl, Fr, Fc, Lfe, Bl, Br, Bc]),
        ("6.1(front)", &[Fl, Fr, Lfe, Flc, Frc, Sl, Sr]),
        ("7.0", &[Fl, Fr, Fc, Bl, Br, Sl, Sr]),
        ("7.0(front)", &[Fl, Fr, Fc, Flc, Frc, Sl, Sr]),
        ("7.1", &[Fl, Fr, Fc, Lfe, Bl, Br, Sl, Sr]),
        // Not one of ffmpeg's names, but one some taggers write for the same layout.
        ("7.1(side)", &[Fl, Fr, Fc, Lfe, Bl, Br, Sl, Sr]),
        ("7.1(wide)", &[Fl, Fr, Fc, Lfe, Bl, Br, Flc, Frc]),
        ("7.1(wide-side)", &[Fl, Fr, Fc, Lfe, Flc, Frc, Sl, Sr]),
        ("octagonal", &[Fl, Fr, Fc, Bl, Br, Bc, Sl, Sr]),
    ];

    if let Some(l) = layout {
        let key = l.trim().to_ascii_lowercase();
        if let Some((_, slots)) = named.iter().find(|(n, _)| *n == key) {
            // The name and the channel count have to agree. Returning the table's two slots
            // for a header that names `stereo` and declares eight channels hands the caller a
            // two-plane order for an eight-plane stream, and the six planes that never made
            // it into the order are simply not in the output. Contradictory metadata gets an
            // error, not a preference.
            return (slots.len() == channels).then(|| slots.to_vec());
        }
        // A name that exists but is not in the table is refused rather than guessed
        // from the channel count. "The container says octagon" and "the container says
        // nothing" are different statements, and answering the first one with the 8ch
        // convention is how a rear-centre ends up in a side-surround — the guess would
        // be right for the common cases and wrong in exactly the cases that got us here.
        // Junk names are not statements at all, so they keep falling through.
        if !key.is_empty() && key != "unknown" && key != "undefined" {
            return None;
        }
    }

    match channels {
        8 => Some(vec![Fl, Fr, Fc, Lfe, Bl, Br, Sl, Sr]),
        6 => Some(vec![Fl, Fr, Fc, Lfe, Bl, Br]),
        2 => Some(vec![Fl, Fr]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Slot::*;

    #[test]
    fn masks_match_the_c_reference() {
        assert_eq!(mask_of(&[Fl, Fr, Fc, Lfe, Bl, Br, Sl, Sr]), MASK_7_1);
        assert_eq!(mask_of(&[Fl, Fr, Fc, Lfe, Bl, Br, Flc, Frc]), MASK_8CH);
        assert_eq!(mask_of(&[Fl, Fr, Fc, Lfe, Bl, Br]), MASK_5_1);
        assert_eq!(mask_of(&[Fl, Fr]), MASK_STEREO);
    }

    #[test]
    fn slot_indices_are_wfx_bit_positions() {
        assert_eq!(Slot::Fl.index(), 0);
        assert_eq!(Slot::Lfe.index(), 3);
        assert_eq!(Slot::Bc.index(), 8);
        assert_eq!(Slot::Sl.index(), 9);
        assert_eq!(Slot::Sr.index(), 10);
        assert_eq!(Slot::Tc.index(), 11);
    }

    #[test]
    fn layout_mapping() {
        assert_eq!(
            order_from_layout(Some("7.1"), 8).unwrap(),
            vec![Fl, Fr, Fc, Lfe, Bl, Br, Sl, Sr]
        );
        // Atmos beds often report 7.1; unknown 8ch still resolves.
        assert_eq!(order_from_layout(Some("unknown"), 8).unwrap().len(), 8);
        assert!(order_from_layout(Some("unknown"), 14).is_none());
        assert_eq!(order_from_layout(None, 6).unwrap().len(), 6);
    }

    /// Every named layout must be listed in the order the decoder actually emits it.
    ///
    /// ffmpeg stores a native layout's channels in ascending channel-id order, and the ids
    /// here are the WAVEFORMATEXTENSIBLE bit positions (`Slot::ALL`). A table entry whose
    /// slots are not ascending assigns a sample plane to the wrong speaker: `6.1` written
    /// as `FL FR FC LFE BL BC BR` reads rear-centre out of the rear-right plane and
    /// rear-right out of rear-centre, and nothing downstream can tell, because the matrix
    /// receives a plausibly-shaped order and folds it into the wrong ear.
    ///
    /// Checked through the public entry point rather than against the table, so it still
    /// bites if a name moves to a different lookup on the way there.
    #[test]
    fn named_layouts_are_in_the_order_the_decoder_emits_them() {
        // (ffmpeg's name, channel count)
        let names = [
            ("mono", 1),
            ("stereo", 2),
            ("2.1", 3),
            ("3.0", 3),
            ("3.0(back)", 3),
            ("quad", 4),
            ("4.0", 4),
            ("quad(side)", 4),
            ("3.1", 4),
            ("5.0", 5),
            ("5.0(side)", 5),
            ("4.1", 5),
            ("5.1", 6),
            ("5.1(side)", 6),
            ("6.0", 6),
            ("6.0(front)", 6),
            ("hexagonal", 6),
            ("6.1", 7),
            ("6.1(back)", 7),
            ("6.1(front)", 7),
            ("7.0", 7),
            ("7.0(front)", 7),
            ("7.1", 8),
            ("7.1(wide)", 8),
            ("7.1(wide-side)", 8),
            ("octagonal", 8),
        ];
        for (name, want) in names {
            let got = order_from_layout(Some(name), want)
                .unwrap_or_else(|| panic!("{name} stopped being supported"));
            assert_eq!(got.len(), want, "{name} lost or gained a channel");
            let bits: Vec<u32> = got.iter().map(|s| s.bit()).collect();
            let mut sorted = bits.clone();
            sorted.sort_unstable();
            assert_eq!(
                bits, sorted,
                "{name} is listed as {bits:?}; the decoder emits {sorted:?}"
            );
            assert_eq!(
                sorted.len(),
                sorted
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                "{name} lists one slot twice: {bits:?}"
            );
        }

        // A name we recognise as absent is refused, not guessed. Before, any unlisted name
        // fell through to the channel-count convention, so a header claiming `stereo` while
        // declaring eight channels was read as a 7.1 file: the metadata contradicts itself
        // and the guess picks a side. Refused, the caller says "give me an explicit --order",
        // which is the only honest answer to a self-contradictory header.
        assert!(
            order_from_layout(Some("stereo"), 8).is_none(),
            "a layout name we know, disagreeing with the channel count, was guessed anyway"
        );
        assert!(order_from_layout(Some("5.1(side)"), 8).is_none());
        // Junk names carry no information, so those still resolve by count — Atmos beds
        // report 7.1 or nothing at all, and refusing them would refuse real renders.
        assert_eq!(order_from_layout(Some("unknown"), 8).unwrap().len(), 8);
        assert_eq!(order_from_layout(Some(""), 6).unwrap().len(), 6);
        assert_eq!(order_from_layout(None, 8).unwrap().len(), 8);
    }

    #[test]
    fn mono_is_not_downmixable() {
        assert!(!is_downmixable(&[Fc]));
        assert!(is_downmixable(&[Fl, Fr]));
    }

    /// The table must say what ffmpeg says each name means. Skipped (with a note) when no
    /// ffmpeg is on PATH, like the other tests that need one.
    #[test]
    fn layout_table_agrees_with_ffmpeg() {
        let Ok(out) = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-layouts"])
            .output()
        else {
            eprintln!("skipped: no ffmpeg on PATH");
            return;
        };
        let text = String::from_utf8_lossy(&out.stdout);
        let mut checked = 0;
        for line in text.lines() {
            let mut parts = line.split_whitespace();
            let (Some(name), Some(decomp), None) = (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            let Some(want) = decomp
                .split('+')
                .map(Slot::parse)
                .collect::<Option<Vec<Slot>>>()
            else {
                continue; // uses speakers outside the module's table (tops, wides, ...)
            };
            let got = order_from_layout(Some(name), want.len());
            assert_eq!(got, Some(want), "ffmpeg's '{name}' is {decomp}");
            checked += 1;
        }
        assert!(
            checked >= 25,
            "only {checked} layouts compared; parsing broke?"
        );
    }

    #[test]
    fn mask_hex_for_ffmpeg() {
        assert_eq!(format_mask_hex(MASK_7_1), "0x63f");
    }
}
