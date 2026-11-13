//! Small shared helpers.

use std::ffi::{OsStr, OsString};
use std::path::Path;

/// One filesystem path, spelled so that `ffmpeg` and `ffprobe` read it as a path.
///
/// Both tools parse a bare argument as `[protocol:]target` before they look at the
/// filesystem, so a relative name holding a `:` before its first `/` is read as a
/// protocol spec and dies. Measured on the installed FFmpeg 9.0.2, on both sides of
/// the pipe:
///
/// ```text
/// $ ffmpeg -i a:b.wav -f null -     -> Error opening input: Protocol not found
/// $ ffmpeg ... out:v.wav            -> Error opening output file out:v.wav:
///                                       Protocol not found
/// ```
///
/// The same name with `./` in front opens both ways. A leading `-` is a separate
/// hazard on the output side, where the name is positional and therefore looks like
/// an option. Prepending `./` settles all of it without moving the file, which is
/// why it is preferred over `file:`: the prefix works identically for the output
/// path, where the failure is the encoder refusing to start at all.
///
/// `OsString` is the point of this function, not decoration. `Path::display()`
/// substitutes U+FFFD for every byte it cannot decode, so a name carrying a stray
/// 0xFF — legal on Unix, and ordinary for files copied off a Windows disc — turns
/// into a name that does not exist on disk. `as_encoded_bytes` keeps the bytes
/// exactly; nothing here decodes them and nothing re-encodes them.
///
/// Only call this with something you mean as a file. A caller that wants a protocol
/// (`pipe:0`, `fd:3`) passes its own literal and does not come through here.
#[must_use]
pub fn ffmpeg_path_arg(p: &Path) -> OsString {
    let os = p.as_os_str();
    let b = os.as_encoded_bytes();
    // Absolute names (including a Windows `C:` prefix and `\\host\share`) cannot be
    // read as anything but a path, and an empty one is not our business.
    if os.is_empty() || p.is_absolute() {
        return os.to_os_string();
    }
    // A bare `-` is the pipeline's stdout marker, and `./-` is a file named `-`
    // inside the current directory: prefixing it would send the decoded samples to
    // a file instead of to the reader on the other end of this child.
    if b == b"-" || b == b"." || b == b".." {
        return os.to_os_string();
    }
    // Already marked as relative, so already unambiguous.
    if b.starts_with(b"./") || b.starts_with(b"../") {
        return os.to_os_string();
    }
    let looks_like_option = b.first() == Some(&b'-');
    let looks_like_protocol = b.iter().take_while(|&&c| c != b'/').any(|&c| c == b':');
    if looks_like_option || looks_like_protocol {
        let mut out = OsString::from("./");
        out.push(os);
        out
    } else {
        os.to_os_string()
    }
}

/// The same path for a line a human reads: a dry-run preview, a log, JSON.
///
/// Lossy by definition, and honest about it — the bytes that reach the process come
/// from [`ffmpeg_path_arg`], never from this. A name that is not valid UTF-8 shows
/// here with the usual replacement characters while still rendering correctly on
/// disk.
#[must_use]
pub fn path_display(p: &Path) -> String {
    ffmpeg_path_arg(p).to_string_lossy().into_owned()
}

/// A `Vec<String>` copy of an argument list, for JSON and for the preview pane.
///
/// Lossy on purpose and labelled as such: these are the words a person reads, not the
/// words the process receives. `RenderPlan` is serialised and `OsString` has no portable
/// serialisation, so the preview is where a non-UTF-8 name shows replacement characters
/// — while the render still gets the real bytes from [`ffmpeg_path_arg`]. A preview that
/// cannot show one odd byte is a smaller claim to make than a render that fails on it,
/// and the two are now independent decisions.
#[must_use]
pub fn args_preview<S: AsRef<OsStr>>(args: &[S]) -> Vec<String> {
    args.iter()
        .map(|a| a.as_ref().to_string_lossy().into_owned())
        .collect()
}

/// Join an argument list for display. Works for `String` and `OsString` alike.
#[must_use]
pub fn args_display<S: AsRef<OsStr>>(args: &[S]) -> String {
    args.iter()
        .map(|a| a.as_ref().to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ")
}

/// True when both paths refer to the same file on disk.
///
/// Comparing canonicalized paths is the portable part; on Unix we also compare
/// (dev, inode) because a file can be reached through several names, and
/// overwriting the input with a partially written output is the one mistake this
/// tool must never make.
pub fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        // One side does not exist yet (the normal case for the output): fall
        // back to a lexical comparison of the absolute paths.
        _ => abs(a) == abs(b),
    }
}

fn abs(p: &Path) -> std::path::PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(p)
    }
}

/// Turn a byte count into something a status line can show.
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} {}", UNITS[0])
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// Seconds as `h:mm:ss` / `m:ss`.
pub fn human_time(s: f64) -> String {
    if !s.is_finite() || s <= 0.0 {
        return "--:--".to_string();
    }
    let t = s.round() as u64;
    if t >= 3600 {
        format!("{}:{:02}:{:02}", t / 3600, (t % 3600) / 60, t % 60)
    } else {
        format!("{}:{:02}", t / 60, t % 60)
    }
}

/// Safe filename for an output derived from an input.
/// Logical CPUs available to this process.
///
/// The ceiling the GUI's threads slider and the CLI's `--threads` both use, so
/// the two frontends cannot disagree about what the machine can run.
pub fn cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

/// Highest sensible worker count: one file per logical CPU.
///
/// Beyond that the workers only compete for the same cores, and each already
/// runs three threads plus two ffmpeg children.
pub fn max_threads() -> usize {
    cpu_count()
}

/// Clamp a requested worker count into `1..=max_threads()`.
pub fn clamp_threads(n: usize) -> usize {
    n.clamp(1, max_threads())
}

pub fn replace_extension(p: &Path, ext: &str) -> std::path::PathBuf {
    let mut name = p
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());
    if let Some(dot) = name.rfind('.') {
        // A dot at position 0 is the name itself, not an extension. Truncating there turns
        // ".profile" into "" and the output into ".flac": a hidden file with no stem, which
        // the filesystem happily accepts and nobody finds in a listing.
        if dot > 0 {
            name.truncate(dot);
        }
    }
    let mut out = p.parent().unwrap_or(Path::new("")).to_path_buf();
    out.push(format!("{name}.{ext}"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn ffmpeg_path_arg_keeps_plain_names_exactly() {
        for name in ["song.mka", "dir/song.mka", "a b.wav", "song-1.flac"] {
            assert_eq!(
                ffmpeg_path_arg(Path::new(name)),
                OsString::from(name),
                "{name} needed no prefix and must not gain one"
            );
        }
        assert_eq!(
            ffmpeg_path_arg(Path::new("/tmp/a:b.wav")),
            OsString::from("/tmp/a:b.wav"),
            "an absolute name is unambiguous however odd its last component"
        );
    }

    #[test]
    fn ffmpeg_path_arg_resolves_the_two_ambiguous_shapes() {
        // Both of these were measured failing on FFmpeg 9.0.2, input and output.
        assert_eq!(
            ffmpeg_path_arg(Path::new("a:b.wav")),
            OsString::from("./a:b.wav")
        );
        assert_eq!(
            ffmpeg_path_arg(Path::new("-dash.wav")),
            OsString::from("./-dash.wav")
        );
        // A name that already says "relative" is left alone, twice over.
        assert_eq!(
            ffmpeg_path_arg(Path::new("./a:b.wav")),
            OsString::from("./a:b.wav")
        );
        assert_eq!(
            ffmpeg_path_arg(Path::new("../a:b.wav")),
            OsString::from("../a:b.wav")
        );
    }

    #[test]
    fn ffmpeg_path_arg_never_eats_the_stdout_marker() {
        // `input_args` ends with `-` to mean "write the decoded samples to the
        // pipe". Prefixing it turns a streaming decode into a file called `-` in
        // the working directory and a pipeline that waits forever.
        assert_eq!(ffmpeg_path_arg(Path::new("-")), OsString::from("-"));
        assert_eq!(ffmpeg_path_arg(Path::new(".")), OsString::from("."));
    }

    #[cfg(unix)]
    #[test]
    fn ffmpeg_path_arg_survives_bytes_that_are_not_utf8() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};
        // 0xFF and 0xFE are not valid UTF-8 and are legal in a Unix name. This is
        // the case Path::display() destroys: it would say "-bad:\u{fffd}\u{fffd}.wav",
        // a name with no file behind it, and the render would fail with "No such
        // file or directory" while pointing at a file the user can see.
        let raw = OsStr::from_bytes(b"-bad:\xff\xfe.wav");
        let got = ffmpeg_path_arg(Path::new(raw)).into_vec();
        assert_eq!(
            got, b"./-bad:\xff\xfe.wav",
            "bytes were substituted, not preserved"
        );
    }

    #[test]
    fn same_file_detects_identity() {
        let d = std::env::temp_dir().join(format!("g6me-util-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let a = d.join("a.bin");
        std::fs::write(&a, b"x").unwrap();
        assert!(same_file(&a, &PathBuf::from(&a)));
        assert!(same_file(&a, &d.join(".").join("a.bin")));
        assert!(!same_file(&a, &d.join("b.flac")));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn human_formatters() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.0 KiB");
        assert_eq!(human_time(75.0), "1:15");
        assert_eq!(human_time(3661.0), "1:01:01");
        assert_eq!(human_time(0.0), "--:--");
    }

    #[test]
    fn extension_swap() {
        assert_eq!(
            replace_extension(Path::new("/x/y/z.mka"), "flac"),
            PathBuf::from("/x/y/z.flac")
        );
        assert_eq!(
            replace_extension(Path::new("noext"), "wav"),
            PathBuf::from("noext.wav")
        );
    }
}
