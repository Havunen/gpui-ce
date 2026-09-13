use std::path::Path;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (filters, output, sys) in [
        (
            include_str!("../platform.txt"),
            "crates/gpui_windows/src/bindings.rs",
            false,
        ),
        (
            include_str!("../process.txt"),
            "crates/gpui_zed_util/src/windows_bindings.rs",
            true,
        ),
    ] {
        let output = root.join(output);
        let mut args = vec!["--out", output.to_str().unwrap()];
        if sys {
            args.extend(["--sys", "--no-deps"]);
        }
        args.push("--filter");
        args.extend(
            filters
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with("//")),
        );
        // The filters intentionally omit dependencies of methods we don't use.
        // Bindgen leaves their ABI slots intact and omits the callable wrappers.
        let warnings = windows_bindgen::bindgen(args);
        println!("{}: {} unused methods omitted", output.display(), warnings.len());
    }
}
