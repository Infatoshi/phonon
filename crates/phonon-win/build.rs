// Windows version resource: the tray balloon title, the Task Manager name and the
// file icon all come from here. Other targets have nothing to embed.
fn main() {
    println!("cargo:rerun-if-changed=../../assets/phonon.ico");
    // The resident recogniser's config builder (src/asr.rs, csrc/sherpa_shim.c).
    println!("cargo:rerun-if-changed=csrc");
    cc::Build::new()
        .file("csrc/sherpa_shim.c")
        .include("csrc")
        .warnings(false)
        .compile("phonon_sherpa_shim");
    // C++ so a decode that throws becomes an error, not a terminated process.
    cc::Build::new()
        .cpp(true)
        .file("csrc/guard.cc")
        .warnings(false)
        .compile("phonon_sherpa_guard");
    // the target, not the build host: a cross build from Linux (cargo-xwin) uses llvm-rc
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../assets/phonon.ico")
            .set("ProductName", "Phonon")
            .set("FileDescription", "Phonon")
            .set("CompanyName", "Phonon")
            .set("LegalCopyright", "GPL-3.0");
        res.compile()
            .expect("phonon-win: compiling the Windows version resource");
    }
}
