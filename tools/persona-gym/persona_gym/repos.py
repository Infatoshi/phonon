"""Curated small public repos, shallow clone, prune to source+docs."""

import shutil
import subprocess
import sys
from pathlib import Path

# (owner/repo, domain). Long-tail bias; each URL is verified at clone time
# and skipped on failure, so a dead entry only shrinks the pool.
CURATED = [
    # embedded
    ("cnlohr/ch32v003fun", "embedded"),
    ("obdev/v-usb", "embedded"),
    ("micronucleus/micronucleus", "embedded"),
    ("ataradov/mcu-starter-projects", "embedded"),
    ("esp8266/source-code-examples", "embedded"),
    ("PaulStoffregen/TimerOne", "embedded"),
    ("jeelabs/esp-link", "embedded"),
    ("peterhinch/micropython-async", "embedded"),
    ("adafruit/Adafruit_NeoPixel", "embedded"),
    ("queezythegreat/arduino-cmake", "embedded"),
    # gamedev
    ("ssloy/tinyraycaster", "gamedev"),
    ("ssloy/tinykaboom", "gamedev"),
    ("ssloy/tinyrenderer", "gamedev"),
    ("jobtalle/Koi", "gamedev"),
    ("raysan5/raylib-games", "gamedev"),
    ("grimfang4/sdl-gpu", "gamedev"),
    ("fogleman/Craft", "gamedev"),
    ("memononen/nanovg", "gamedev"),
    ("tsoding/olive.c", "gamedev"),
    ("OneLoneCoder/olcPixelGameEngine", "gamedev"),
    # bio
    ("lh3/seqtk", "bio"),
    ("lh3/miniasm", "bio"),
    ("lh3/bwa", "bio"),
    ("lh3/wgsim", "bio"),
    ("attractivechaos/klib", "bio"),
    ("torognes/vsearch", "bio"),
    ("weizhongli/cdhit", "bio"),
    ("gpertea/gffread", "bio"),
    ("ekg/fastahack", "bio"),
    ("arq5x/bedtools2", "bio"),
    # web
    ("bottlepy/bottle", "web"),
    ("cherrypy/cheroot", "web"),
    ("jcubic/jquery.terminal", "web"),
    ("kennethreitz/records", "web"),
    ("defunkt/jquery-pjax", "web"),
    ("picocss/pico", "web"),
    ("kognise/water.css", "web"),
    ("xz/new.css", "web"),
    ("pallets/itsdangerous", "web"),
    ("sinatra/mustermann", "web"),
    # ML
    ("karpathy/micrograd", "ml"),
    ("karpathy/minbpe", "ml"),
    ("karpathy/makemore", "ml"),
    ("glouw/tinn", "ml"),
    ("codeplea/genann", "ml"),
    ("attractivechaos/kann", "ml"),
    ("eriklindernoren/ML-From-Scratch", "ml"),
    ("joelgrus/data-science-from-scratch", "ml"),
    ("100/Cranium", "ml"),
    ("pjreddie/darknet", "ml"),
    # audio
    ("mackron/miniaudio", "audio"),
    ("mackron/dr_libs", "audio"),
    ("erikd/libsamplerate", "audio"),
    ("Signalsmith-Audio/signalsmith-stretch", "audio"),
    ("xiph/rnnoise", "audio"),
    ("adamstark/AudioFile", "audio"),
    ("cycfi/q", "audio"),
    ("keithito/tacotron", "audio"),
    ("tyiannak/pyAudioAnalysis", "audio"),
    ("jarikomppa/soloud", "audio"),
]

BIN_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".icns", ".webp", ".svgz",
    ".wav", ".mp3", ".ogg", ".flac", ".mp4", ".mov", ".avi",
    ".zip", ".gz", ".bz2", ".xz", ".7z", ".jar", ".tar",
    ".so", ".a", ".o", ".dylib", ".dll", ".exe", ".bin", ".dat",
    ".pdf", ".psd", ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".weights", ".pt", ".pth", ".onnx", ".npz", ".npy", ".pickle", ".pkl",
}
MAX_FILE_BYTES = 400 * 1024


def prune(dest: Path) -> None:
    """Delete .git, binaries and big files. Keep source and docs only."""
    shutil.rmtree(dest / ".git", ignore_errors=True)
    for p in sorted(dest.rglob("*"), reverse=True):
        try:
            if p.is_symlink():
                p.unlink()
            elif p.is_file():
                if p.suffix.lower() in BIN_EXTS or p.stat().st_size > MAX_FILE_BYTES:
                    p.unlink()
            elif p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        except OSError:
            pass


def clone_into_cache(slug: str, cache: Path) -> Path | None:
    """Shallow-clone one repo into cache/<owner>__<repo>, pruned. None on failure."""
    dest = cache / slug.replace("/", "__")
    if dest.exists():
        return dest
    cache.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    url = f"https://github.com/{slug}"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(tmp)],
            check=True, capture_output=True, timeout=180,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[repos] clone failed for {slug}: {e}", file=sys.stderr)
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    prune(tmp)
    tmp.replace(dest)
    return dest


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
