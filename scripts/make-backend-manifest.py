"""Backend manifest (schema 1, crates/phonon-win/src/backends.rs) for the CUDA bundles in a directory.

usage: python3 -I scripts/make-backend-manifest.py <archive-dir> <url-prefix> > manifest.json
  <archive-dir>: phonon-backend-linux-x64-cuda-sm<SM>.tar.xz (or .tar.bz2) files (scripts/build-cuda-backend.sh).
  <url-prefix>: where they will be served, ending in "/" (an empty string keeps bare file names).
Then sign it: phonon-win sign-backends --key <seed file> manifest.json (writes manifest.json.sig, which the
app fetches from <manifest URL>.sig). sm_120 builds are sm_120a machine code (ggml rewrites the arch), so their
entry is "exact" and named ...-sm120a.
"""

import hashlib
import json
import os
import re
import sys

# CUDA 13.0 (driver 580): the floor of the CUDA 13 toolkit's minor-version compatibility
MIN_CUDA_DRIVER = 13000


def entry(path, sm):
    major, minor = divmod(sm, 10)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    item = {
        "id": f"linux-x64-cuda-sm{sm}" + ("a" if sm >= 120 else ""),
        "os": "linux",
        "arch": "x86_64",
        "kind": "cuda",
        "sm": [major, minor],
        "min_cuda_driver": MIN_CUDA_DRIVER,
        "url": "",
        "bytes": os.path.getsize(path),
        "sha256": digest.hexdigest(),
        "server": "llama-server",
    }
    if sm >= 120:
        item["exact"] = True
    return item


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    backends = []
    for name in sorted(os.listdir(src)):
        match = re.fullmatch(r"phonon-backend-linux-x64-cuda-sm(\d+)\.tar\.(?:xz|bz2)", name)
        if match:
            item = entry(os.path.join(src, name), int(match.group(1)))
            item["url"] = prefix + name
            backends.append(item)
    json.dump({"schema": 1, "backends": backends}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
