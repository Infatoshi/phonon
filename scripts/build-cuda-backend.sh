#!/bin/bash
# Build one hardware-family correction runtime: llama.cpp b12132 + Phonon's merged CUDA patch + the refiner-only
# ("lean") patch, CUDA machine code for one GPU architecture, no cuBLAS, cudart linked in. The result needs only
# the NVIDIA driver on the user's machine. Output: <out>/phonon-backend-linux-x64-cuda-sm<SM>.tar.gz, plus its
# size and sha256 for the backend manifest (crates/phonon-win/src/backends.rs).
#
# usage: scripts/build-cuda-backend.sh <SM> <out-dir> [jobs]
#   SM: 86 (RTX 30), 89 (RTX 40), 120 (RTX 50), 75 (RTX 20 / GTX 16; CUDA 13 still builds it).
#   Pascal (61) needs a CUDA 12.x toolkit and a min_cuda_driver of 12000 in its manifest entry.
# Needs git, cmake, ninja, strip and a CUDA toolkit (CUDA_HOME, default /usr/local/cuda).
# Each cicc takes about 5 GB of RAM: keep jobs low on shared machines.
# Machine code only (<SM>-real): PTX for the same arch would add about 9.5 MB gzipped and is only needed by GPUs
# newer than the build, which get their own build instead. Untested: whether ggml's arch rewrite (120 -> 120a)
# applies to "120-real"; check `cuobjdump --list-elf` on a Blackwell build.
set -euo pipefail
SM=${1:?architecture, e.g. 86}; OUT=$(mkdir -p "${2:?output dir}" && cd "$2" && pwd); JOBS=${3:-6}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
WORK=$OUT/work-sm$SM
rm -rf "$WORK"; mkdir -p "$WORK"
git clone -q --depth 1 --branch b12132 https://github.com/ggml-org/llama.cpp "$WORK/src"
git -C "$WORK/src" apply "$ROOT/kernels/cuda/merged/b12132.patch"
git -C "$WORK/src" apply "$ROOT/kernels/cuda/lean/b12132-lean.patch"
cmake -S "$WORK/src" -B "$WORK/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$SM-real" \
  -DGGML_CUDA_PHONON_LEAN=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH='$ORIGIN'
CPU_TARGETS=$(ninja -C "$WORK/build" -t targets all | grep -oE '^ggml-cpu-[a-z0-9_]+' | sort -u | tr '\n' ' ')
# shellcheck disable=SC2086
ninja -C "$WORK/build" -j"$JOBS" ggml-cuda $CPU_TARGETS llama-server

# The bundle: the server, its libraries, every CPU variant (ggml picks one at load), the CUDA module.
B=$WORK/build/bin; P=$WORK/pkg
mkdir -p "$P"
cp -a "$B/llama-server" "$P/"
for f in "$B"/lib*.so*; do cp -a "$f" "$P/"; done
find "$P" -type f \( -name '*.so*' -o -name llama-server \) -exec strip --strip-unneeded {} +
if readelf -d "$P/libggml-cuda.so" | grep -q cublas; then
  echo "libggml-cuda.so still links cuBLAS" >&2; exit 1
fi
A=$OUT/phonon-backend-linux-x64-cuda-sm$SM.tar.gz
(cd "$P" && tar czf "$A" .)
echo "sm $SM  archive $A  bytes $(stat -c %s "$A")  sha256 $(sha256sum "$A" | cut -d' ' -f1)"
echo "cuda module bytes $(stat -c %s "$P/libggml-cuda.so")"
