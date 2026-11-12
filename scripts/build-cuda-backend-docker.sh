#!/bin/bash
# Run scripts/build-cuda-backend.sh inside Ubuntu 22.04 with the CUDA 13.3 toolkit, so the bundle loads on every
# Linux the app supports (glibc 2.35 and up). Needs docker; no GPU (it only compiles).
#
# usage: scripts/build-cuda-backend-docker.sh <out-dir> <jobs> <SM>...
#   e.g. scripts/build-cuda-backend-docker.sh /data/backends 6 75 86 89 120
# Builds the families one after another (each cicc takes about 5 GB of RAM). DOCKER_ARGS adds docker run options,
# e.g. DOCKER_ARGS="--memory 16g --cpus 6" on a shared machine.
set -euo pipefail
OUT=$(mkdir -p "${1:?output dir}" && cd "$1" && pwd); JOBS=${2:?jobs}; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-nvidia/cuda:13.3.1-devel-ubuntu22.04}
for SM in "$@"; do
  # shellcheck disable=SC2086
  docker run --rm --user 0 ${DOCKER_ARGS:-} -v "$ROOT:/repo:ro" -v "$OUT:/out" -e SM="$SM" -e JOBS="$JOBS" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
      set -euo pipefail
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq >/dev/null && apt-get install -y -qq git ninja-build python3-pip xz-utils binutils >/dev/null
      pip3 install -q cmake
      CUDA_HOME=/usr/local/cuda bash /repo/scripts/build-cuda-backend.sh "$SM" /out "$JOBS"
      rm -rf /out/work-sm$SM/src /out/work-sm$SM/build
      chown -R "$HOST_UID:$HOST_GID" /out'
done
