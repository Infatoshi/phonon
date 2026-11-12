#!/bin/bash
# Build the Mac CPU shortlist evaluator: llama.cpp b12132 + phonon patch + shortlist patch (shared libs, no Metal),
# then ios/shortlist_eval.cpp against it. usage: ./build_shortlist_eval.sh LLAMA_WORK_DIR (a patched b12132 tree)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$1
cd "$W"
cmake -B build-mac -DGGML_METAL=OFF -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build-mac --target llama -j 8 > /dev/null
c++ -std=c++17 -O2 -I include -I ggml/include -I vendor "$HERE/shortlist_eval.cpp" -L build-mac/bin \
  -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$W/build-mac/bin" -o build-mac/bin/shortlist_eval
echo "built $W/build-mac/bin/shortlist_eval"
