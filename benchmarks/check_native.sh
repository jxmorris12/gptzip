#!/usr/bin/env bash
# Requires GCC/Clang sanitizer runtimes and an environment with the test deps.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
gptzip_python="${GPTZIP_PYTHON:-python3}"
restore_build() {
    env -u CFLAGS -u LDFLAGS "$gptzip_python" setup.py build_ext --inplace --force
}
trap restore_build EXIT
CFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
LDFLAGS='-fsanitize=address,undefined' \
    "$gptzip_python" setup.py build_ext --inplace --force
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 \
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
LD_PRELOAD="$(cc -print-file-name=libasan.so)" \
    "$gptzip_python" -m pytest -c pyproject.toml \
    tests/test_probability.py tests/test_range_parity.py \
    tests/test_fused_pdf.py tests/test_native_state.py -q
