#!/usr/bin/env bash
# WHAT: Build RaoLM's tests, put MLX's Metal library where the test bundles can load it, and
#       run them. Extra arguments go to `swift test` (e.g. --filter RaoLMCoreTests).
# IN:   RAOLM_MLX_TESTS (default 1) enables the MLX suites; RAOLM_THREAD_TESTS=1 also runs the
#       suite that launches a Thread from ../Thread/.build/release/thread.
# PIN:  build-metallib.sh copies mlx.metallib into every .xctest bundle so MLX can find it, and
#       those copies break the bundle's ad-hoc re-sign on the next build ("code object is not
#       signed at all"). So the copies are stripped first, the build runs, they are reinstalled,
#       and the tests run with --skip-build.
#
#   scripts/test.sh
#   scripts/test.sh --filter RaoLMCoreTests
#   RAOLM_THREAD_TESTS=1 scripts/test.sh --filter RaoLMThreadTests
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -d .build/debug ]; then
    BUILD_DIR="$(cd .build/debug && pwd -P)"
    while IFS= read -r -d '' bundle; do
        rm -f "$bundle/Contents/MacOS/mlx.metallib"
        rm -rf "$bundle/Contents/MacOS/Resources"
    done < <(find "$BUILD_DIR" -maxdepth 1 -name '*.xctest' -print0 2>/dev/null)
fi

swift build --build-tests
./build-metallib.sh debug > /dev/null
RAOLM_MLX_TESTS="${RAOLM_MLX_TESTS:-1}" swift test --skip-build "$@"
