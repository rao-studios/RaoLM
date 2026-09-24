#!/usr/bin/env bash
# WHAT: Build MLX's mlx.metallib into this package's .build.
# IN:   [debug|release] (default debug). FRIGATE_DIR overrides where Frigate lives.
# OUT:  mlx.metallib beside every RaoLM binary and inside every .xctest bundle.
# PIN:  A delegate, not an implementation. Frigate owns the vendored .metal sources, so it
#       owns the compile; `swift build` has no Metal step, and without this script the first
#       GPU op dies with "Failed to load the default metallib". Run it after `swift build`
#       (or `swift build --build-tests`), because it installs into the existing .build tree.
#
#   ./build-metallib.sh [debug|release]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRIGATE_DIR="${FRIGATE_DIR:-$REPO_ROOT/../Frigate}"
CANONICAL="$FRIGATE_DIR/scripts/build-metallib.sh"

if [ ! -x "$CANONICAL" ]; then
    echo "build-metallib: cannot find $CANONICAL" >&2
    echo "  Set FRIGATE_DIR to your Frigate checkout." >&2
    exit 1
fi

exec "$CANONICAL" "${1:-debug}" --package "$REPO_ROOT"
