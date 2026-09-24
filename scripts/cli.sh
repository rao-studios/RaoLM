#!/usr/bin/env bash
# WHAT: Run the raolm CLI from a checkout: build it when a source is newer than the binary
#       (RaoLM's, Frigate's, Conduit's or SinatraHarness's), put MLX's Metal library beside it
#       the first time, then hand every argument to raolm. With no arguments in a terminal,
#       raolm opens the studio (raolm ui).
# IN:   Arguments go to raolm unchanged.
#       RAOLM_CONFIG=release   build and run the release binary (default debug)
#       RAOLM_BUILD=always     run swift build even when nothing looks newer
#       RAOLM_SKIP_BUILD=1     run the binary that is already built
#       RAOLM_METALLIB=rebuild reinstall mlx.metallib (after a Frigate/MLX update)
# PIN:  Never changes directory, so relative paths in the arguments resolve against yours.
#       Build chatter goes to stderr and raolm's own output to stdout, so
#       `scripts/cli.sh generate … --format json | jq` stays clean. The metallib is only
#       installed when it is missing: build-metallib.sh also copies it into the test bundles,
#       which scripts/test.sh has to strip again before the next test build.
#
#   scripts/cli.sh                                   # the studio
#   scripts/cli.sh ui --fixtures Tests/RaoLMStudioTests/Fixtures/demo
#   scripts/cli.sh doctor
#   scripts/cli.sh demo
#   scripts/cli.sh ground --run <run> --generation <json>
#   RAOLM_CONFIG=release scripts/cli.sh demo
#
set -euo pipefail

# Follow symlinks, so the script also works when linked into ~/bin.
source_path="${BASH_SOURCE[0]}"
while [ -L "$source_path" ]; do
    target="$(readlink "$source_path")"
    case "$target" in
        /*) source_path="$target" ;;
        *) source_path="$(dirname "$source_path")/$target" ;;
    esac
done
REPO_ROOT="$(cd "$(dirname "$source_path")/.." && pwd)"

config="${RAOLM_CONFIG:-debug}"
case "$config" in
    debug | release) ;;
    *)
        echo "cli.sh: RAOLM_CONFIG must be debug or release, not '$config'" >&2
        exit 64
        ;;
esac

bin_dir="$REPO_ROOT/.build/$config"
[ -d "$bin_dir" ] || bin_dir="$(swift build --package-path "$REPO_ROOT" -c "$config" --show-bin-path)"
binary="$bin_dir/raolm"

# Sources whose change needs a rebuild: RaoLM's and its path dependencies' (see Package.swift).
watched=()
for path in "$REPO_ROOT/Sources" "$REPO_ROOT/Package.swift" \
    "$REPO_ROOT/../Frigate/Sources" "$REPO_ROOT/../Frigate/Package.swift" \
    "$REPO_ROOT/../Conduit/Sources" "$REPO_ROOT/../Conduit/Package.swift" \
    "$REPO_ROOT/../../../repositories/SinatraHarness/Sources" "$REPO_ROOT/../../../repositories/SinatraHarness/Package.swift"; do
    [ -e "$path" ] && watched+=("$path")
done

needs_build() {
    [ "${RAOLM_BUILD:-}" = "always" ] && return 0
    [ -x "$binary" ] || return 0
    [ -n "$(find "${watched[@]}" -newer "$binary" -type f -print -quit 2>/dev/null)" ]
}

if [ -z "${RAOLM_SKIP_BUILD:-}" ] && needs_build; then
    # -q prints only errors, to stderr.
    echo "cli.sh: building raolm ($config)" >&2
    if ! swift build --package-path "$REPO_ROOT" -c "$config" --product raolm -q >&2; then
        echo "cli.sh: swift build -c $config failed" >&2
        exit 1
    fi
    bin_dir="$(swift build --package-path "$REPO_ROOT" -c "$config" --show-bin-path)"
    binary="$bin_dir/raolm"
    # SwiftPM does not relink when a touched source compiles to the same code; mark the
    # binary as built against the current sources so the next run skips the build.
    touch "$binary"
fi
if [ ! -x "$binary" ]; then
    echo "cli.sh: no raolm binary in $bin_dir" >&2
    echo "  build it first, or unset RAOLM_SKIP_BUILD" >&2
    exit 1
fi

if [ ! -f "$bin_dir/mlx.metallib" ] || [ "${RAOLM_METALLIB:-}" = "rebuild" ]; then
    echo "cli.sh: installing mlx.metallib beside the $config binary" >&2
    "$REPO_ROOT/build-metallib.sh" "$config" > /dev/null
fi

exec "$binary" "$@"
