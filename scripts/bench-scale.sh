#!/usr/bin/env bash
# WHAT: Phase 5's bench-scale runs (Docs/ARCHITECTURE.md, "bench-scale: the rules"): for N in 3, 6,
#       12 and 24 in turn, the dataset braid-nN, a braid built from it on rao-commons-1 (80 documents
#       fed per node, three nodes training at a time) and bench-scale on it against N = 3; then the
#       combined report. One MLX job at a time: every step waits for the one before.
# IN:   T9 (default /Volumes/T9/rao/projects/raolm), NS (default "3 6 12 24"), WORKSHOP (the commons
#       workshop holding rao-commons-1). A braid that already has world.json is not rebuilt; a
#       dataset that exists is not regenerated.
#
#   scripts/bench-scale.sh
#   NS="3 6" scripts/bench-scale.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

T9="${T9:-/Volumes/T9/rao/projects/raolm}"
NS="${NS:-3 6 12 24}"
WORKSHOP="${WORKSHOP:-$T9/commons}"
PACK=ae6f5b5d208a
export RAOLM_CONFIG=release RAOLM_SKIP_BUILD=1
cli() { scripts/cli.sh "$@"; }

first=$(echo $NS | cut -d' ' -f1)
reports=()
for n in $NS; do
    root="$T9/braid-n$n"
    if [ ! -f "$T9/datasets/braid-n$n/manifest.json" ]; then
        cli dataset generate --name "braid-n$n" --nodes "$n" --peers 2 --per-type 30 --paraphrase 6 --excerpt 5 --summary 6 --variant 2 --homonym 2
    fi
    mkdir -p "$root/braid/umbrella" "$root/braid/bench"
    [ -d "$root/braid/umbrella/$PACK" ] || cp -R "$WORKSHOP/braid/umbrella/$PACK" "$root/braid/umbrella/"
    if [ ! -f "$root/braid/world.json" ]; then
        echo "== building braid-n$n"
        cli braid sync --offline --fresh --dataset "braid-n$n" --preset base --feed 80 --at-once 3 --data-dir "$root"
    fi
    out="$root/braid/bench/scale.json"
    if [ ! -f "$out" ]; then
        echo "== bench-scale N = $n"
        if [ "$n" = "$first" ]; then
            cli braid bench-scale --data-dir "$root" --out "$out"
        else
            cli braid bench-scale --data-dir "$root" --reference "$T9/braid-n$first/braid/bench/scale.json" --out "$out"
        fi
    fi
    reports+=("$out")
done

echo "== bench-scale: every N"
cli braid bench-scale --report "${reports[@]}"
