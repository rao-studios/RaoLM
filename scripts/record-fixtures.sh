#!/usr/bin/env bash
# WHAT: Re-record the studio's fixtures (Tests/RaoLMStudioTests/Fixtures/demo) from a fresh
#       `raolm demo`: a tiny model memorising a 24-document archive, evaluated with grounding.
#       `raolm ui --fixtures Tests/RaoLMStudioTests/Fixtures/demo` replays them with no MLX
#       and no Thread, and RaoLMStudioTests renders every screen from them.
# IN:   A Thread binary (the demo hosts one on its own storage). scripts/cli.sh builds raolm
#       and installs its Metal library when needed.
# PIN:  Checkpoints and provenance indexes are left out (they are large and the replay does
#       not need them); four evaluation generations and the demo's first cited generation are
#       kept with their grounding records, minified. The corpus is regenerated from its seed.
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

scratch="$(mktemp -d "${TMPDIR:-/tmp}/raolm-fixtures.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

scripts/cli.sh demo --documents 24 --epochs 60 --batch-size 2 --seq-len 128 --eval-every 10 --index-every 30 \
    --facts-sample 8 --data-dir "$scratch" | tail -3

run="$(ls -d "$scratch"/runs/* | head -1)"
fixtures="Tests/RaoLMStudioTests/Fixtures/demo"
rm -rf "$fixtures"
mkdir -p "$fixtures/runs"

python3 - "$run" "$fixtures" <<'PY'
import json, os, shutil, sys
run, fixtures = sys.argv[1], sys.argv[2]
run_id = os.path.basename(run)
dest = os.path.join(fixtures, "runs", run_id)
os.makedirs(os.path.join(dest, "generations"))
shutil.copytree(os.path.join(run, "ledger"), os.path.join(dest, "ledger"))
manifest = json.load(open(os.path.join(run, "run.json")))
json.dump(manifest, open(os.path.join(dest, "run.json"), "w"), indent=2, sort_keys=True)
json.dump(json.load(open(os.path.join(run, "eval.json"))), open(os.path.join(dest, "eval.json"), "w"), indent=2, sort_keys=True)
gen_dir = os.path.join(run, "generations")
keep = ["eval-001.json", "eval-003.json", "eval-005.json", "eval-006.json"]
keep += sorted(g for g in os.listdir(gen_dir) if g.startswith("gen-") and not g.endswith(".grounding.json"))[:1]
for name in keep:
    generation = json.load(open(os.path.join(gen_dir, name)))
    json.dump(generation, open(os.path.join(dest, "generations", name), "w"), separators=(",", ":"), sort_keys=True)
    sidecar = os.path.join(gen_dir, generation["generationID"] + ".grounding.json")
    if os.path.exists(sidecar):
        json.dump(json.load(open(sidecar)), open(os.path.join(dest, "generations", os.path.basename(sidecar)), "w"),
                  separators=(",", ":"), sort_keys=True)
thread = manifest.get("thread") or {"nodeID": "00000000-0000-0000-0000-000000000000", "httpPort": 8095, "grpcPort": 9095}
json.dump({
    "corpus": {"slug": manifest["corpus"]["slug"], "seed": 42, "documents": manifest["corpus"]["documentCount"], "maxChars": 600},
    "thread": {"nodeID": thread["nodeID"], "httpPort": thread["httpPort"], "grpcPort": thread["grpcPort"],
               "documents": manifest["corpus"]["documentCount"], "groups": 1, "owners": 1},
    "doctor": [
        {"name": "fixtures", "ok": True, "detail": f"recorded raolm demo run {run_id} (24 documents, tiny, 60 epochs)"},
        {"name": "raolm metallib", "ok": True, "detail": "not needed: the fixture backend replays, it does not run MLX"},
        {"name": "tokenizer", "ok": True, "detail": "vocab 49152, eos 0 (bundled SmolLM2)"},
        {"name": "thread binary", "ok": False, "detail": "fixture mode has no Thread"},
    ],
}, open(os.path.join(fixtures, "fixtures.json"), "w"), indent=2)
print(f"recorded {run_id}: {len(keep)} generations")
PY
du -sh "$fixtures"
