#!/usr/bin/env bash
# End-to-end solve + plot for one transverse RVE.
#
#   bash examples/run_visualization.sh
#   bash examples/run_visualization.sh examples/ud_transverse.yaml results

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
YAML="${1:-$ROOT/examples/ud_transverse.yaml}"
OUT="${2:-$ROOT/results}"

micromamba run -n b3-tex b3-micromech solve "$YAML" --out "$OUT"
micromamba run -n b3-tex b3-micromech plot "$YAML" --out "$OUT/plots"