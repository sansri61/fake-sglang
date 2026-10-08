#!/usr/bin/env bash
# Any mix of prefill / decode / aggregated workers, placed in a network
# topology, from one YAML file. See launch/cluster.py and launch/clusters/.
#   ./launch/cluster.sh launch/clusters/two-zones.yaml [--dry-run] [-- worker args]
set -euo pipefail
source "$(dirname "$0")/common.sh"
PY="$PY" exec "$PY" "$ROOT/launch/cluster.py" "$@"
