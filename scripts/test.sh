#!/usr/bin/env bash
#
# The suite's one entry point (see tools/README.md "Tests").
#
#   scripts/test.sh            # inner loop: every test except the `slow` tier
#   scripts/test.sh --full     # merge gate: every test, slow tier included
#   scripts/test.sh -k waqf    # extra args go to pytest
#
# Threads are pinned to one per worker: with xdist, unpinned BLAS/torch pools oversubscribe
# the CPU (measured: heavy single tests 2-10x slower). Workers default to 6 — room for the
# editor and one other run on a 10-core laptop — and `--durations=15` prints the baseline for
# the next speed complaint.

set -euo pipefail

usage() {
    cat <<'USAGE'
scripts/test.sh [--full] [pytest args...]

  (no args)  inner loop: every test except the `slow` tier
  --full     merge gate: every test, the slow tier included (first argument only)

Env: TEST_WORKERS (default 6), TEST_PYTHON (default .venv-test/bin/python, else `python`)
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -n "${TEST_PYTHON:-}" ]; then
    python="$TEST_PYTHON"
elif [ -x "$repo_root/.venv-test/bin/python" ]; then
    python="$repo_root/.venv-test/bin/python"
else
    python=python
fi

full=0
case "${1:-}" in
    --full) full=1; shift ;;
    -h|--help) usage; exit 0 ;;
esac

if ! "$python" -c "import xdist" 2>/dev/null; then
    echo "scripts/test.sh: pytest-xdist is not installed for $python" >&2
    echo "  install it (pip install pytest-xdist) or set TEST_PYTHON to the test venv" >&2
    exit 2
fi

# The slow tier is a marker filter, so `--full` simply drops it.
if [ "$full" -eq 0 ]; then
    set -- -m "not slow" "$@"
fi

cd "$repo_root/tools"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$python" -m pytest -q -p no:cacheprovider \
    -n "${TEST_WORKERS:-6}" --dist loadgroup --durations=15 "$@"
