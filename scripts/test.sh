#!/usr/bin/env bash
#
# The suite's one entry point (see tools/README.md "Tests").
#
#   scripts/test.sh            # inner loop: every test except the `slow` tier
#   scripts/test.sh --full     # merge gate: every test, slow tier included
#   scripts/test.sh -k waqf    # extra args go to pytest
#
# Threads are pinned to one per worker: with xdist, unpinned BLAS/torch pools oversubscribe
# the CPU (measured: heavy single tests 2-10x slower under xdist). Workers default to 6 — room
# for the editor and one other run on a 10-core laptop — and `--durations=15` prints the
# baseline for the next speed complaint.

set -euo pipefail

usage() {
    cat <<'USAGE'
scripts/test.sh [--full] [pytest args...]

  (no args)  inner loop: every test except the `slow` tier
  --full     merge gate: every test, the slow tier included (first argument only)

Env: TEST_WORKERS (default 6), TEST_PYTHON (default: this worktree's .venv-test, else the
primary checkout's, else `python`)
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The loop provisions one test venv in the primary checkout, not one per worktree.
primary_root="$repo_root"
if common_dir="$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
    primary_root="$(dirname "$common_dir")"
fi

if [ -n "${TEST_PYTHON:-}" ]; then
    python="$TEST_PYTHON"
elif [ -x "$repo_root/.venv-test/bin/python" ]; then
    python="$repo_root/.venv-test/bin/python"
elif [ -x "$primary_root/.venv-test/bin/python" ]; then
    python="$primary_root/.venv-test/bin/python"
else
    python=python
fi

full=0
case "${1:-}" in
    --full) full=1; shift ;;
    -h|--help) usage; exit 0 ;;
esac

cd "$repo_root/tools"

if ! "$python" -c "import xdist" 2>/dev/null; then
    echo "scripts/test.sh: pytest-xdist is not installed for $python" >&2
    echo "  install it (pip install -r tools/requirements-test.txt) or set TEST_PYTHON" >&2
    exit 2
fi

# A wrong rootdir (an ini file not loaded) silently costs a tier: `-m "not slow"` would then
# filter against nothing and the server-side regeneration tests would run in the inner loop.
# `--markers` exits non-zero in pytest 9 even on success, so the text, not the status, is read.
markers="$("$python" -m pytest --markers 2>/dev/null || true)"
case "$markers" in
    *"@pytest.mark.slow:"*) ;;
    *)
        echo "scripts/test.sh: the \`slow\` tier is not declared for $PWD (tools/pytest.ini not loaded)" >&2
        exit 2
        ;;
esac

# The slow tier is a marker filter, so `--full` simply drops it.
if [ "$full" -eq 0 ]; then
    set -- -m "not slow" "$@"
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$python" -m pytest -q -p no:cacheprovider \
    -n "${TEST_WORKERS:-6}" --dist loadgroup --durations=15 "$@"
