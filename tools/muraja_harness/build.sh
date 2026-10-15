#!/usr/bin/env bash
# tools/muraja_harness/build.sh
#
# Build the Muraja grading harnesses against a pinned Muraja checkout kept OUTSIDE this repo.
# Muraja's source is never copied into HifzGuide.
#
#   bash tools/muraja_harness/build.sh [<muraja commit, full sha>]   (default: the pin below)
#
# Two binaries, each compiled with this directory's scoring.swift:
#   muraja_harness      main.swift: one item's checks, a fresh session per scoring;
#   muraja_log_replay   log_replay.swift: one of Muraja's device-session logs through Muraja's own
#                       tools/replay_log.swift, as its tools/run_log_tests.sh replays it.
#
# Steps:
#   1. fetch sysofwan/Muraja at the commit into $MURAJA_HARNESS_DIR/<commit>/muraja (gh auth);
#   2. stage its quran.db and layout.db with Muraja's own tools/fetch_assets.sh --db-only (which
#      downloads the pinned HifzGuide release and verifies its checksum);
#   3. collect the FollowAlong sources with Muraja's own tools/lib/linux_sources.sh, plus
#      Data/QuranDatabase.swift, into a scratch directory per binary; the log replay also takes
#      the analysis sources and tools/replay_log.swift with the reports it calls, the set
#      tools/run_log_tests.sh compiles;
#   4. rename ScoringParameters.forMode to shippedForMode and isSoftMismatch to
#      shippedIsSoftMismatch in each scratch copy (scoring.swift replaces both); in the log
#      replay's, drop replay_log.swift's @main (log_replay.swift is the entry point) and point the
#      word scorer's logger at log_replay.swift's GradeLineLogger, which keeps the grade lines a
#      device log holds; each edit must happen exactly once;
#   5. compile with -DTEST_HARNESS -swift-version 6, as Muraja's tools/run_replay_log.sh does;
#   6. write harness.json (binaries, databases, commit, compiler) next to the binaries.
#
# The default $MURAJA_HARNESS_DIR is ~/.cache/hifzguide/muraja-harness.

set -euo pipefail

MURAJA_COMMIT="${1:-99c326f3fb7c6c53ff525568976c7e2442ccde3c}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${MURAJA_HARNESS_DIR:-$HOME/.cache/hifzguide/muraja-harness}/$MURAJA_COMMIT"
CHECKOUT="$ROOT/muraja"
BUILD="$ROOT/build"
mkdir -p "$ROOT"

if [ ! -d "$CHECKOUT/.git" ]; then
  git init -q "$CHECKOUT"
  git -C "$CHECKOUT" remote add origin https://github.com/sysofwan/Muraja.git
fi
if [ "$(git -C "$CHECKOUT" rev-parse -q --verify HEAD 2>/dev/null)" != "$MURAJA_COMMIT" ]; then
  git -C "$CHECKOUT" -c credential.helper='!gh auth git-credential' \
    fetch -q --depth 1 origin "$MURAJA_COMMIT"
  git -C "$CHECKOUT" checkout -q --detach FETCH_HEAD
fi
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$MURAJA_COMMIT"

bash "$CHECKOUT/tools/fetch_assets.sh" --db-only >/dev/null
# shellcheck source=/dev/null
. "$CHECKOUT/tools/lib/release.env"
CORE_DB="$CHECKOUT/ios/HifzGuide/Bundled/quran.db"
LAYOUT_DB="$CHECKOUT/ios/HifzGuide/AssetPackRoot/$LAYOUT_DIR/layout.db"

rm -rf "$BUILD"
mkdir -p "$BUILD"
# shellcheck source=/dev/null
source "$CHECKOUT/tools/lib/linux_sources.sh"
COMPILER="$(swiftc --version 2>/dev/null | head -1)"
EXTRA_FLAGS=(-lsqlite3)
if [[ "$(uname)" == "Darwin" ]]; then
  EXTRA_FLAGS+=(-framework Foundation)
fi

# edit_once <file> <a line it must hold exactly once> <sed expression>
edit_once() {
  test "$(grep -cxF "$2" "$1")" = 1
  sed -i.orig "$3" "$1"
  rm "$1.orig"
}

# build_binary <binary> <entry point in this directory> [<more Muraja sources>...]
build_binary() {
  local binary="$1" entry="$2" src="$BUILD/$1-src"
  shift 2
  mkdir -p "$src"
  collect_follow_along_sources "$CHECKOUT/ios/HifzGuide" "$src"
  prepare_extra_sources "$src" "$CHECKOUT/ios/HifzGuide/Data/QuranDatabase.swift" "$@"
  edit_once "$src/FollowAlongTypes.swift" \
    '  static func forMode(_ mode: ScoringMode) -> ScoringParameters {' \
    's/static func forMode(_ mode: ScoringMode)/static func shippedForMode(_ mode: ScoringMode)/'
  test "$(grep -c 'func forMode(' "$src/FollowAlongTypes.swift")" = 0
  edit_once "$src/PhonemeSifat.swift" \
    'func isSoftMismatch(_ a: Unicode.Scalar, _ b: Unicode.Scalar, softPairsEnabled: Bool) -> Bool {' \
    's/^func isSoftMismatch(/func shippedIsSoftMismatch(/'
  test "$(cat "$src"/*.swift | grep -c 'func isSoftMismatch(')" = 0
  if [ -f "$src/replay_log.swift" ]; then
    edit_once "$src/replay_log.swift" '@main' 's/^@main$//'
    edit_once "$src/QuranFollowAlong+WordScoring.swift" 'private let scoringLogger = Logger(' \
      's/^private let scoringLogger = Logger(/private let scoringLogger = GradeLineLogger(/'
  fi
  printf 'let murajaCommit = "%s"\nlet compilerVersion = "%s"\n' "$MURAJA_COMMIT" "$COMPILER" \
    > "$src/HarnessBuildInfo.swift"
  cp "$HERE/scoring.swift" "$src/scoring.swift"
  cp "$HERE/$entry" "$src/$entry"
  swiftc -O -DTEST_HARNESS -swift-version 6 -o "$BUILD/$binary" "$src"/*.swift "${EXTRA_FLAGS[@]}"
}

build_binary muraja_harness main.swift
ANALYSIS="$BUILD/analysis-src"
mkdir -p "$ANALYSIS"
collect_analysis_sources "$CHECKOUT/ios" "$ANALYSIS"
build_binary muraja_log_replay log_replay.swift "$ANALYSIS"/*.swift \
  "$CHECKOUT/tools/replay_log.swift" "$CHECKOUT/tools/oracle_report.swift" \
  "$CHECKOUT/tools/frontier_report.swift" "$CHECKOUT/tools/reread_scoreboard.swift"

cat > "$ROOT/harness.json" <<EOF
{
  "muraja_commit": "$MURAJA_COMMIT",
  "compiler": "$COMPILER",
  "binary": "$BUILD/muraja_harness",
  "log_replay_binary": "$BUILD/muraja_log_replay",
  "quran_db": "$CORE_DB",
  "layout_db": "$LAYOUT_DB",
  "hifzguide_release": "$HIFZGUIDE_RELEASE_TAG",
  "quran_db_sha256": "$SHA256_QURAN_DB"
}
EOF
echo "Built $BUILD/muraja_harness and $BUILD/muraja_log_replay"
echo "Manifest $ROOT/harness.json"
