#!/usr/bin/env bash
# tools/muraja_harness/build.sh
#
# Build the Muraja grading harness (main.swift) against a pinned Muraja checkout kept OUTSIDE this
# repo. Muraja's source is never copied into HifzGuide.
#
#   bash tools/muraja_harness/build.sh
#
# Steps:
#   1. fetch sysofwan/Muraja at $MURAJA_COMMIT into $MURAJA_HARNESS_DIR/<commit>/muraja (gh auth);
#   2. stage its quran.db and layout.db with Muraja's own tools/fetch_assets.sh --db-only (which
#      downloads the pinned HifzGuide release and verifies its checksum);
#   3. collect the FollowAlong sources with Muraja's own tools/lib/linux_sources.sh, plus
#      Data/QuranDatabase.swift, into a scratch build directory;
#   4. rename ScoringParameters.forMode to shippedForMode in that scratch copy (the only edit; see
#      main.swift) and check it happened exactly once;
#   5. compile with -DTEST_HARNESS -swift-version 6, as Muraja's tools/run_replay_log.sh does;
#   6. write harness.json (binary, databases, commit, compiler) next to the binary.
#
# The default $MURAJA_HARNESS_DIR is ~/.cache/hifzguide/muraja-harness.

set -euo pipefail

MURAJA_COMMIT=99c326f3fb7c6c53ff525568976c7e2442ccde3c
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
mkdir -p "$BUILD/src"
# shellcheck source=/dev/null
source "$CHECKOUT/tools/lib/linux_sources.sh"
collect_follow_along_sources "$CHECKOUT/ios/HifzGuide" "$BUILD/src"
prepare_extra_sources "$BUILD/src" "$CHECKOUT/ios/HifzGuide/Data/QuranDatabase.swift"

TYPES="$BUILD/src/FollowAlongTypes.swift"
SIGNATURE='static func forMode(_ mode: ScoringMode) -> ScoringParameters {'
test "$(grep -cF "$SIGNATURE" "$TYPES")" = 1
sed -i.orig "s/static func forMode(_ mode: ScoringMode)/static func shippedForMode(_ mode: ScoringMode)/" "$TYPES"
rm "$TYPES.orig"
test "$(grep -c 'func forMode(' "$TYPES")" = 0

COMPILER="$(swiftc --version 2>/dev/null | head -1)"
cat > "$BUILD/src/HarnessBuildInfo.swift" <<EOF
let murajaCommit = "$MURAJA_COMMIT"
let compilerVersion = "$COMPILER"
EOF
cp "$HERE/main.swift" "$BUILD/src/main.swift"

EXTRA_FLAGS=(-lsqlite3)
if [[ "$(uname)" == "Darwin" ]]; then
  EXTRA_FLAGS+=(-framework Foundation)
fi
swiftc -O -DTEST_HARNESS -swift-version 6 -o "$BUILD/muraja_harness" "$BUILD"/src/*.swift "${EXTRA_FLAGS[@]}"

cat > "$ROOT/harness.json" <<EOF
{
  "muraja_commit": "$MURAJA_COMMIT",
  "compiler": "$COMPILER",
  "binary": "$BUILD/muraja_harness",
  "quran_db": "$CORE_DB",
  "layout_db": "$LAYOUT_DB",
  "hifzguide_release": "$HIFZGUIDE_RELEASE_TAG",
  "quran_db_sha256": "$SHA256_QURAN_DB"
}
EOF
echo "Built $BUILD/muraja_harness"
echo "Manifest $ROOT/harness.json"
