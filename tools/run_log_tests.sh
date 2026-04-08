#!/usr/bin/env bash
#
# Run log-based regression tests for the follow-along engine.
#
# Usage:
#   bash tools/run_log_tests.sh                  # Run tests against baselines
#   bash tools/run_log_tests.sh --update-baseline  # Generate/update baselines
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DB_PATH="$REPO_DIR/ios/Muraja/Resources/quran.db"
BUILD_DIR="/tmp/replay_log_build"

# Source files needed for compilation
FOLLOW_ALONG_DIR="$REPO_DIR/ios/Muraja/FollowAlong"
SOURCE_FILES=(
  "$FOLLOW_ALONG_DIR/FollowAlongTypes.swift"
  "$FOLLOW_ALONG_DIR/PhonemeIndex.swift"
  "$FOLLOW_ALONG_DIR/PhonemeNormalization.swift"
  "$FOLLOW_ALONG_DIR/SmithWatermanAlignment.swift"
  "$FOLLOW_ALONG_DIR/PhonemeSifat.swift"
  "$FOLLOW_ALONG_DIR/QuranFollowAlong.swift"
  "$FOLLOW_ALONG_DIR/QuranFollowAlong+WordScoring.swift"
  "$REPO_DIR/ios/Muraja/Data/QuranDatabase.swift"
  "$SCRIPT_DIR/replay_log.swift"
)

# Create build dir
mkdir -p "$BUILD_DIR"

# Create SQLite module map for Linux (macOS has it built in)
if [[ "$(uname)" != "Darwin" ]]; then
  SQLITE_MODULE="/tmp/sqlite3_module"
  mkdir -p "$SQLITE_MODULE"
  cat > "$SQLITE_MODULE/module.modulemap" <<EOF
module CSQLite [system] {
  header "/usr/include/sqlite3.h"
  link "sqlite3"
  export *
}
EOF
  EXTRA_FLAGS="-I$SQLITE_MODULE -lsqlite3"
else
  EXTRA_FLAGS="-lsqlite3"
fi

# Strip OSLog imports for non-Darwin or Linux builds
TEMP_DIR="/tmp/replay_log_src"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

for f in "${SOURCE_FILES[@]}"; do
  base=$(basename "$f")
  if [[ "$(uname)" != "Darwin" ]]; then
    sed 's/^import OSLog$//' "$f" > "$TEMP_DIR/$base"
  else
    cp "$f" "$TEMP_DIR/$base"
  fi
done

echo "Building replay_log..."
# shellcheck disable=SC2086
swiftc -O \
  -module-name ReplayLog \
  $EXTRA_FLAGS \
  "$TEMP_DIR"/*.swift \
  -o "$BUILD_DIR/replay_log" 2>&1

if [ $? -ne 0 ]; then
  echo "❌ Build failed"
  exit 1
fi
echo "Build succeeded."
echo

# Run with appropriate mode
if [[ "${1:-}" == "--update-baseline" ]]; then
  "$BUILD_DIR/replay_log" "$DB_PATH" --update-baseline
else
  "$BUILD_DIR/replay_log" "$DB_PATH" --test
fi
