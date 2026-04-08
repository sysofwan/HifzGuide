#!/bin/bash
# tools/run_follow_along_test.sh
#
# Build and run the follow-along self-test.
# Compiles the exact same Swift source files used in the iOS app.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
IOS_DIR="$REPO_DIR/ios/Muraja"
DB_PATH="$IOS_DIR/Resources/quran.db"
OUTPUT="$SCRIPT_DIR/.build/test_follow_along"

mkdir -p "$SCRIPT_DIR/.build"

echo "Building test harness..."
swiftc -O \
  -o "$OUTPUT" \
  "$IOS_DIR/FollowAlong/QuranFollowAlong.swift" \
  "$IOS_DIR/FollowAlong/QuranFollowAlong+WordScoring.swift" \
  "$IOS_DIR/FollowAlong/FollowAlongTypes.swift" \
  "$IOS_DIR/FollowAlong/PhonemeIndex.swift" \
  "$IOS_DIR/FollowAlong/PhonemeNormalization.swift" \
  "$IOS_DIR/FollowAlong/SmithWatermanAlignment.swift" \
  "$IOS_DIR/FollowAlong/PhonemeSifat.swift" \
  "$IOS_DIR/Data/QuranDatabase.swift" \
  "$SCRIPT_DIR/test_follow_along.swift" \
  -lsqlite3 \
  -framework Foundation \
  -swift-version 6 \
  2>&1

echo "Running test..."
echo
"$OUTPUT" "$DB_PATH" "$@"
