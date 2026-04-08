#!/bin/bash
# tools/run_replay_log.sh
#
# Build and run the log replay harness.
# Replays Console.app follow-along logs through the alignment engine.
#
# Usage:
#   bash tools/run_replay_log.sh <logfile.txt>
#   bash tools/run_replay_log.sh <logfile.txt> --verbose

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
IOS_DIR="$REPO_DIR/ios/Muraja"
DB_PATH="$IOS_DIR/Resources/quran.db"
OUTPUT="$SCRIPT_DIR/.build/replay_log"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <logfile.txt> [--verbose]"
  echo "  Replays Console.app follow-along logs through the alignment engine."
  exit 1
fi

mkdir -p "$SCRIPT_DIR/.build"

echo "Building replay harness..."
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
  "$SCRIPT_DIR/replay_log.swift" \
  -lsqlite3 \
  -framework Foundation \
  -swift-version 6 \
  2>&1

echo "Replaying log..."
echo
"$OUTPUT" "$DB_PATH" "$@"
