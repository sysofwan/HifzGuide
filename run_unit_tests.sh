#!/bin/bash
# tools/run_unit_tests.sh
#
# Build and run unit tests for the Muraja Swift code.
# Works on both macOS and Linux.
#
# macOS: uses xcodebuild test (Xcode's native test runner)
# Linux: builds a standalone test binary with swiftc + XCTMain
#
# Usage:
#   cd tools && bash run_unit_tests.sh
#   # Or from repo root:
#   bash tools/run_unit_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
IOS_DIR="$REPO_DIR/ios/Muraja"
TESTS_DIR="$REPO_DIR/ios/MurajaTests"
DB_PATH="$IOS_DIR/Resources/quran.db"

# --- macOS: use xcodebuild test ---
if [[ "$(uname)" == "Darwin" ]]; then
  echo "Running unit tests via xcodebuild..."
  xcodebuild test \
    -scheme Muraja \
    -project "$REPO_DIR/ios/Muraja.xcodeproj" \
    -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
    -only-testing:MurajaTests \
    -quiet \
    2>&1
  echo
  echo "All tests passed."
  exit 0
fi

# --- Linux: build standalone test binary with swiftc ---
OUTPUT="$SCRIPT_DIR/.build/run_unit_tests"
mkdir -p "$SCRIPT_DIR/.build"

# Prepare temporary sources
TEMP_DIR="$SCRIPT_DIR/.build/temp_sources"
mkdir -p "$TEMP_DIR"

# On Linux, OSLog is not available — strip the import and provide a stub
echo "Preparing for Linux build..."
# Strip OSLog imports from files that use it
sed 's/^import OSLog$//' "$IOS_DIR/FollowAlong/QuranFollowAlong.swift" > "$TEMP_DIR/QuranFollowAlong.swift"
sed 's/^import OSLog$//' "$IOS_DIR/FollowAlong/QuranFollowAlong+WordScoring.swift" > "$TEMP_DIR/QuranFollowAlong+WordScoring.swift"
sed 's/^import OSLog$//' "$IOS_DIR/FollowAlong/PhonemeIndex.swift" > "$TEMP_DIR/PhonemeIndex.swift"

# Create OSLog stub
cat > "$TEMP_DIR/OSLogShim.swift" << 'OSLOG_EOF'
import Foundation
struct Logger: Sendable {
  init(subsystem: String, category: String) {}
  func info(_ message: @autoclosure () -> String) {}
  func debug(_ message: @autoclosure () -> String) {}
  func error(_ message: @autoclosure () -> String) {}
  func warning(_ message: @autoclosure () -> String) {}
}
OSLOG_EOF

FOLLOW_ALONG_SRC="$TEMP_DIR/QuranFollowAlong.swift"
WORD_SCORING_SRC="$TEMP_DIR/QuranFollowAlong+WordScoring.swift"
PHONEME_INDEX_SRC="$TEMP_DIR/PhonemeIndex.swift"
EXTRA_SRCS="$TEMP_DIR/OSLogShim.swift"

# Set up SQLite3 module map for Linux
SQLITE_MODULE="/tmp/sqlite3_module"
mkdir -p "$SQLITE_MODULE"
cat > "$SQLITE_MODULE/module.modulemap" << 'MODULEMAP_EOF'
module SQLite3 [system] {
  header "/usr/include/sqlite3.h"
  link "sqlite3"
  export *
}
MODULEMAP_EOF
SQLITE_FLAGS="-I$SQLITE_MODULE -lsqlite3"

# Create the test runner entry point
cat > "$TEMP_DIR/TestMain.swift" << 'MAIN_EOF'
import Foundation
import XCTest

@main
struct TestMain {
  static func main() {
    // Set up database tests before running
    QuranDatabaseTests.setUp()
    PhonemeIndexTests.setUp()
    FollowAlongTests.setUp()

    XCTMain([
      testCase(PhonemeSifatTests.allTests),
      testCase(PhonemeVocabularyTests.allTests),
      testCase(NormalizationTests.allTests),
      testCase(SmithWatermanTests.allTests),
      testCase(QuranPositionTests.allTests),
      testCase(QuranDatabaseTests.allTests),
      testCase(PhonemeIndexTests.allTests),
      testCase(FollowAlongTests.allTests),
    ])
  }
}
MAIN_EOF

echo "Building unit tests..."
swiftc -O \
  -o "$OUTPUT" \
  "$FOLLOW_ALONG_SRC" \
  "$WORD_SCORING_SRC" \
  "$PHONEME_INDEX_SRC" \
  "$IOS_DIR/FollowAlong/FollowAlongTypes.swift" \
  "$IOS_DIR/FollowAlong/PhonemeNormalization.swift" \
  "$IOS_DIR/FollowAlong/SmithWatermanAlignment.swift" \
  "$IOS_DIR/FollowAlong/PhonemeSifat.swift" \
  "$IOS_DIR/Data/QuranDatabase.swift" \
  "$IOS_DIR/Models/PhonemeVocabulary.swift" \
  ${EXTRA_SRCS:+"$EXTRA_SRCS"} \
  "$TESTS_DIR/PhonemeSifatTests.swift" \
  "$TESTS_DIR/PhonemeVocabularyTests.swift" \
  "$TESTS_DIR/NormalizationTests.swift" \
  "$TESTS_DIR/SmithWatermanTests.swift" \
  "$TESTS_DIR/QuranPositionTests.swift" \
  "$TESTS_DIR/QuranDatabaseTests.swift" \
  "$TESTS_DIR/PhonemeIndexTests.swift" \
  "$TESTS_DIR/FollowAlongTests.swift" \
  "$TEMP_DIR/TestMain.swift" \
  $SQLITE_FLAGS \
  -swift-version 6 \
  2>&1

echo
echo "Running unit tests..."
echo
export QURAN_DB_PATH="$DB_PATH"
"$OUTPUT"
