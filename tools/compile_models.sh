#!/bin/bash
#
# Compile .mlpackage models to .mlmodelc for on-device use.
# Requires Xcode command line tools (xcrun coremlcompiler).
#
# Usage:
#   bash compile_models.sh [input_dir] [output_dir]
#   # Default input:  coreml_models_chunked/
#   # Default output: compiled_models/
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_DIR="${1:-$SCRIPT_DIR/coreml_models_chunked}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/compiled_models}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "❌ Input directory not found: $INPUT_DIR"
  echo "Usage: bash compile_models.sh [input_dir] [output_dir]"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Compiling .mlpackage models from: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

count=0
for pkg in "$INPUT_DIR"/*_6BIT.mlpackage; do
  [ -d "$pkg" ] || continue
  name="$(basename "$pkg" .mlpackage)"
  echo "Compiling $name..."
  xcrun coremlcompiler compile "$pkg" "$OUTPUT_DIR"
  echo "  ✅ $name.mlmodelc"
  count=$((count + 1))
done

if [ "$count" -eq 0 ]; then
  echo "❌ No *_6BIT.mlpackage files found in $INPUT_DIR"
  exit 1
fi

echo ""
echo "✅ Compiled $count models to $OUTPUT_DIR"

# Create zip for release
ZIP_PATH="$SCRIPT_DIR/models.zip"
echo "Creating $ZIP_PATH..."
cd "$OUTPUT_DIR"
zip -r -q "$ZIP_PATH" *.mlmodelc
echo "✅ models.zip created ($(du -h "$ZIP_PATH" | cut -f1))"
