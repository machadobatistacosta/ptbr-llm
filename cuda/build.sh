#!/bin/bash
# Build WKV CUDA kernel on Kaggle
# Usage: bash cuda/build.sh

set -e

echo "🔨 Compiling WKV CUDA kernel..."

# Detect nvcc
if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc not found! Is CUDA toolkit installed?"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../"

# Compile with optimal settings for T4 (sm_75)
nvcc -O3 -shared -Xcompiler -fPIC \
  -DTmax=256 \
  --maxrregcount 60 \
  -gencode arch=compute_75,code=sm_75 \
  -o "${OUTPUT_DIR}/libwkv_cuda.so" \
  "${SCRIPT_DIR}/wkv_cuda.cu"

echo "✅ Built libwkv_cuda.so"
echo "   Location: ${OUTPUT_DIR}/libwkv_cuda.so"

# Verify the exports
echo "📋 Exported symbols:"
nm -D "${OUTPUT_DIR}/libwkv_cuda.so" | grep "wkv_"
