#!/usr/bin/env python3
"""Apply 6-bit palettization to all 6 chunked FP32 models."""

import os
import sys
import gc
import time

def dir_size_mb(path):
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(path) for f in fns
    )
    return total / (1024 * 1024)

def palettize_chunk(fp32_path, output_path, nbits=6):
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        palettize_weights,
    )
    
    print(f"\nLoading {os.path.basename(fp32_path)} ({dir_size_mb(fp32_path):.1f} MB)...")
    mlmodel = ct.models.MLModel(fp32_path)
    
    print(f"Applying {nbits}-bit palettization...")
    config = OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=nbits, mode="kmeans")
    )
    quantized = palettize_weights(mlmodel, config=config)
    
    quantized.save(output_path)
    print(f"Saved: {os.path.basename(output_path)} ({dir_size_mb(output_path):.1f} MB)")
    
    del quantized, mlmodel
    gc.collect()

def main():
    input_dir = "coreml_models_chunked"
    output_dir = "coreml_models_chunked"
    
    chunk_names = ["MuaalemChunkA", "MuaalemChunkB", "MuaalemChunkC",
                   "MuaalemChunkD", "MuaalemChunkE", "MuaalemChunkF"]
    
    nbits = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    suffix = f"_{nbits}BIT"
    
    print(f"=== {nbits}-bit Palettization of 6 Chunks ===")
    start = time.time()
    
    for name in chunk_names:
        fp32_path = os.path.join(input_dir, f"{name}_FP32.mlpackage")
        out_path = os.path.join(output_dir, f"{name}{suffix}.mlpackage")
        
        if not os.path.exists(fp32_path):
            print(f"SKIP: {fp32_path} not found")
            continue
        
        if os.path.exists(out_path):
            print(f"SKIP: {out_path} already exists ({dir_size_mb(out_path):.1f} MB)")
            continue
        
        palettize_chunk(fp32_path, out_path, nbits=nbits)
    
    elapsed = time.time() - start
    print(f"\n=== Done in {elapsed/60:.1f} min ===")
    
    # Summary
    print("\nSize comparison:")
    for name in chunk_names:
        fp32 = os.path.join(input_dir, f"{name}_FP32.mlpackage")
        int8 = os.path.join(input_dir, f"{name}_INT8.mlpackage")
        pal = os.path.join(input_dir, f"{name}{suffix}.mlpackage")
        sizes = []
        for p, label in [(fp32, "FP32"), (int8, "INT8"), (pal, f"{nbits}BIT")]:
            if os.path.exists(p):
                sizes.append(f"{label}={dir_size_mb(p):.0f}MB")
        print(f"  {name}: {', '.join(sizes)}")

if __name__ == "__main__":
    main()
