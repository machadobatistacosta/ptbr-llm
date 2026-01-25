# Feasibility Report: 1.5B Model on 16GB VRAM (Dual-GPU)

## Executive Summary
Training a **1.5B parameter RWKV model** typically requires ~24GB+ VRAM (FP16 weights + gradients + optimizer states). However, with **Dual-GPU (2x8GB)** and specific optimizations (Model Sharding + Gradient Checkpointing), it is feasible but requires architectural changes.

## 1. Backend Analysis (`src/main.rs`)
**Issue**: The current implementation explicitly hardcodes `CudaDevice::new(0)` in the `get_device()` function. 
```rust
// src/main.rs:30
pub fn get_device() -> CudaDevice {
    CudaDevice::new(0) // <--- Hardcoded Index
}
```
**Resolution**: 
1. `src/main.rs` **UPDATED**: Now accepts an index `get_device(index: usize)`.
2. **Next Step**: Update `trainer.rs` to potentially manage two backends if true Model Sharding is implemented.

## 2. 1.5B Feasibility Math
| Component | Formula | Size (1.5B Params) |
|-----------|---------|-------------------|
| **Weights** | 1.5B * 2 bytes (FP16) | 3.0 GB |
| **Gradients** | 1.5B * 2 bytes (FP16) | 3.0 GB |
| **Optimizer (Adam)** | 1.5B * 8 bytes (FP32 M+V) | **12.0 GB** |
| **Activations** | Context Dependent | ~2-4 GB |
| **TOTAL** | | **~20-22 GB** |

**Conclusion**: 
- **Single 8GB GPU**: Impossible.
- **Dual 8GB GPU (16GB Total)**: Feasible **ONLY** with Model Sharding or 8-bit Optimizer.
- **Burn Support**: Burn `0.14` does not yet have native 8-bit Adam. 
- **Strategy**: We must split the model weights across the two GPUs.

## 3. Fallback Logic: Model Sharding Plan
Instead of Data Parallelism (which copies weights to both), we will use **Pipeline Parallelism / Sharding**.

### Proposed Architecture
- **GPU 0 (8GB)**: 
  - `Embedding`
  - Layers 0-11
- **GPU 1 (8GB)**:
  - Layers 12-23 (Head)
  - `Loss` Calculation

### Implementation Plan (`src/model/rwkv.rs`)
The `RWKV` struct must hold sub-modules on different devices. Burn Modules automatically move children to a device, but we can manually override this during initialization.

```rust
// Pseudo-code for Sharded RWKV
pub struct ShardedRWKV<B: Backend> {
    pub part1: RWKVPart1<B>, // Device 0
    pub part2: RWKVPart2<B>, // Device 1
}

impl ShardedRWKV {
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x_dev0 = self.part1.forward(x);
        // Explicit Transfer: Device 0 -> Device 1
        let x_dev1 = x_dev0.to_device(&self.part2.device); 
        self.part2.forward(x_dev1)
    }
}
```

## 4. Optimization: Gradient Checkpointing
To save activation memory (allowing longer context or batch size), we must implement re-computation. Burn allows `checkpointing` in the `Autodiff` backend, but it's often manual.

**Action**: Implement `burn::module::Checkpointing` or explicit re-run closures for the `RWKVBlock`.

## 5. Next Steps (Actionable)
1.  **Dual Backend Support**: Update `src/main.rs` to initialize a `SecondaryBackend` if available.
2.  **Sharding Implementation**: Refactor `src/model/rwkv.rs` to allow splitting layers.
3.  **Low-Rank Adaptation (Plan B)**: If Sharding is too complex for the timeframe, switch to **Fine-Tuning (LoRA)** where 1.5B fits easily in 16GB (since weights are frozen).

## Recommendation
For now, proceed with **400M training** (fits on single 8GB) to validate the "Balanced Soberania" dataset. Once verified, implement Sharding for the 1.5B scale-up.
