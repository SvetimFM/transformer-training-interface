# GPU Diagnostics

Basic GPU benchmarking for transformer workloads.

Wraps PyTorch and nvidia-smi to measure TFLOPS, memory bandwidth, and inference speed.

## Usage

### CLI

```bash
# Quick test (1-2 min)
python3 -m src.diagnostics.cli --mode quick

# Full test with model inference (2-3 min, downloads ~1GB model)
python3 -m src.diagnostics.cli --mode deep
```

### Web UI (Optional)

```bash
python3 src/diagnostics/web/server.py
# Open http://localhost:8000
```

## What It Tests

### Hardware Detection
- Reads GPU specs from `torch.cuda.get_device_properties()`
- Parses nvidia-smi for PCIe/NVLink topology
- Classifies GPU type (consumer/professional/datacenter)

### Memory Bandwidth
- Times `torch.randn()` and `.copy_()` operations
- Measures host-to-device, device-to-device transfers
- Compares to theoretical peak from specs

### Compute (TFLOPS)
- Runs matrix multiplications at different precisions (FP32/FP16/BF16)
- Measures TFLOPS and compares to spec sheet
- Typical consumer GPUs hit 50-60% of theoretical peak

### Real Inference
- Downloads small models from HuggingFace (Qwen 0.5B-3B, TinyLlama)
- Measures tokens/sec and latency
- Forces GPU to P0 power state via sustained load

### Multi-GPU (2+ GPUs only)
- Runs benchmarks in parallel across GPUs
- NCCL communication tests (All-Reduce, Broadcast, P2P)
- Measures inter-GPU bandwidth

## Output Example

```
================================================================================
🔬 HARDWARE DIAGNOSTICS
================================================================================
Backend: CUDA (1 device)
Mode: QUICK

1. HARDWARE DETECTION
   NVIDIA GeForce RTX 3070 Ti
   VRAM: 8.0 GB | Class: Consumer | Memory: GDDR6X
   PCIe: Gen4 x16

2. MEMORY PROFILING
   Bandwidth: 247.4 GB/s (55% of spec)

3. COMPUTE PROFILING
   FP16: 44.6 TFLOPS (54% of spec)
   FP32: 13.1 TFLOPS

📊 SUMMARY
✓ GPU performing normally for consumer hardware
✓ Can run ~7B models in FP16
```

## Dependencies

```bash
pip install torch transformers huggingface-hub

# Optional (web UI)
pip install fastapi uvicorn
```

## Limitations

- **Single node only** - no multi-node cluster support
- **NVIDIA only** - no AMD/ROCm support
- **Basic benchmarks** - just matrix multiply and model inference
- **No validation** - assumes hardware is working correctly
- **Toy burn-in** - not suitable for actual hardware validation

## Architecture

```
src/diagnostics/
├── cli.py                   # Main entry point
├── core/
│   ├── hardware_detector.py # Wraps torch.cuda.get_device_properties()
│   ├── memory_profiler.py   # Times tensor operations
│   ├── compute_profiler.py  # Matrix multiply benchmarks
│   ├── inference_profiler.py # Loads HF models and times inference
│   ├── nccl_profiler.py     # NCCL bandwidth tests (multi-GPU)
│   ├── topology_detector.py # Parses nvidia-smi output
│   └── model_loader.py      # Downloads models from HuggingFace
└── web/
    ├── server.py            # FastAPI wrapper around CLI
    └── static/index.html    # Basic dashboard

```

## Use Cases

**Good for:**
- "Is my GPU working correctly?"
- "What TFLOPS do I actually get?"
- "Can this GPU run a 7B model?"
- "Is NVLink faster than PCIe?"

**Not for:**
- Hardware validation before purchase
- Production deployment decisions
- Comparing different GPU models (use vendor benchmarks)
- Diagnosing hardware failures

## FAQ

**Q: Why 50-60% efficiency on consumer GPUs?**
A: Normal. Datacenter GPUs get 70-80% due to better firmware and ECC overhead.

**Q: What does "burn-in" mode do?**
A: Runs benchmarks in a loop for N hours. Monitors temperature and clocks. Not a real stress test.

**Q: Does it work on AMD?**
A: No. CUDA only.

**Q: Is this production-ready?**
A: No. It's a dev tool for quick checks.

## License

MIT - Part of Train Your Dragon project.
