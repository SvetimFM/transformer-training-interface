# Hardware Diagnostics & Performance Testing Suite

**Comprehensive GPU/CPU benchmarking for transformer training and LLM inference.**

For hobbyists: Understand your hardware's true capabilities - no more guessing!
For professionals: Validate cluster performance before production deployment.

## Quick Start

### CLI (Headless, Automation, HPC)

```bash
# Quick diagnostics (1-2 minutes)
python3 -m src.diagnostics.cli --mode quick

# Deep analysis with real models (~3 minutes)
python3 -m src.diagnostics.cli --mode deep

# Long-running stability test
python3 -m src.diagnostics.cli --mode burn-in --duration 24h
```

### Web UI (Interactive, Visual)

```bash
# Start web server
python3 src/diagnostics/web/server.py

# Open in browser
http://localhost:8000
```

**Web UI Features:**
- Real-time topology visualization
- Live benchmark progress
- Interactive HMI-style dashboard
- One-click report downloads

## What It Tests

### 1. Hardware Detection 🔍
- GPU count, names, and memory (VRAM)
- CUDA/ROCm, cuDNN, NCCL versions
- GPU topology (PCIe Gen, lanes, NVLink)
- GPU classification (consumer/datacenter/professional)
- System info (CPU, cores, platform)

**Output:** Detailed hardware inventory with expected performance baselines.

### 2. Memory Profiling 💾
- **Capacity**: Usable VRAM (accounting for overhead)
- **Bandwidth**: Data transfer speed (GB/s)
  - Device-to-device (most important for inference)
  - Host-to-device (affects model loading)
  - Device-to-host (affects result copying)
- **Allocation patterns**: Simulates transformer model loading

**Why it matters:** Memory bandwidth directly affects token generation speed.

### 3. Compute Profiling ⚡
- **TFLOPS** (Trillion Floating Point Operations Per Second)
- Multiple precisions:
  - FP32 (baseline)
  - FP16 (most common for LLM inference)
  - BF16 (better numerical stability, Ampere+)
  - TF32 (automatic mixed precision on Ampere+)
- **Efficiency analysis**: % of theoretical peak
- **Health monitoring**: Clocks, temperature, power, throttling

**Why it matters:** Higher TFLOPS = faster inference. Efficiency tells you if GPU is performing optimally.

### 4. Real Inference Profiling 🚀 (NEW!)
- **Actual transformer models** (DistilBERT, TinyLlama, LLaVA)
- **Tokens per second** throughput
- **Time to First Token** (TTFT) latency
- **Batch scaling** analysis
- **Memory usage** under real workloads
- **Sustained load** testing (forces GPU to P0 state)

**Why it matters:** Synthetic benchmarks don't tell the whole story. Real models have attention, KV caching, and memory patterns that affect performance.

### 5. Multi-GPU Support 🎮 (NEW!)
- **Parallel benchmarking** across all GPUs
- **NVLink topology** detection and visualization
- **Aggregate metrics** (total TFLOPS, average efficiency)
- **Per-GPU health status**

**Why it matters:** Critical for DGX systems, HPC clusters, and multi-GPU workstations.

### 6. Burn-in Testing 🔥
- **Extended stability** validation (hours to days)
- **Thermal monitoring** (temperature, power draw)
- **Throttling detection** (P-state tracking)
- **Error detection** (memory errors, crashes)

**Why it matters:** Ensure hardware reliability before production deployment.

## Example Output

```
================================================================================
🔬 HARDWARE DIAGNOSTICS & PERFORMANCE TESTING
================================================================================
Backend: CUDA (1 device(s))
Mode: DEEP
================================================================================

1. HARDWARE DETECTION ✅
   NVIDIA GeForce RTX 3070 Ti
   VRAM: 8.0 GB | Class: Consumer | Memory: GDDR6X
   Tensor Cores: Gen3 (Ampere) | ECC: Not supported
   PCIe: Gen4 x16

2. MEMORY PROFILING ✅
   Usable VRAM: 7.60 GB (95.0%)
   Bandwidth: 247.4 GB/s (device-to-device)
   Efficiency: 55.2% of theoretical peak

3. COMPUTE PROFILING ✅
   FP16 Performance: 44.63 TFLOPS @ 1935 MHz
   Efficiency: 54.4% of theoretical peak (82 TFLOPS)
   FP32 Performance: 13.08 TFLOPS
   BF16 Performance: 44.52 TFLOPS
   TF32 Performance: 20.03 TFLOPS

   Status: ⚠️ GPU in P2 power state (not P0)
   Temperature: 64°C | Power: 212W / 290W

4. REAL INFERENCE PROFILING ✅
   Model: distilbert-base-uncased (256 MB)
   Throughput: 1,234 tokens/sec (batch=1)
   Time to First Token: 12.5 ms
   Decode Latency: 0.81 ms/token
   Memory Used: 1.2 GB / 8.0 GB

================================================================================
📊 RECOMMENDATIONS
================================================================================
✓ GPU operates at 54% efficiency - excellent for consumer hardware
✓ Can run models up to ~7B parameters (FP16)
✓ Can run ~13B models with 4-bit quantization
⚠️ Consider locking GPU clocks for peak performance:
   sudo nvidia-smi -lgc 2100
================================================================================
```

## Test Modes

| Mode | Duration | Tests | Use Case |
|------|----------|-------|----------|
| **Quick** | 1-2 min | Hardware + Compute (FP16/FP32) | Sanity check, CI/CD |
| **Deep** | 2-3 min | All tests + Real inference | Full characterization |
| **Burn-in** | Hours+ | Sustained load + monitoring | Stability validation |

### Quick Mode
- Hardware detection
- Basic compute benchmark (FP16/FP32)
- Fast sanity check

**When to use:** Quick validation, automated testing.

### Deep Mode
- Full hardware topology
- All precision benchmarks (FP32/FP16/BF16/TF32)
- **Real transformer inference** (loads DistilBERT or TinyLlama)
- Health monitoring integration
- Multi-GPU support

**When to use:** Setting up new hardware, troubleshooting performance.

### Burn-in Mode
- Continuous sustained load
- Temperature/power tracking
- Throttling detection
- Crash recovery

**When to use:** Validating hardware before production, stress testing.

## CLI vs Web UI

**Both interfaces share the same diagnostic core - no code duplication!**

See [ARCHITECTURE.md](./ARCHITECTURE.md) for design rationale.

### CLI: For Automation & HPC
```bash
# Scripting
python -m diagnostics.cli --mode deep --output results.json

# HPC job submission
sbatch --gres=gpu:8 diagnostics_job.sh

# CI/CD pipelines
pytest && python -m diagnostics.cli --mode quick || exit 1

# Docker containers
docker run --gpus all my-image python -m diagnostics.cli
```

### Web UI: For Interactive Exploration
- Visual topology diagrams
- Live progress updates
- Downloadable reports
- No command-line knowledge needed
- Great for demos and training

**Model Cache Location:** `~/.cache/diagnostics/models/`
Models are downloaded once and shared between CLI and Web interfaces.

## Multi-GPU & HPC Deployment

### Single Node (Multi-GPU)
```bash
# CLI: Test all GPUs in parallel
python -m diagnostics.cli --mode deep

# Web UI with port forwarding
ssh -L 8000:localhost:8000 user@gpu-node
python src/diagnostics/web/server.py
# Then open: http://localhost:8000
```

### DGX A100 / HPC Clusters
The diagnostics suite automatically detects:
- NVLink topology (NV1-NV12 connections)
- PCIe hierarchy
- NUMA affinity
- GPU interconnect types

**Example output for 8x A100:**
```
GPU Grid:
├─ GPU0: A100 (40GB HBM2e) - NV12 to: GPU1, GPU2, GPU3
├─ GPU1: A100 (40GB HBM2e) - NV12 to: GPU0, GPU2, GPU3
├─ ...
Aggregate: 2400 TFLOPS FP16 (8x 300 TFLOPS)
```

## Architecture

```
src/diagnostics/
├── ARCHITECTURE.md          # Design rationale
├── README.md                # This file
├── cli.py                   # CLI orchestration ✅
├── __init__.py
├── core/                    # Shared diagnostic modules
│   ├── backend_detector.py  # CUDA/ROCm/CPU detection
│   ├── burn_in.py           # Stability testing ✅
│   ├── compute_profiler.py  # TFLOPS benchmarks ✅
│   ├── cpu_profiler.py      # CPU fallback ✅
│   ├── hardware_detector.py # GPU specs ✅
│   ├── health_monitor.py    # Temperature/power/throttling ✅
│   ├── inference_profiler.py # Real transformers ✅
│   ├── memory_profiler.py   # Bandwidth/capacity ✅
│   ├── metrics.py           # Test framework ✅
│   ├── model_loader.py      # HuggingFace model management ✅
│   ├── multi_gpu_profiler.py # Parallel GPU testing ✅
│   └── topology_detector.py # PCIe/NVLink mapping ✅
├── tests/                   # 112 passing tests ✅
│   ├── test_*.py
└── web/                     # Web interface
    ├── server.py            # FastAPI backend ✅
    ├── static/
    │   └── index.html       # HMI-style dashboard ✅
    └── README.md            # Web UI docs
```

## Programmatic Usage

```python
from diagnostics.core.hardware_detector import HardwareDetector
from diagnostics.core.compute_profiler import ComputeProfiler
from diagnostics.core.inference_profiler import InferenceProfiler
from diagnostics.core.multi_gpu_profiler import MultiGPUProfiler
from diagnostics.core.metrics import TestMode
from diagnostics.cli import DiagnosticsCLI

# Option 1: Use individual profilers
profiler = ComputeProfiler(device="cuda:0")
result = profiler.run(mode=TestMode.DEEP)
print(f"FP16: {result.metrics['fp16_tflops']} TFLOPS")

# Option 2: Use full CLI orchestration
cli = DiagnosticsCLI()
cli.run_all_tests(mode=TestMode.DEEP)
report = cli.suite.to_dict()  # JSON-serializable
```

## Understanding the Metrics

### GPU Classification
| Class | Examples | Expected FP16 Efficiency | Memory |
|-------|----------|-------------------------|--------|
| **Consumer** | RTX 3070 Ti, 4090 | 50-60% | GDDR6/X |
| **Professional** | RTX A6000, V100 | 65-75% | GDDR6, HBM2 |
| **Datacenter** | A100, H100 | 75-85% | HBM2e, HBM3 |

**Why efficiency varies:** Datacenter GPUs have optimized firmware, ECC memory, better scheduling.

### Memory Bandwidth
- **Excellent**: >300 GB/s (modern high-end GPUs)
- **Good**: 150-300 GB/s (mid-range)
- **Limited**: <150 GB/s (older/budget GPUs)

**Note:** 50-70% efficiency is normal for memory copy operations.

### Compute (TFLOPS)
- **FP16 vs FP32 ratio**: Should be 2-8x on Tensor Core GPUs
- **Efficiency**: 40-70% of theoretical peak is normal
- **If <40%**: Check for throttling, driver issues, or P2 power state

### Real Inference
- **Tokens/sec**: Varies by model size and batch size
- **TTFT (Time to First Token)**: Lower is better (<20ms ideal)
- **Memory overhead**: Expect 2x model size for activations + KV cache

## Dependencies

```bash
# Core (required)
pip install torch transformers huggingface_hub

# Web UI (optional)
pip install fastapi uvicorn

# Testing
pip install pytest
```

## FAQ

**Q: Why is my bandwidth only 50% of theoretical?**
A: Normal for copy operations. Raw memory access (not measured here) is faster.

**Q: Why does DEEP mode download models?**
A: Real inference testing requires actual transformers (DistilBERT, TinyLlama). Models are cached locally.

**Q: Can this damage my GPU?**
A: No, these are standard benchmarks similar to what GPU manufacturers run.

**Q: How accurate are model size estimates?**
A: ±20%, depends on exact architecture and optimization.

**Q: Why separate CLI and Web UI?**
A: CLI for automation/HPC, Web for interactive use. See [ARCHITECTURE.md](./ARCHITECTURE.md).

**Q: Does it work on AMD GPUs?**
A: Not yet. ROCm support planned for future release.

## Roadmap

### ✅ Completed
- Hardware detection with GPU classification
- Memory & compute profiling
- Real inference benchmarks
- Multi-GPU parallel testing
- NVLink topology detection
- Burn-in stability testing
- CLI orchestration (112 passing tests)
- Web dashboard with HMI-style visualization

### 🚧 In Progress
- Server-Sent Events for live progress
- Enhanced UI with expandable test results
- Generation config warning cleanup

### 📋 Planned
- Full HPC cluster support (SLURM, multi-node)
- InfiniBand topology detection
- Automated baseline comparison
- Prometheus metrics export
- AMD ROCm support

## License

MIT - Part of the Train Your Dragon Transformer Training Interface project.

## Support

- Issues: [GitHub Issues](https://github.com/anthropics/train-your-dragon)
- Docs: See [ARCHITECTURE.md](./ARCHITECTURE.md) for design details
- Web UI: See [web/README.md](./web/README.md) for dashboard docs

Happy benchmarking! 🚀
