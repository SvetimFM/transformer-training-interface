# Hardware Diagnostics Web Dashboard

**Industrial HMI-Style System Topology Visualization**

## Features

✅ **Real-time system topology detection**
- CPU and PCIe hierarchy
- GPU detection with full metrics
- NVMe drives (on bare-metal Linux)

✅ **Integrated benchmark system**
- Quick mode (~30 seconds)
- Deep mode (~2-3 minutes)
- Real-time performance metrics

✅ **HMI-style visualization**
- Engineering drawing layout
- Color-coded health status
- Numeric readouts for all metrics
- DCS/SCADA inspired design

## Quick Start

### 1. Start the Web Server

```bash
cd src/diagnostics/web
python3 server.py
```

The server will start on `http://localhost:8000`

### 2. Open in Browser

Navigate to: `http://localhost:8000`

You'll see:
- System topology diagram (CPU → GPU hierarchy)
- Real-time metrics (TFLOPS, efficiency, temperature, power)
- Control buttons to run benchmarks

### 3. Run Benchmarks

Click the buttons to run diagnostics:
- **Refresh Topology**: Reload system detection
- **Run Quick Benchmark**: ~30 second test (FP16 + FP32)
- **Run Deep Benchmark**: ~2-3 minute test (all precisions)

Results automatically update the topology visualization!

## API Endpoints

### GET `/api/topology`
Returns current system topology with benchmark results (if available).

**Response:**
```json
{
  "device_id": "cpu0",
  "device_type": "cpu",
  "name": "AMD Ryzen 7 5800X",
  "metrics": {...},
  "children": [
    {
      "device_id": "gpu0",
      "device_type": "gpu",
      "name": "NVIDIA GeForce RTX 3070 Ti",
      "metrics": {
        "fp16_tflops": 43.16,
        "compute_efficiency_percent": 52.6,
        "temperature_celsius": 40,
        ...
      },
      "health_status": "warning"
    }
  ]
}
```

### POST `/api/benchmark/{mode}`
Run benchmark and update topology.

**Parameters:**
- `mode`: "quick" or "deep"

**Response:**
```json
{
  "status": "complete",
  "metrics": {...}
}
```

## Dashboard Features

### Color-Coded Health Status

- 🟢 **Green** (Healthy): Efficiency ≥ 60%
- 🟡 **Yellow** (Warning): Efficiency 40-60%
- 🔴 **Red** (Error): Efficiency < 40%

### Metrics Displayed

**GPU:**
- FP16 Performance (TFLOPS)
- Compute Efficiency (% of theoretical peak)
- Temperature (°C)
- Clock Speed (MHz)
- Power Draw (W)
- VRAM (GB)
- PCIe Generation and Lanes

**CPU:**
- Model name
- Core count
- Platform

## Architecture

```
src/diagnostics/web/
├── server.py              # FastAPI backend
├── static/
│   └── index.html         # HMI-style dashboard
└── README.md             # This file
```

**Backend:** FastAPI + Uvicorn
**Frontend:** Vanilla JavaScript + SVG/CSS
**Style:** Industrial HMI/SCADA control panel

## Performance Analysis Results

**Current System (RTX 3070 Ti):**
- **Measured:** 43 TFLOPS FP16 @ 1935 MHz
- **Theoretical Peak:** 82 TFLOPS
- **Efficiency:** 52.6%
- **Status:** ✅ Excellent for consumer GPU (50-60% typical)

**Bottleneck Analysis:**
- ✅ Power draw: 245W (85% of limit) - fully loaded
- ✅ SM utilization: 99-100% - no compute headroom
- ✅ Temperature: 74°C - thermal headroom available
- ✅ Conclusion: Operating at practical maximum

**Why not 75%+ efficiency?**
- Consumer GPUs (RTX 30 series): 50-60% typical
- Data center GPUs (A100): 75-80% typical
- Difference: Better scheduling, ECC memory, optimized firmware

## Next Steps

### Extend Topology Detection
- Add multi-GPU support
- Detect NVLink connections
- Add network interfaces
- Parse full PCIe tree on bare-metal

### Enhanced Visualizations
- Before/after comparison view
- Performance trend graphs
- Thermal history charts
- Power consumption timeline

### Advanced Features
- Export results to JSON/CSV
- Save baseline comparisons
- Automated optimization suggestions
- Integration with training pipelines

## Troubleshooting

**"Error loading topology"**
- Make sure server is running (`python3 server.py`)
- Check console for Python errors
- Verify nvidia-smi is accessible

**No GPU detected**
- Ensure CUDA drivers are installed
- Run `nvidia-smi` to verify GPU visibility
- Check if WSL2 has GPU passthrough enabled

**Benchmarks fail**
- Verify PyTorch + CUDA installation
- Check GPU memory availability
- Look for Python errors in server logs

## Credits

Built for the Train Your Dragon transformer training interface.

Hardware Diagnostics System by Claude & User
- Topology detection
- Performance profiling
- HMI visualization
