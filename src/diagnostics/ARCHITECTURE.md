# Hardware Diagnostics Architecture

## Design Philosophy

The diagnostics suite follows a **modular, dual-interface architecture** that separates concerns between:
- **Core diagnostic logic** (backend-agnostic, reusable)
- **CLI interface** (automation, scripting, HPC)
- **Web interface** (interactive, visual, development)

This separation is **intentional and beneficial** - not technical debt.

## Why Two Interfaces?

### CLI: Automation & Production
**Use cases:**
- HPC cluster job submission (`sbatch diagnostics.sh`)
- CI/CD pipelines (`pytest && python -m diagnostics.cli`)
- Headless servers (no X11/graphics)
- Scripting and automation (`for gpu in 0 1 2 3; do ...`)
- Remote SSH sessions
- Docker containers
- Cron jobs for monitoring

**Advantages:**
- No dependencies on web frameworks
- Stdout/stderr for logging
- Exit codes for automation
- JSON output for parsing
- Fast startup

### Web UI: Interactive Development
**Use cases:**
- Local development machines
- Interactive exploration
- Debugging performance issues
- Demos and presentations
- Real-time visualization
- Training new team members

**Advantages:**
- Visual topology diagrams
- Live progress updates
- Clickable, explorable results
- No command-line knowledge needed
- Download formatted reports

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│           Diagnostic Core Modules                │
│  (backend-agnostic, shared by CLI and Web)       │
├─────────────────────────────────────────────────┤
│ • hardware_detector.py                           │
│ • compute_profiler.py                            │
│ • memory_profiler.py                             │
│ • inference_profiler.py                          │
│ • health_monitor.py                              │
│ • topology_detector.py                           │
│ • multi_gpu_profiler.py                          │
│ • burn_in.py                                     │
│ • model_loader.py                                │
│ • metrics.py (TestResult, TestSuite, etc.)       │
└─────────────────────────────────────────────────┘
         ▲                           ▲
         │                           │
         │                           │
    ┌────┴─────┐              ┌─────┴──────┐
    │   CLI    │              │  Web UI    │
    │ cli.py   │              │ server.py  │
    └──────────┘              └────────────┘
         │                           │
         │                           │
    stdout/JSON              FastAPI + SSE
    exit codes               HTML/CSS/JS
```

## Core Modules (DRY Principle)

All diagnostic logic lives in `src/diagnostics/core/`. Both interfaces import these modules.

### Hardware Detection
- `backend_detector.py` - CUDA/ROCm/CPU detection
- `hardware_detector.py` - GPU specs, VRAM, device capabilities
- `topology_detector.py` - PCIe tree, NVLink, NUMA nodes

### Performance Profiling
- `compute_profiler.py` - GEMM benchmarks (TFLOPS)
- `inference_profiler.py` - Real transformer models (tokens/sec)
- `memory_profiler.py` - Bandwidth, allocation patterns
- `cpu_profiler.py` - CPU fallback benchmarks

### Multi-GPU Support
- `multi_gpu_profiler.py` - Parallel GPU benchmarking
- NVLink topology parsing
- Aggregate metrics

### Health Monitoring
- `health_monitor.py` - Temperature, power, clocks, throttling
- P-state tracking (P0/P2/P8)
- PCIe bandwidth validation

### Stress Testing
- `burn_in.py` - Extended stability testing
- Thermal validation
- Error detection

### Test Framework
- `metrics.py` - TestResult, TestSuite, TestStatus
- Progress tracking
- Standardized result formatting

## CLI Orchestration

**File:** `cli.py`

**Responsibilities:**
- Argument parsing (`--mode`, `--output`)
- Test sequencing (hardware → memory → compute → inference)
- Progress reporting to stdout
- Result aggregation
- Exit code generation

**Example:**
```python
cli = DiagnosticsCLI()
cli.run_all_tests(mode=TestMode.DEEP)
# Runs: HardwareDetector → MemoryProfiler → ComputeProfiler → InferenceProfiler
```

**Output:**
- Human-readable terminal output
- Optional JSON export
- Exit code 0 (success) or 1 (failure)

## Web Interface

**Files:**
- `web/server.py` - FastAPI backend
- `web/static/index.html` - HMI-style dashboard

**Responsibilities:**
- REST API endpoints
- Topology visualization
- Live benchmark execution
- Report generation
- **Reuses CLI orchestration** (no duplication!)

**Key Insight:**
Web server calls `DiagnosticsCLI().run_all_tests()` internally - it doesn't reimplement test logic.

```python
# server.py
cli = DiagnosticsCLI()
cli.run_all_tests(mode=test_mode)
results = cli.suite.to_dict()  # JSON serialization built-in
```

## Data Flow

### CLI Flow
```
User → cli.py → DiagnosticsCLI.run_all_tests()
                       ↓
              Hardware/Memory/Compute/Inference Profilers
                       ↓
              TestSuite with TestResults
                       ↓
              Print to stdout + exit code
```

### Web Flow
```
Browser → FastAPI → DiagnosticsCLI.run_all_tests()
                           ↓
                  Same profilers (DRY!)
                           ↓
                  TestSuite.to_dict()
                           ↓
                  JSON → Browser → Render
```

## Model Caching

**Location:** `~/.cache/diagnostics/models/`

**Models downloaded:**
- DistilBERT-base (256 MB)
- TinyLlama-1.1B (4.4 GB)
- LLaVA-1.5-7B (13 GB) - optional for large VRAM

**Note:** Models are shared between CLI and Web interfaces. Download once, use everywhere.

## HPC Deployment

### Single Node (Multi-GPU)
```bash
# CLI
python -m diagnostics.cli --mode deep --output report.json

# Web (for live visualization)
python -m diagnostics.web.server
# Then: ssh -L 8000:localhost:8000 user@node
```

### SLURM Cluster (Future)
```bash
sbatch --gres=gpu:8 diagnostics_job.sh
# Job script calls CLI, outputs JSON
# Aggregate results across nodes
```

## Testing Strategy

**Unit tests:** Test individual profilers in isolation
**Integration tests:** Test CLI orchestration end-to-end
**Functional tests:** Verify actual GPU measurements

**Test count:** 112 passing tests

**Run tests:**
```bash
pytest src/diagnostics/tests/ -v
```

## Extension Points

### Adding New Tests

1. **Create profiler** in `core/`:
```python
class NewProfiler(DiagnosticTest):
    def run(self, mode: TestMode) -> TestResult:
        # Your logic here
        return self._create_result(...)
```

2. **Add to CLI orchestration** in `cli.py`:
```python
profiler = NewProfiler()
result = self._run_test(profiler, mode)
self.suite.add_result(result)
```

3. **Web UI gets it automatically** (via `suite.to_dict()`)

### Adding New Metrics

1. Update profiler to populate `metrics` dict
2. Add interpretation logic
3. No UI changes needed (rendered generically)

## Future Enhancements

**Documented for future PRs:**
- [ ] Server-Sent Events for live progress
- [ ] Expandable test result cards in UI
- [ ] Multi-node HPC support (MPI coordination)
- [ ] InfiniBand topology detection
- [ ] Automated baseline comparison
- [ ] Prometheus metrics export
- [ ] Grafana dashboard templates

## Dependencies

**Core:**
- PyTorch (CUDA support)
- transformers (for inference profiling)
- huggingface_hub (model downloads)

**Web only:**
- FastAPI
- uvicorn

**CLI only:**
- No extra dependencies (uses core)

## Summary

**The dual-interface design is a feature, not a bug:**
- Maximizes code reuse (DRY)
- Serves different use cases (HPC vs dev)
- Can evolve independently
- Both remain first-class citizens

CLI and Web are **presentation layers** over shared diagnostic core.
