# Architecture

## Why Two Interfaces?

The diagnostics suite has both a CLI and a web UI because they serve different use cases:

### CLI
- Scripting and automation
- CI/CD pipelines
- Headless servers
- SSH sessions

```bash
python -m diagnostics.cli --mode deep --output results.json
```

### Web UI
- Interactive exploration
- Visualization
- Quick checks on desktop machines

```bash
python src/diagnostics/web/server.py
# Open http://localhost:8000
```

## Code Structure

Both interfaces call the same core modules. No duplication.

```
Core Modules (diagnostics/core/)
├─ hardware_detector.py
├─ memory_profiler.py
├─ compute_profiler.py
├─ inference_profiler.py
└─ nccl_profiler.py
         ↑
         │
    ┌────┴─────┐
    │          │
  CLI        Web UI
  cli.py     server.py
```

### CLI Flow
```
User → cli.py → run_all_tests()
              → HardwareDetector().run()
              → MemoryProfiler().run()
              → ComputeProfiler().run()
              → InferenceProfiler().run()
              → Print to stdout
```

### Web Flow
```
Browser → FastAPI → DiagnosticsCLI().run_all_tests()
                  → (same profilers)
                  → JSON response
```

The web server literally calls the CLI internally. It's just a wrapper.

## Core Modules

### DiagnosticTest Base Class
All profilers inherit from this:

```python
class ComputeProfiler(DiagnosticTest):
    def run(self, mode: TestMode) -> TestResult:
        # Do benchmarks
        return TestResult(metrics={...})
```

Returns a `TestResult` with:
- `status`: passed/failed/skipped
- `metrics`: dict of measurements
- `interpretation`: list of human-readable conclusions

### TestSuite
Collects results from all profilers:

```python
suite = TestSuite()
suite.add_result(hardware_result)
suite.add_result(compute_result)
# ...
suite.to_dict()  # JSON-serializable
```

## Model Caching

Models downloaded to `~/.cache/diagnostics/models/`:
- Qwen2.5-0.5B (~1GB)
- TinyLlama-1.1B (~4GB)
- Shared between CLI and Web

## Adding New Tests

1. Create profiler in `core/`:
```python
class NewProfiler(DiagnosticTest):
    def run(self, mode):
        # Your test logic
        return self._create_result(metrics={...})
```

2. Add to CLI in `cli.py`:
```python
result = self._run_test(NewProfiler(), mode)
self.suite.add_result(result)
```

3. Web UI gets it automatically via `suite.to_dict()`

## That's It

No complex dependency injection, no event buses, no microservices. Just:
- Profilers that return results
- CLI that prints them
- Web server that JSONifies them

Keep it simple.
