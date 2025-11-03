"""
FastAPI Web Server for Hardware Diagnostics Dashboard

Serves:
- System topology API endpoints
- HMI-style visualization dashboard
- Real-time benchmark progress (via Server-Sent Events)
"""

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diagnostics.core.topology_detector import TopologyDetector
from diagnostics.core.compute_profiler import ComputeProfiler
from diagnostics.core.multi_gpu_profiler import MultiGPUProfiler
from diagnostics.core.metrics import TestMode
from diagnostics.cli import DiagnosticsCLI

app = FastAPI(title="Hardware Diagnostics Dashboard")

# Global topology cache
_topology_cache = None
_benchmark_cache = None


@app.get("/")
async def root():
    """Serve main dashboard HTML"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    else:
        return HTMLResponse("<h1>Dashboard Coming Soon...</h1>")


@app.get("/api/topology")
async def get_topology():
    """Get system topology with benchmark results"""
    global _topology_cache, _benchmark_cache

    # Detect topology
    detector = TopologyDetector()
    detector.detect()

    # If we have benchmark cache, enrich topology
    if _benchmark_cache:
        detector.enrich_with_benchmarks(_benchmark_cache)

    _topology_cache = detector.topology

    return detector.topology.to_dict()


@app.post("/api/benchmark/{mode}")
async def run_benchmark(mode: str):
    """
    Run full diagnostic suite using CLI orchestration.

    Args:
        mode: "quick" or "deep"

    Note: This will take 30s-3min depending on mode!
    For DEEP mode, loads real transformer models for accurate testing.
    """
    global _benchmark_cache
    import time
    import io
    import sys

    test_mode = TestMode.QUICK if mode == "quick" else TestMode.DEEP

    # Redirect stdout to capture CLI output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        # Run full CLI diagnostic suite
        start_time = time.time()
        cli = DiagnosticsCLI()
        cli.run_all_tests(mode=test_mode)
        elapsed = time.time() - start_time

        # Restore stdout
        cli_output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # Extract results from CLI suite
        suite_dict = cli.suite.to_dict()

        # Find compute profiler results for caching
        compute_metrics = {}
        for test in cli.suite.tests:
            if "Compute Profiler" in test.test_name:
                compute_metrics = test.metrics
                break

        # Cache results for topology enrichment
        _benchmark_cache = {
            'gpu0': compute_metrics
        }

        return {
            "status": "complete",
            "mode": mode,
            "duration_seconds": round(elapsed, 1),
            "suite": suite_dict,
            "compute_metrics": compute_metrics,
            "test_count": len(cli.suite.tests),
            "passed_count": sum(1 for t in cli.suite.tests if t.status.value == "passed")
        }

    except Exception as e:
        # Restore stdout on error
        sys.stdout = old_stdout
        return {
            "status": "error",
            "error": str(e),
            "message": "Benchmark failed"
        }
    finally:
        # Ensure stdout is always restored
        if sys.stdout != old_stdout:
            sys.stdout = old_stdout


@app.post("/api/benchmark/multi-gpu/{mode}")
async def run_multi_gpu_benchmark(mode: str):
    """
    Run benchmarks on all GPUs in parallel.

    Args:
        mode: "quick" or "deep"

    Note: This will take 30s-3min depending on mode!
    Works only on multi-GPU systems.
    """
    global _benchmark_cache
    import time

    test_mode = TestMode.QUICK if mode == "quick" else TestMode.DEEP

    try:
        # Run multi-GPU benchmark
        start_time = time.time()
        profiler = MultiGPUProfiler()
        results = profiler.run_all_gpus(mode=test_mode)
        elapsed = time.time() - start_time

        # Cache all GPU results (per_gpu dict contains gpu0, gpu1, etc.)
        _benchmark_cache = results["per_gpu"]

        return {
            "status": "complete",
            "mode": mode,
            "duration_seconds": round(elapsed, 1),
            "results": results,
            "summary": {
                "total_gpus": results["aggregate"]["total_gpus"],
                "successful_gpus": results["aggregate"]["successful_gpus"],
                "total_fp16_tflops": results["aggregate"].get("total_fp16_tflops", 0),
                "avg_efficiency_percent": results["aggregate"].get("avg_efficiency_percent", 0)
            }
        }
    except RuntimeError as e:
        # CUDA not available or single GPU
        if "CUDA is not available" in str(e):
            return {
                "status": "error",
                "error": "CUDA not available",
                "message": "Multi-GPU benchmarks require CUDA support"
            }
        else:
            return {
                "status": "error",
                "error": str(e),
                "message": "Multi-GPU benchmark failed"
            }


@app.get("/api/report/download")
async def download_report():
    """
    Generate and download complete system report with all metrics.

    Returns JSON file with:
    - System topology
    - Benchmark results
    - Performance analysis
    - Recommendations
    """
    import json
    from datetime import datetime

    global _benchmark_cache

    # Get topology
    detector = TopologyDetector()
    detector.detect()

    if _benchmark_cache:
        detector.enrich_with_benchmarks(_benchmark_cache)

    # Build comprehensive report
    report = {
        "generated_at": datetime.now().isoformat(),
        "system_topology": detector.topology.to_dict(),
        "benchmark_results": _benchmark_cache or {},
        "analysis": {
            "gpu_count": len([c for c in detector.topology.children if c.device_type == "gpu"]),
            "total_vram_gb": sum(
                c.metrics.get('vram_gb', 0)
                for c in detector.topology.children
                if c.device_type == "gpu"
            )
        }
    }

    # Add performance breakdown if we have benchmark data
    if _benchmark_cache and 'gpu0' in _benchmark_cache:
        metrics = _benchmark_cache['gpu0']
        if 'compute_efficiency_percent' in metrics:
            report["analysis"]["efficiency_breakdown"] = {
                "measured_efficiency": metrics['compute_efficiency_percent'],
                "theoretical_peak_tflops": metrics.get('theoretical_peak_fp16_tflops'),
                "actual_tflops": metrics.get('fp16_tflops'),
                "bottlenecks": []
            }

            # Identify bottlenecks
            if metrics.get('throttled'):
                report["analysis"]["efficiency_breakdown"]["bottlenecks"].append({
                    "type": "thermal_throttling",
                    "reason": metrics.get('throttle_reason'),
                    "impact": "performance_limited"
                })

            clock_pct = (metrics.get('clock_mhz_during_test', 0) /
                        metrics.get('clock_max_mhz', 1) * 100)
            if clock_pct < 90:
                report["analysis"]["efficiency_breakdown"]["bottlenecks"].append({
                    "type": "low_clocks",
                    "clock_actual_mhz": metrics.get('clock_mhz_during_test'),
                    "clock_max_mhz": metrics.get('clock_max_mhz'),
                    "impact_percent": round(100 - clock_pct, 1)
                })

    # Return as downloadable JSON
    filename = f"hardware_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_str = json.dumps(report, indent=2)

    return Response(
        content=json_str,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@app.get("/api/benchmark/results")
async def get_benchmark_results():
    """Get cached benchmark results"""
    global _benchmark_cache
    return _benchmark_cache or {}


if __name__ == "__main__":
    import uvicorn
    print("Starting Hardware Diagnostics Dashboard...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
