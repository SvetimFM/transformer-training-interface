"""
Multi-GPU Parallel Profiler

Runs benchmarks on all available GPUs concurrently.
Aggregates results and provides system-wide performance metrics.

Used for:
- DGX A100 8-GPU nodes
- Multi-GPU workstations
- HPC clusters (single node)
"""

import torch
import threading
from typing import Dict, List, Any
from .compute_profiler import ComputeProfiler
from .metrics import TestMode, TestResult


class MultiGPUProfiler:
    """
    Profile all GPUs in parallel.

    Runs compute benchmarks on each GPU simultaneously using threading.
    """

    def __init__(self):
        """Initialize multi-GPU profiler"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.gpu_count = torch.cuda.device_count()
        self.results: Dict[int, TestResult] = {}

    def run_all_gpus(self, mode: TestMode = TestMode.QUICK) -> Dict[str, Any]:
        """
        Run benchmarks on all GPUs in parallel.

        Args:
            mode: TestMode (QUICK or DEEP)

        Returns:
            Dict with per-GPU results and aggregate metrics
        """
        self.results = {}
        threads = []

        # Launch benchmark thread for each GPU
        for gpu_id in range(self.gpu_count):
            thread = threading.Thread(
                target=self._benchmark_gpu,
                args=(gpu_id, mode)
            )
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Aggregate results
        return self._aggregate_results()

    def _benchmark_gpu(self, gpu_id: int, mode: TestMode):
        """
        Run benchmark on a single GPU (called in thread).

        Args:
            gpu_id: GPU index
            mode: TestMode
        """
        try:
            device = f"cuda:{gpu_id}"
            profiler = ComputeProfiler(device=device)
            result = profiler.run(mode=mode)
            self.results[gpu_id] = result
        except Exception as e:
            # Store error result
            from .metrics import TestStatus
            self.results[gpu_id] = TestResult(
                test_name=f"GPU {gpu_id} Compute Profiler",
                status=TestStatus.FAILED,
                metrics={},
                interpretation=f"GPU {gpu_id} benchmark failed",
                recommendation="Check GPU availability",
                error_message=str(e)
            )

    def _aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate per-GPU results into system-wide metrics.

        Returns:
            Dict with per_gpu results and aggregated totals
        """
        aggregate = {
            "per_gpu": {},
            "aggregate": {
                "total_gpus": self.gpu_count,
                "successful_gpus": 0,
                "failed_gpus": 0,
            }
        }

        # Collect metrics
        total_fp16 = 0
        total_fp32 = 0
        total_vram = 0
        all_efficiencies = []

        for gpu_id, result in self.results.items():
            # Store per-GPU result
            aggregate["per_gpu"][f"gpu{gpu_id}"] = {
                "status": result.status.value,
                "metrics": result.metrics,
                "interpretation": result.interpretation
            }

            # Aggregate if successful
            if result.status.value == "passed":
                aggregate["aggregate"]["successful_gpus"] += 1

                if "fp16_tflops" in result.metrics:
                    total_fp16 += result.metrics["fp16_tflops"]
                if "fp32_tflops" in result.metrics:
                    total_fp32 += result.metrics["fp32_tflops"]
                if "compute_efficiency_percent" in result.metrics:
                    all_efficiencies.append(result.metrics["compute_efficiency_percent"])
            else:
                aggregate["aggregate"]["failed_gpus"] += 1

        # Calculate aggregate metrics
        if aggregate["aggregate"]["successful_gpus"] > 0:
            aggregate["aggregate"]["total_fp16_tflops"] = round(total_fp16, 2)
            aggregate["aggregate"]["total_fp32_tflops"] = round(total_fp32, 2)

            if all_efficiencies:
                aggregate["aggregate"]["avg_efficiency_percent"] = round(
                    sum(all_efficiencies) / len(all_efficiencies), 1
                )

        return aggregate


def main():
    """Test multi-GPU profiling"""
    print("="*80)
    print("MULTI-GPU PARALLEL PROFILER")
    print("="*80)
    print()

    profiler = MultiGPUProfiler()
    print(f"Detected {profiler.gpu_count} GPU(s)")
    print()

    if profiler.gpu_count == 1:
        print("⚠️  Only 1 GPU detected - multi-GPU features won't be demonstrated")
        print("   Running anyway for testing...")
        print()

    print("Running benchmarks on all GPUs in parallel...")
    print("(This will take ~30 seconds for quick mode)")
    print()

    results = profiler.run_all_gpus(mode=TestMode.QUICK)

    print("="*80)
    print("RESULTS")
    print("="*80)
    print()

    # Per-GPU results
    print("Per-GPU Performance:")
    print("-" * 80)
    for gpu_key, gpu_data in results["per_gpu"].items():
        gpu_id = gpu_key.replace("gpu", "")
        metrics = gpu_data["metrics"]

        print(f"  {gpu_key.upper()}:")
        print(f"    Status: {gpu_data['status']}")
        if "fp16_tflops" in metrics:
            print(f"    FP16: {metrics['fp16_tflops']} TFLOPS")
        if "compute_efficiency_percent" in metrics:
            print(f"    Efficiency: {metrics['compute_efficiency_percent']}%")
        print()

    # Aggregate
    print("Aggregate Metrics:")
    print("-" * 80)
    agg = results["aggregate"]
    print(f"  Total GPUs: {agg['total_gpus']}")
    print(f"  Successful: {agg['successful_gpus']}")
    print(f"  Failed: {agg['failed_gpus']}")
    if "total_fp16_tflops" in agg:
        print(f"  Total FP16: {agg['total_fp16_tflops']} TFLOPS")
    if "avg_efficiency_percent" in agg:
        print(f"  Average Efficiency: {agg['avg_efficiency_percent']}%")

    print()
    print("="*80)


if __name__ == "__main__":
    main()
