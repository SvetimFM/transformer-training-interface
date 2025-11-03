"""
NCCL Communication Profiler

Tests inter-GPU communication performance using PyTorch distributed primitives.
Critical for validating multi-GPU training infrastructure.

Tests:
- All-Reduce bandwidth (gradient synchronization)
- Broadcast bandwidth (parameter distribution)
- Point-to-Point transfers (GPU-to-GPU)
- Latency measurements
- Message size scaling
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .metrics import DiagnosticTest, TestResult, TestStatus, TestMode, MetricType


@dataclass
class NCCLBenchmarkResult:
    """Result from a single NCCL benchmark"""
    operation: str
    message_size_mb: float
    bandwidth_gbps: float
    latency_ms: float
    num_gpus: int
    error: Optional[str] = None


class NCCLProfiler(DiagnosticTest):
    """
    Profile inter-GPU communication using NCCL.

    Spawns one process per GPU and runs collective communication benchmarks.
    Measures bandwidth and latency for different message sizes.
    """

    def __init__(self):
        super().__init__(
            name="NCCL Communication Profiler",
            description="Measure inter-GPU communication bandwidth (All-Reduce, Broadcast, P2P)",
            metric_type=MetricType.COMMUNICATION
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self.gpu_count = torch.cuda.device_count()

        if self.gpu_count < 2:
            raise RuntimeError(
                f"NCCL tests require at least 2 GPUs, found {self.gpu_count}. "
                "For single-GPU systems, this test is skipped."
            )

        if not dist.is_nccl_available():
            raise RuntimeError("NCCL backend not available in PyTorch")

    def run(self, mode: TestMode = TestMode.QUICK) -> TestResult:
        """
        Run NCCL communication benchmarks.

        Args:
            mode: QUICK tests small messages, DEEP tests full range
        """
        try:
            # Define message sizes to test (in MB)
            if mode == TestMode.QUICK:
                # Quick: Just test a few key sizes
                message_sizes_mb = [1, 64, 256]  # 1MB, 64MB, 256MB
                num_iterations = 10
            else:
                # Deep: Test full range
                message_sizes_mb = [1, 4, 16, 64, 256, 512, 1024]  # 1MB to 1GB
                num_iterations = 20

            # Run benchmarks using multiprocessing
            results = self._run_distributed_benchmarks(message_sizes_mb, num_iterations)

            # Aggregate results
            metrics = self._aggregate_results(results)

            # Interpret results
            interpretation = self._interpret_results(metrics)

            return self._create_result(
                status=TestStatus.PASSED,
                metrics=metrics,
                interpretation=interpretation
            )

        except Exception as e:
            return self._create_result(
                status=TestStatus.FAILED,
                metrics={},
                error_message=str(e)
            )

    def _run_distributed_benchmarks(
        self,
        message_sizes_mb: List[float],
        num_iterations: int
    ) -> List[NCCLBenchmarkResult]:
        """
        Run benchmarks across all GPUs using multiprocessing.

        Spawns one process per GPU and runs NCCL collectives.
        """
        # Set environment for distributed training
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'

        # Use multiprocessing to spawn worker processes
        # We'll use a shared queue to collect results
        mp.set_start_method('spawn', force=True)

        # Create result queue
        manager = mp.Manager()
        result_queue = manager.Queue()

        # Spawn processes (one per GPU)
        processes = []
        for rank in range(self.gpu_count):
            p = mp.Process(
                target=self._worker_process,
                args=(rank, self.gpu_count, message_sizes_mb, num_iterations, result_queue)
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Collect results from queue
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        return results

    @staticmethod
    def _worker_process(
        rank: int,
        world_size: int,
        message_sizes_mb: List[float],
        num_iterations: int,
        result_queue
    ):
        """
        Worker process that runs on a single GPU.

        Args:
            rank: Process rank (GPU ID)
            world_size: Total number of processes (GPUs)
            message_sizes_mb: Message sizes to test
            num_iterations: Number of iterations per test
            result_queue: Queue to put results
        """
        # Set device for this process
        torch.cuda.set_device(rank)

        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )

        try:
            # Only rank 0 collects and reports results
            if rank == 0:
                # Test All-Reduce
                for size_mb in message_sizes_mb:
                    result = NCCLProfiler._benchmark_allreduce(
                        size_mb, num_iterations, rank, world_size
                    )
                    result_queue.put(result)

                # Test Broadcast
                for size_mb in message_sizes_mb:
                    result = NCCLProfiler._benchmark_broadcast(
                        size_mb, num_iterations, rank, world_size
                    )
                    result_queue.put(result)

                # Test Point-to-Point (rank 0 <-> rank 1)
                if world_size >= 2:
                    for size_mb in message_sizes_mb:
                        result = NCCLProfiler._benchmark_p2p(
                            size_mb, num_iterations, rank, world_size
                        )
                        result_queue.put(result)
            else:
                # Non-rank-0 processes participate in collectives
                for size_mb in message_sizes_mb:
                    # All-Reduce
                    NCCLProfiler._benchmark_allreduce(
                        size_mb, num_iterations, rank, world_size
                    )
                    # Broadcast
                    NCCLProfiler._benchmark_broadcast(
                        size_mb, num_iterations, rank, world_size
                    )
                    # P2P (only rank 1 participates)
                    if rank == 1 and world_size >= 2:
                        NCCLProfiler._benchmark_p2p(
                            size_mb, num_iterations, rank, world_size
                        )

        finally:
            # Clean up process group
            dist.destroy_process_group()

    @staticmethod
    def _benchmark_allreduce(
        size_mb: float,
        num_iterations: int,
        rank: int,
        world_size: int
    ) -> Optional[NCCLBenchmarkResult]:
        """
        Benchmark All-Reduce operation.

        All-Reduce is critical for gradient synchronization in distributed training.
        """
        # Calculate tensor size
        num_elements = int(size_mb * 1024 * 1024 / 4)  # 4 bytes per float32

        # Create tensor
        tensor = torch.randn(num_elements, dtype=torch.float32, device='cuda')

        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Calculate metrics (only rank 0 returns result)
        if rank == 0:
            # All-Reduce moves 2(N-1)/N * size data per GPU (ring algorithm)
            # For simplicity, use 2 * size (each GPU sends and receives)
            data_transferred_gb = (size_mb / 1024) * 2 * num_iterations
            bandwidth_gbps = data_transferred_gb / elapsed
            latency_ms = (elapsed / num_iterations) * 1000

            return NCCLBenchmarkResult(
                operation="all_reduce",
                message_size_mb=size_mb,
                bandwidth_gbps=bandwidth_gbps,
                latency_ms=latency_ms,
                num_gpus=world_size
            )

        return None

    @staticmethod
    def _benchmark_broadcast(
        size_mb: float,
        num_iterations: int,
        rank: int,
        world_size: int
    ) -> Optional[NCCLBenchmarkResult]:
        """
        Benchmark Broadcast operation.

        Broadcast is used for distributing parameters/buffers across GPUs.
        """
        num_elements = int(size_mb * 1024 * 1024 / 4)
        tensor = torch.randn(num_elements, dtype=torch.float32, device='cuda')

        # Warmup
        for _ in range(5):
            dist.broadcast(tensor, src=0)

        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            dist.broadcast(tensor, src=0)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        if rank == 0:
            # Broadcast sends size_mb to (world_size - 1) GPUs
            data_transferred_gb = (size_mb / 1024) * (world_size - 1) * num_iterations
            bandwidth_gbps = data_transferred_gb / elapsed
            latency_ms = (elapsed / num_iterations) * 1000

            return NCCLBenchmarkResult(
                operation="broadcast",
                message_size_mb=size_mb,
                bandwidth_gbps=bandwidth_gbps,
                latency_ms=latency_ms,
                num_gpus=world_size
            )

        return None

    @staticmethod
    def _benchmark_p2p(
        size_mb: float,
        num_iterations: int,
        rank: int,
        world_size: int
    ) -> Optional[NCCLBenchmarkResult]:
        """
        Benchmark Point-to-Point transfer.

        Tests direct GPU-to-GPU bandwidth (important for pipeline parallelism).
        """
        num_elements = int(size_mb * 1024 * 1024 / 4)
        tensor = torch.randn(num_elements, dtype=torch.float32, device='cuda')

        # Warmup
        for _ in range(5):
            if rank == 0:
                dist.send(tensor, dst=1)
            elif rank == 1:
                dist.recv(tensor, src=0)

        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            if rank == 0:
                dist.send(tensor, dst=1)
            elif rank == 1:
                dist.recv(tensor, src=0)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        if rank == 0:
            # P2P sends size_mb once per iteration
            data_transferred_gb = (size_mb / 1024) * num_iterations
            bandwidth_gbps = data_transferred_gb / elapsed
            latency_ms = (elapsed / num_iterations) * 1000

            return NCCLBenchmarkResult(
                operation="point_to_point",
                message_size_mb=size_mb,
                bandwidth_gbps=bandwidth_gbps,
                latency_ms=latency_ms,
                num_gpus=2  # P2P is always between 2 GPUs
            )

        return None

    def _aggregate_results(self, results: List[NCCLBenchmarkResult]) -> Dict[str, Any]:
        """
        Aggregate benchmark results into metrics dict.
        """
        metrics = {
            "num_gpus": self.gpu_count,
            "operations_tested": []
        }

        # Group by operation
        by_operation = {}
        for result in results:
            if result.operation not in by_operation:
                by_operation[result.operation] = []
            by_operation[result.operation].append(result)

        # Find peak bandwidth for each operation
        for operation, op_results in by_operation.items():
            metrics["operations_tested"].append(operation)

            # Find peak bandwidth (usually at largest message size)
            peak_bw = max(r.bandwidth_gbps for r in op_results)
            metrics[f"{operation}_peak_bandwidth_gbps"] = round(peak_bw, 2)

            # Find latency at smallest message size
            min_latency = min(r.latency_ms for r in op_results)
            metrics[f"{operation}_min_latency_ms"] = round(min_latency, 3)

            # Store all results for detailed analysis
            metrics[f"{operation}_results"] = [
                {
                    "size_mb": r.message_size_mb,
                    "bandwidth_gbps": round(r.bandwidth_gbps, 2),
                    "latency_ms": round(r.latency_ms, 3)
                }
                for r in sorted(op_results, key=lambda x: x.message_size_mb)
            ]

        return metrics

    def _interpret_results(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Interpret NCCL benchmark results.
        """
        interpretations = []

        num_gpus = metrics.get("num_gpus", 0)
        interpretations.append(f"Tested inter-GPU communication on {num_gpus} GPUs")

        # Interpret All-Reduce (most critical for training)
        if "all_reduce_peak_bandwidth_gbps" in metrics:
            ar_bw = metrics["all_reduce_peak_bandwidth_gbps"]
            ar_lat = metrics["all_reduce_min_latency_ms"]

            interpretations.append(
                f"All-Reduce: {ar_bw:.1f} GB/s peak, {ar_lat:.2f} ms latency"
            )

            # Expected bandwidth based on interconnect
            # NVLink: 300-600 GB/s, PCIe Gen4 x16: 16-32 GB/s
            if ar_bw > 200:
                interpretations.append("✓ Excellent - NVLink detected (>200 GB/s)")
            elif ar_bw > 100:
                interpretations.append("✓ Good - Fast NVLink or PCIe Gen4 x16")
            elif ar_bw > 30:
                interpretations.append("⚠ Moderate - PCIe Gen3/4 interconnect")
            else:
                interpretations.append("⚠ Limited - Slow PCIe or bandwidth bottleneck")

        # Interpret Broadcast
        if "broadcast_peak_bandwidth_gbps" in metrics:
            bc_bw = metrics["broadcast_peak_bandwidth_gbps"]
            interpretations.append(f"Broadcast: {bc_bw:.1f} GB/s peak")

        # Interpret Point-to-Point
        if "point_to_point_peak_bandwidth_gbps" in metrics:
            p2p_bw = metrics["point_to_point_peak_bandwidth_gbps"]
            interpretations.append(f"Point-to-Point: {p2p_bw:.1f} GB/s")

            # P2P gives us the raw GPU-to-GPU bandwidth
            if p2p_bw > 300:
                interpretations.append("✓ NVLink Gen3+ detected (300+ GB/s)")
            elif p2p_bw > 150:
                interpretations.append("✓ NVLink Gen2 or high-speed interconnect")
            elif p2p_bw > 30:
                interpretations.append("⚠ PCIe interconnect (16-32 GB/s typical)")
            else:
                interpretations.append("⚠ Slow interconnect - check topology")

        return interpretations


def main():
    """
    Standalone test for NCCL profiler.
    """
    try:
        profiler = NCCLProfiler()
        print("Running NCCL Communication Benchmarks...\n")

        result = profiler.run(mode=TestMode.DEEP)

        print(f"Status: {result.status.value}")
        print("\nMetrics:")
        for key, value in result.metrics.items():
            if not key.endswith("_results"):  # Skip detailed results
                print(f"  {key}: {value}")

        print("\nInterpretation:")
        for line in result.interpretation:
            print(f"  {line}")

    except RuntimeError as e:
        print(f"NCCL tests skipped: {e}")


if __name__ == "__main__":
    main()
