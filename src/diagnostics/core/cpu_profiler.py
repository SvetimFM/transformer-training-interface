"""
CPU Profiler Module

Provides comprehensive CPU benchmarking for systems without GPUs.
Helps users understand:
- How much slower CPU is compared to GPU
- Whether their CPU can handle basic inference
- Thread scaling behavior
- BLAS library performance

Educational focus: Set realistic expectations for CPU inference.
"""

import torch
import time
import psutil
import platform
import multiprocessing
from typing import Dict, Any, List
from .metrics import DiagnosticTest, TestResult, TestMode, TestStatus, MetricType, format_bytes
from .backend_detector import BackendDetector, ComputeBackend


class CPUProfiler(DiagnosticTest):
    """
    CPU-specific performance profiler.

    Tests:
    - BLAS performance (matrix multiplication)
    - Thread scaling (1 to all cores)
    - Memory bandwidth
    - Inference performance (if model loader available)
    """

    def __init__(self):
        super().__init__(
            name="CPU Profiler",
            description="Measure CPU compute performance and thread scaling",
            metric_type=MetricType.COMPUTE
        )

        backend = BackendDetector.detect()
        if backend.backend != ComputeBackend.CPU:
            raise RuntimeError(f"CPUProfiler requires CPU-only mode, found {backend.backend}")

        self.cpu_count_physical = psutil.cpu_count(logical=False) or 1
        self.cpu_count_logical = psutil.cpu_count(logical=True) or 1
        self.cpu_name = self._get_cpu_name()

    def _get_cpu_name(self) -> str:
        """Get CPU model name"""
        try:
            if platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            return line.split(':')[1].strip()
            elif platform.system() == "Darwin":  # macOS
                import subprocess
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    return result.stdout.strip()
        except:
            pass

        return f"{platform.processor()} ({self.cpu_count_physical} cores)"

    def run(self, mode: TestMode = TestMode.QUICK) -> TestResult:
        """
        Run CPU profiling tests.

        Args:
            mode: Test mode (quick, deep, or burn-in)

        Returns:
            TestResult with CPU performance metrics
        """
        try:
            metrics = {}

            # CPU info
            metrics['cpu_name'] = self.cpu_name
            metrics['physical_cores'] = self.cpu_count_physical
            metrics['logical_cores'] = self.cpu_count_logical
            metrics['total_ram_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)

            # Test 1: BLAS Performance
            self._report_progress("Testing BLAS performance...", 20)
            blas_metrics = self._test_blas_performance(mode)
            metrics.update(blas_metrics)

            # Test 2: Thread Scaling
            self._report_progress("Testing thread scaling...", 50)
            thread_metrics = self._test_thread_scaling(mode)
            metrics.update(thread_metrics)

            # Test 3: Memory Bandwidth
            self._report_progress("Testing memory bandwidth...", 70)
            memory_metrics = self._test_memory_bandwidth()
            metrics.update(memory_metrics)

            # Test 4: Inference Test (Deep mode only)
            if mode in [TestMode.DEEP, TestMode.BURN_IN]:
                try:
                    self._report_progress("Testing CPU inference (this is slow)...", 85)
                    inference_metrics = self._test_cpu_inference()
                    metrics.update(inference_metrics)
                except Exception as e:
                    metrics['inference_error'] = str(e)

            self._report_progress("Complete", 100)

            # Generate interpretation and recommendations
            interpretation = self._generate_interpretation(metrics)
            recommendation = self._generate_recommendations(metrics)

            return self._create_result(
                status=TestStatus.PASSED,
                metrics=metrics,
                interpretation=interpretation,
                recommendation=recommendation
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._create_result(
                status=TestStatus.FAILED,
                metrics={},
                interpretation="CPU profiling failed",
                recommendation="Check CPU and system resources",
                error_message=str(e)
            )

    def _test_blas_performance(self, mode: TestMode) -> Dict[str, Any]:
        """
        Test BLAS (Basic Linear Algebra Subprograms) performance.

        This measures matrix multiplication speed using MKL, OpenBLAS,
        or whatever BLAS library PyTorch is linked against.
        """
        metrics = {}

        # Detect BLAS library
        try:
            import numpy as np
            blas_info = np.__config__.show()
            # This is printed to stdout, we'll just note that it was checked
            metrics['blas_library'] = "Detected (check via numpy.__config__.show())"
        except:
            metrics['blas_library'] = "Unknown"

        # Test different matrix sizes
        sizes = [512, 1024, 2048] if mode == TestMode.QUICK else [512, 1024, 2048, 4096]

        for size in sizes:
            # Create random matrices
            A = torch.randn(size, size, dtype=torch.float32)
            B = torch.randn(size, size, dtype=torch.float32)

            # Warm-up
            _ = torch.mm(A, B)

            # Measure
            iterations = 10 if mode == TestMode.QUICK else 20
            start = time.perf_counter()

            for _ in range(iterations):
                C = torch.mm(A, B)

            elapsed = time.perf_counter() - start

            # Calculate GFLOPS
            # Matrix multiplication: 2 * N^3 FLOPs
            flops = iterations * 2 * (size ** 3)
            gflops = flops / elapsed / 1e9

            metrics[f'blas_gflops_{size}x{size}'] = round(gflops, 2)

        # Calculate peak GFLOPS from largest matrix
        largest_size = max(sizes)
        metrics['peak_cpu_gflops'] = metrics[f'blas_gflops_{largest_size}x{largest_size}']

        return metrics

    def _test_thread_scaling(self, mode: TestMode) -> Dict[str, Any]:
        """
        Test how performance scales with number of threads.

        This is educational - shows diminishing returns with more threads.
        """
        metrics = {}

        # Test with different thread counts
        if mode == TestMode.QUICK:
            thread_counts = [1, self.cpu_count_physical]
        else:
            thread_counts = [1, self.cpu_count_physical // 2, self.cpu_count_physical, self.cpu_count_logical]
            # Remove duplicates and zeros
            thread_counts = sorted(list(set([t for t in thread_counts if t > 0])))

        size = 2048
        A = torch.randn(size, size, dtype=torch.float32)
        B = torch.randn(size, size, dtype=torch.float32)

        for threads in thread_counts:
            torch.set_num_threads(threads)

            # Warm-up
            _ = torch.mm(A, B)

            # Measure
            iterations = 10
            start = time.perf_counter()

            for _ in range(iterations):
                C = torch.mm(A, B)

            elapsed = time.perf_counter() - start

            # Calculate speedup vs 1 thread
            gflops = (iterations * 2 * (size ** 3)) / elapsed / 1e9
            metrics[f'threads_{threads}_gflops'] = round(gflops, 2)

        # Calculate scaling efficiency
        if f'threads_1_gflops' in metrics:
            baseline = metrics['threads_1_gflops']
            for threads in thread_counts:
                if threads > 1:
                    actual_speedup = metrics[f'threads_{threads}_gflops'] / baseline
                    ideal_speedup = threads
                    efficiency = (actual_speedup / ideal_speedup) * 100
                    metrics[f'threads_{threads}_efficiency_pct'] = round(efficiency, 1)

        # Reset to default (use all cores)
        torch.set_num_threads(self.cpu_count_logical)

        return metrics

    def _test_memory_bandwidth(self) -> Dict[str, Any]:
        """
        Test CPU RAM bandwidth.

        Measures how fast data can be read from and written to RAM.
        """
        metrics = {}

        # Test with large arrays to avoid cache effects
        size_mb = 256  # 256 MB
        elements = (size_mb * 1024 * 1024) // 4  # float32 is 4 bytes

        # Create large tensor
        data = torch.randn(elements, dtype=torch.float32)

        # Test 1: Read bandwidth (sum operation reads all data)
        iterations = 10
        start = time.perf_counter()

        for _ in range(iterations):
            _ = data.sum()

        elapsed = time.perf_counter() - start

        # Calculate bandwidth (GB/s)
        bytes_read = iterations * size_mb * 1024 * 1024
        read_bandwidth = (bytes_read / elapsed) / (1024**3)
        metrics['cpu_memory_read_bandwidth_gbps'] = round(read_bandwidth, 2)

        # Test 2: Write bandwidth (fill operation writes all data)
        start = time.perf_counter()

        for _ in range(iterations):
            data.fill_(0.0)

        elapsed = time.perf_counter() - start

        bytes_written = iterations * size_mb * 1024 * 1024
        write_bandwidth = (bytes_written / elapsed) / (1024**3)
        metrics['cpu_memory_write_bandwidth_gbps'] = round(write_bandwidth, 2)

        # Test 3: Copy bandwidth
        data2 = torch.empty_like(data)
        start = time.perf_counter()

        for _ in range(iterations):
            data2.copy_(data)

        elapsed = time.perf_counter() - start

        bytes_copied = iterations * size_mb * 1024 * 1024 * 2  # read + write
        copy_bandwidth = (bytes_copied / elapsed) / (1024**3)
        metrics['cpu_memory_copy_bandwidth_gbps'] = round(copy_bandwidth, 2)

        return metrics

    def _test_cpu_inference(self) -> Dict[str, Any]:
        """
        Test actual CPU inference with a small model.

        WARNING: This is very slow! Only runs in deep mode.
        """
        metrics = {}

        try:
            from .model_loader import ModelLoader

            loader = ModelLoader()

            # Always use smallest model for CPU
            model_name = "Qwen2.5-0.5B"  # 0.5B is smallest

            # Check if cached
            if not loader.is_model_cached(model_name):
                metrics['cpu_inference_note'] = "Model not cached, skipping CPU inference test"
                return metrics

            # Load model on CPU
            model, tokenizer = loader.load_model_and_tokenizer(
                model_name,
                device="cpu",
                dtype=torch.float32  # CPU doesn't support FP16 well
            )

            # Warm-up
            prompt = "Hello, how are you?"
            inputs = tokenizer(prompt, return_tensors="pt")
            _ = model.generate(**inputs, max_new_tokens=5)

            # Measure throughput (very basic, small batch)
            num_tokens = 20
            start = time.perf_counter()

            outputs = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)

            elapsed = time.perf_counter() - start

            tokens_per_sec = num_tokens / elapsed
            metrics['cpu_inference_tokens_per_sec'] = round(tokens_per_sec, 2)
            metrics['cpu_inference_model'] = model_name

            # Cleanup
            del model, tokenizer

        except Exception as e:
            metrics['cpu_inference_error'] = str(e)

        return metrics

    def _generate_interpretation(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable interpretation"""
        lines = []

        cpu_name = metrics.get('cpu_name', 'Unknown CPU')
        physical_cores = metrics.get('physical_cores', 0)
        logical_cores = metrics.get('logical_cores', 0)

        lines.append(f"CPU: {cpu_name}")
        lines.append(f"Cores: {physical_cores} physical, {logical_cores} logical")

        # BLAS performance
        peak_gflops = metrics.get('peak_cpu_gflops', 0)
        if peak_gflops > 0:
            lines.append(f"Peak Performance: {peak_gflops:.1f} GFLOPS (FP32)")

            # Compare to typical GPU
            lines.append(f"  → ~300-1000x slower than modern GPU")

        # Thread scaling
        threads_1_gflops = metrics.get('threads_1_gflops', 0)
        threads_max_key = max([k for k in metrics.keys() if k.startswith('threads_') and k.endswith('_gflops')],
                               key=lambda x: int(x.split('_')[1]), default=None)

        if threads_max_key:
            threads_max_gflops = metrics.get(threads_max_key, 0)
            max_threads = int(threads_max_key.split('_')[1])
            speedup = threads_max_gflops / threads_1_gflops if threads_1_gflops > 0 else 0
            lines.append(f"Thread Scaling: {speedup:.1f}x with {max_threads} threads")

        # Memory bandwidth
        read_bw = metrics.get('cpu_memory_read_bandwidth_gbps', 0)
        if read_bw > 0:
            lines.append(f"Memory Bandwidth: {read_bw:.1f} GB/s (read)")

        # CPU inference (if available)
        cpu_tok_per_sec = metrics.get('cpu_inference_tokens_per_sec', 0)
        if cpu_tok_per_sec > 0:
            lines.append(f"CPU Inference: {cpu_tok_per_sec:.1f} tokens/sec ({metrics.get('cpu_inference_model', 'Unknown')})")
            lines.append("  → ~10-50x slower than GPU for same model")

        return "\n  ".join(lines)

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> str:
        """Generate actionable recommendations"""
        recommendations = []

        peak_gflops = metrics.get('peak_cpu_gflops', 0)
        cpu_tok_per_sec = metrics.get('cpu_inference_tokens_per_sec', 0)

        # CPU performance assessment
        if cpu_tok_per_sec > 0:
            if cpu_tok_per_sec >= 5:
                recommendations.append("⚠ CPU inference is possible but VERY slow")
                recommendations.append("  → Expect 30+ seconds for short responses")
            else:
                recommendations.append("❌ CPU inference is impractically slow")
                recommendations.append("  → 1+ minute for simple responses")

        recommendations.append("")
        recommendations.append("💡 Recommendations:")
        recommendations.append("  1. Get a GPU - even entry-level GPUs are 10-50x faster")
        recommendations.append("  2. Consider cloud GPU instances (Vast.ai, RunPod, etc.)")
        recommendations.append("  3. For CPU-only: Use tiny models (<1B params) and quantization")
        recommendations.append("  4. Alternatively: Use API services (OpenAI, Anthropic, etc.)")

        # Thread scaling assessment
        efficiency_key = max([k for k in metrics.keys() if '_efficiency_pct' in k], default=None)
        if efficiency_key:
            efficiency = metrics.get(efficiency_key, 0)
            if efficiency > 70:
                recommendations.append("")
                recommendations.append(f"✓ Good thread scaling ({efficiency:.1f}% efficient)")
            elif efficiency > 40:
                recommendations.append("")
                recommendations.append(f"⚠ Moderate thread scaling ({efficiency:.1f}% efficient)")
                recommendations.append("  → Check for SMT/Hyper-Threading enabled")
            else:
                recommendations.append("")
                recommendations.append(f"❌ Poor thread scaling ({efficiency:.1f}% efficient)")
                recommendations.append("  → May have NUMA issues or poor BLAS library")

        return "\n  ".join(recommendations) if recommendations else "Consider getting a GPU for practical LLM inference."


if __name__ == "__main__":
    """Test CPU profiler"""
    print("Testing CPU Profiler...")
    print()

    profiler = CPUProfiler()
    result = profiler.run(mode=TestMode.QUICK)

    result.print_summary(verbose=True)
