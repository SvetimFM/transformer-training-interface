"""
Compute Profiler

Measures GPU compute throughput in TFLOPS (Trillion Floating Point Operations Per Second).

Tests:
1. Matrix Multiplication (GEMM) - The core operation in transformers
2. Multiple precisions - FP32, FP16, BF16, INT8
3. Various matrix sizes - Small to large

Features:
- Baseline capture for before/after comparison
- Sustained benchmarks to reach thermal steady-state
- Integration with health monitoring for throttle detection

Why this matters:
- TFLOPS directly correlates with inference speed
- Different precisions have different performance characteristics
- Knowing your GPU's compute capacity helps size models appropriately
"""

import torch
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from .metrics import (
    DiagnosticTest, TestResult, TestMode, TestStatus, MetricType,
    format_throughput, ProgressTracker
)
from .health_monitor import GPUHealthMonitor


class ComputeProfiler(DiagnosticTest):
    """
    Comprehensive compute performance profiling.

    Measures TFLOPS for various operations and data types.
    Focus on GEMM (General Matrix Multiply) which is the core of transformers.
    """

    def __init__(self, device: str = "cuda:0"):
        super().__init__(
            name="Compute Profiler",
            description="Measure GPU compute throughput (TFLOPS) for various precisions",
            metric_type=MetricType.COMPUTE
        )
        self.device = device

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.device_props = torch.cuda.get_device_properties(device)

        # Initialize health monitoring (optional - graceful failure)
        self.health_monitor = None
        try:
            self.health_monitor = GPUHealthMonitor()
        except Exception:
            # Health monitoring not available - continue without it
            pass

    def run(self, mode: TestMode = TestMode.QUICK) -> TestResult:
        """Run compute profiling tests"""
        try:
            metrics = {}

            if mode == TestMode.QUICK:
                # Quick: Just FP16 (most common for LLM inference)
                self._report_progress("Testing FP16 performance...", 25)
                fp16_tflops = self._benchmark_gemm(torch.float16, sizes=[(2048, 2048, 2048)])
                metrics["fp16_tflops"] = round(fp16_tflops, 2)

                self._report_progress("Testing FP32 performance...", 75)
                fp32_tflops = self._benchmark_gemm(torch.float32, sizes=[(2048, 2048, 2048)])
                metrics["fp32_tflops"] = round(fp32_tflops, 2)

            elif mode in [TestMode.DEEP, TestMode.BURN_IN]:
                # Comprehensive: All supported precisions with sustained load
                # IMPORTANT: Warmup GPU first to force P0 performance state
                # Need 30-45s to reach thermal steady-state and P0 clocks
                warmup_duration = 45 if mode == TestMode.BURN_IN else 30
                self._warmup_gpu(duration_seconds=warmup_duration)

                progress_step = 0
                # Use more iterations for sustained measurement (100 vs 20)
                deep_iterations = 100

                # FP32
                self._report_progress("Testing FP32 performance...", progress_step)
                fp32_tflops = self._benchmark_gemm(torch.float32, iterations=deep_iterations)
                metrics["fp32_tflops"] = round(fp32_tflops, 2)
                progress_step += 25

                # FP16
                self._report_progress("Testing FP16 performance...", progress_step)
                fp16_tflops = self._benchmark_gemm(torch.float16, iterations=deep_iterations)
                metrics["fp16_tflops"] = round(fp16_tflops, 2)
                progress_step += 25

                # BF16 (if supported - Ampere and newer)
                if self.device_props.major >= 8:
                    self._report_progress("Testing BF16 performance...", progress_step)
                    bf16_tflops = self._benchmark_gemm(torch.bfloat16, iterations=deep_iterations)
                    metrics["bf16_tflops"] = round(bf16_tflops, 2)
                progress_step += 25

                # TF32 (if supported - Ampere and newer)
                if self.device_props.major >= 8:
                    self._report_progress("Testing TF32 performance...", progress_step)
                    tf32_tflops = self._benchmark_gemm_tf32(iterations=deep_iterations)
                    metrics["tf32_tflops"] = round(tf32_tflops, 2)

            self._report_progress("Complete", 100)

            # Capture health snapshot after benchmarks
            if self.health_monitor:
                try:
                    gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
                    snapshot = self.health_monitor.get_snapshot(gpu_id)

                    # Add health metrics
                    if snapshot.clock_sm:
                        metrics["clock_mhz_during_test"] = snapshot.clock_sm
                    if snapshot.clock_sm_max:
                        metrics["clock_max_mhz"] = snapshot.clock_sm_max
                    if snapshot.temperature:
                        metrics["temperature_celsius"] = round(snapshot.temperature, 1)
                    if snapshot.power_draw:
                        metrics["power_draw_watts"] = round(snapshot.power_draw, 1)

                    metrics["throttled"] = snapshot.is_throttled
                    if snapshot.throttle_reason:
                        metrics["throttle_reason"] = snapshot.throttle_reason

                except Exception:
                    # Health snapshot failed - continue without it
                    pass

            # Add theoretical peak for comparison
            theoretical_peak = self._get_theoretical_peak()
            if theoretical_peak:
                metrics["theoretical_peak_fp16_tflops"] = theoretical_peak
                if "fp16_tflops" in metrics:
                    efficiency = (metrics["fp16_tflops"] / theoretical_peak) * 100
                    metrics["compute_efficiency_percent"] = round(efficiency, 1)

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
                interpretation="Compute profiling failed",
                recommendation="Check GPU availability",
                error_message=str(e)
            )

    def _warmup_gpu(self, duration_seconds: int = 15):
        """
        Warm up GPU to force it into P0 performance state.

        Runs sustained matrix multiplications to:
        - Ramp GPU from P2/P3 → P0 power state
        - Heat GPU to operating temperature
        - Stabilize clocks at maximum

        Args:
            duration_seconds: How long to run warmup (default 15s)
        """
        self._report_progress(f"Warming up GPU to P0 state ({duration_seconds}s)...", 0)

        # Use FP16 for warmup (fastest)
        A = torch.randn(4096, 4096, dtype=torch.float16, device=self.device)
        B = torch.randn(4096, 4096, dtype=torch.float16, device=self.device)
        C = torch.empty(4096, 4096, dtype=torch.float16, device=self.device)

        start_time = time.perf_counter()
        iteration = 0

        while (time.perf_counter() - start_time) < duration_seconds:
            torch.matmul(A, B, out=C)
            iteration += 1

            # Update progress every ~10 iterations
            if iteration % 10 == 0:
                elapsed = time.perf_counter() - start_time
                progress = min(int((elapsed / duration_seconds) * 100), 99)
                self._report_progress(f"Warming up GPU ({int(elapsed)}s/{duration_seconds}s)...", progress)

        torch.cuda.synchronize()

        # Clean up
        del A, B, C
        torch.cuda.empty_cache()

    def _benchmark_gemm(
        self,
        dtype: torch.dtype,
        sizes: List[Tuple[int, int, int]] = None,
        iterations: int = 20
    ) -> float:
        """
        Benchmark matrix multiplication for given dtype.

        Args:
            dtype: Data type to test (float32, float16, bfloat16)
            sizes: List of (M, N, K) matrix dimensions. If None, uses defaults.
            iterations: Number of iterations to run for measurement (default 20)

        Returns:
            TFLOPS achieved
        """
        if sizes is None:
            # Default sizes: Medium to large matrices typical in transformers
            sizes = [
                (1024, 1024, 1024),   # Small
                (2048, 2048, 2048),   # Medium
                (4096, 4096, 4096),   # Large
            ]

        tflops_results = []

        for m, n, k in sizes:
            try:
                # Create random matrices
                A = torch.randn(m, k, dtype=dtype, device=self.device)
                B = torch.randn(k, n, dtype=dtype, device=self.device)
                C = torch.empty(m, n, dtype=dtype, device=self.device)

                # Warm-up (important for accurate measurements)
                for _ in range(3):
                    torch.matmul(A, B, out=C)
                torch.cuda.synchronize()

                # Measure
                start = time.perf_counter()

                for _ in range(iterations):
                    torch.matmul(A, B, out=C)

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                # Calculate TFLOPS
                # FLOPS for matrix multiply: 2 * M * N * K (one multiply, one add per element)
                flops_per_matmul = 2 * m * n * k
                total_flops = flops_per_matmul * iterations
                tflops = (total_flops / elapsed) / 1e12

                tflops_results.append(tflops)

                # Clean up
                del A, B, C
                torch.cuda.empty_cache()

            except RuntimeError as e:
                # OOM or other error - skip this size
                continue

        # Return the best result (largest matrix that fit)
        return max(tflops_results) if tflops_results else 0.0

    def _benchmark_gemm_tf32(self, iterations: int = 20) -> float:
        """
        Benchmark with TF32 (TensorFloat-32) enabled.

        TF32 is Ampere's mixed precision format: FP32 input/output with lower precision compute.

        Args:
            iterations: Number of iterations to run for measurement
        """
        # Save current setting
        original_setting = torch.backends.cuda.matmul.allow_tf32

        try:
            # Enable TF32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Run benchmark with FP32 (will use TF32 internally)
            result = self._benchmark_gemm(torch.float32, sizes=[(2048, 2048, 2048)], iterations=iterations)

            return result

        finally:
            # Restore setting
            torch.backends.cuda.matmul.allow_tf32 = original_setting

    def _get_theoretical_peak(self) -> float:
        """
        Get theoretical peak TFLOPS for FP16/Tensor Core operations.

        Returns:
            Theoretical peak TFLOPS, or None if unknown
        """
        gpu_name = self.device_props.name.lower()

        # Known theoretical peaks for FP16 with Tensor Cores (TFLOPS)
        known_specs = {
            # RTX 30 series
            "3090": 142,  # with sparsity: 285
            "3080 ti": 136,
            "3080": 119,
            "3070 ti": 82,
            "3070": 82,
            "3060 ti": 80,
            "3060": 51,

            # RTX 40 series
            "4090": 330,  # with sparsity: 661
            "4080": 242,
            "4070 ti": 160,
            "4070": 116,
            "4060 ti": 88,
            "4060": 60,

            # Data center
            "a100": 312,  # 80GB version
            "a40": 150,
            "a30": 165,
            "a10": 125,
            "v100": 125,
            "h100": 989,  # With sparsity: 1979
        }

        for model, tflops in known_specs.items():
            if model in gpu_name:
                return tflops

        return None

    def _generate_interpretation(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable interpretation"""
        lines = []

        # FP16 (most important for LLM inference)
        if "fp16_tflops" in metrics:
            fp16 = metrics["fp16_tflops"]

            # Add clock speed info if available
            clock_info = ""
            if "clock_mhz_during_test" in metrics:
                clock = metrics["clock_mhz_during_test"]
                clock_max = metrics.get("clock_max_mhz")
                if clock_max and clock < clock_max * 0.95:
                    # Clocks are significantly below max
                    clock_info = f" @ {clock} MHz (max: {clock_max} MHz)"
                else:
                    clock_info = f" @ {clock} MHz"

            lines.append(f"FP16 Performance: {fp16} TFLOPS{clock_info}")

            if "theoretical_peak_fp16_tflops" in metrics:
                peak = metrics["theoretical_peak_fp16_tflops"]
                eff = metrics.get("compute_efficiency_percent", 0)
                lines.append(f"  Efficiency: {eff}% of theoretical peak ({peak} TFLOPS)")

                # Explain why efficiency is low if health data available
                if eff < 75 and "clock_mhz_during_test" in metrics:
                    clock = metrics["clock_mhz_during_test"]
                    clock_max = metrics.get("clock_max_mhz")

                    if clock_max and clock < clock_max * 0.8:
                        clock_loss = (1 - clock / clock_max) * 100
                        lines.append(f"  ⚠ GPU running at {clock_loss:.0f}% below max clocks")

                        if metrics.get("throttled"):
                            reason = metrics.get("throttle_reason", "unknown")
                            lines.append(f"  ⚠ Throttle detected: {reason}")
                        else:
                            lines.append(f"  ℹ GPU in power-save P-state (not P0 performance state)")

                    # Temperature info
                    if "temperature_celsius" in metrics:
                        temp = metrics["temperature_celsius"]
                        lines.append(f"  ℹ Temperature: {temp}°C")

        # Other precisions
        if "fp32_tflops" in metrics:
            fp32 = metrics["fp32_tflops"]
            lines.append(f"FP32 Performance: {fp32} TFLOPS")

        if "bf16_tflops" in metrics:
            bf16 = metrics["bf16_tflops"]
            lines.append(f"BF16 Performance: {bf16} TFLOPS")

        if "tf32_tflops" in metrics:
            tf32 = metrics["tf32_tflops"]
            lines.append(f"TF32 Performance: {tf32} TFLOPS")

        return "\n  ".join(lines)

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> str:
        """Generate recommendations based on compute performance"""
        recommendations = []

        fp16_tflops = metrics.get("fp16_tflops", 0)
        fp32_tflops = metrics.get("fp32_tflops", 0)

        # Check FP16 speedup
        if fp16_tflops > 0 and fp32_tflops > 0:
            speedup = fp16_tflops / fp32_tflops
            if speedup >= 2:
                recommendations.append(f"✓ Excellent FP16 acceleration ({speedup:.1f}x faster than FP32)")
                recommendations.append("  → Use FP16/mixed precision for inference")
            elif speedup >= 1.5:
                recommendations.append(f"✓ Good FP16 speedup ({speedup:.1f}x)")
            else:
                recommendations.append(f"⚠ Limited FP16 acceleration ({speedup:.1f}x) - check Tensor Core utilization")

        # Check compute efficiency with detailed diagnosis
        efficiency = metrics.get("compute_efficiency_percent", 0)
        if efficiency > 0:
            if efficiency >= 80:
                recommendations.append("✓ Excellent compute efficiency!")
            elif efficiency >= 60:
                recommendations.append("✓ Good compute efficiency")
            else:
                # Low efficiency - diagnose based on health metrics
                clock = metrics.get("clock_mhz_during_test")
                clock_max = metrics.get("clock_max_mhz")

                if clock and clock_max and clock < clock_max * 0.8:
                    recommendations.append(f"⚠ Low efficiency ({efficiency}%) - GPU clocks at {clock}/{clock_max} MHz")

                    if metrics.get("throttled"):
                        reason = metrics.get("throttle_reason", "unknown")
                        recommendations.append(f"  → Throttling detected: {reason}")
                        recommendations.append("  → Check cooling/thermal paste, increase fan speed, or reduce ambient temperature")
                    else:
                        recommendations.append("  → GPU in power-save mode - benchmarks too short to trigger P0 state")
                        recommendations.append("  → Run longer benchmarks or use nvidia-smi to lock clocks:")
                        recommendations.append(f"     sudo nvidia-smi -lgc {clock_max}")
                else:
                    # Low efficiency but clocks are fine
                    recommendations.append(f"⚠ Low efficiency ({efficiency}%) despite good clocks")
                    recommendations.append("  → May be memory-bound or experiencing other bottlenecks")

        # BF16 recommendations
        if "bf16_tflops" in metrics:
            bf16 = metrics["bf16_tflops"]
            if bf16 >= fp16_tflops * 0.9:
                recommendations.append("✓ BF16 supported - good for training (better numerical stability)")

        return "\n  ".join(recommendations) if recommendations else "Compute performance looks good!"

    # ========== Baseline Capture & Comparison ==========

    def save_baseline(self, metrics: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """
        Save current performance metrics as baseline for future comparison.

        Args:
            metrics: Test metrics to save
            filepath: Optional custom path. If None, uses default location.

        Returns:
            Path where baseline was saved
        """
        if filepath is None:
            baseline_dir = Path.home() / ".cache" / "transformer_diagnostics" / "baselines"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(baseline_dir / f"compute_baseline_{timestamp}.json")

        baseline = {
            'timestamp': datetime.now().isoformat(),
            'gpu_name': self.device_props.name,
            'cuda_version': torch.version.cuda or "Unknown",
            'pytorch_version': torch.__version__,
            'metrics': metrics
        }

        with open(filepath, 'w') as f:
            json.dump(baseline, f, indent=2)

        return filepath

    @staticmethod
    def load_baseline(filepath: str) -> Dict[str, Any]:
        """
        Load previously saved baseline.

        Args:
            filepath: Path to baseline JSON file

        Returns:
            Baseline data dictionary
        """
        with open(filepath, 'r') as f:
            return json.load(f)

    def compare_to_baseline(self, current_metrics: Dict[str, Any], baseline_path: str) -> Dict[str, Any]:
        """
        Compare current performance to saved baseline.

        Args:
            current_metrics: Current test metrics
            baseline_path: Path to baseline file

        Returns:
            Comparison dictionary with gains/losses
        """
        baseline = self.load_baseline(baseline_path)
        baseline_metrics = baseline['metrics']

        comparison = {
            'baseline_timestamp': baseline['timestamp'],
            'baseline_gpu': baseline['gpu_name'],
            'current_gpu': self.device_props.name,
        }

        # Compare FP16 performance
        if 'fp16_tflops' in current_metrics and 'fp16_tflops' in baseline_metrics:
            current_fp16 = current_metrics['fp16_tflops']
            baseline_fp16 = baseline_metrics['fp16_tflops']

            gain_tflops = current_fp16 - baseline_fp16
            gain_percent = (gain_tflops / baseline_fp16 * 100) if baseline_fp16 > 0 else 0

            comparison['fp16'] = {
                'baseline': baseline_fp16,
                'current': current_fp16,
                'gain_tflops': round(gain_tflops, 2),
                'gain_percent': round(gain_percent, 1)
            }

        # Compare efficiency
        if 'compute_efficiency_percent' in current_metrics and 'compute_efficiency_percent' in baseline_metrics:
            current_eff = current_metrics['compute_efficiency_percent']
            baseline_eff = baseline_metrics['compute_efficiency_percent']

            comparison['efficiency'] = {
                'baseline': baseline_eff,
                'current': current_eff,
                'improvement': round(current_eff - baseline_eff, 1)
            }

        # Compare FP32 performance
        if 'fp32_tflops' in current_metrics and 'fp32_tflops' in baseline_metrics:
            current_fp32 = current_metrics['fp32_tflops']
            baseline_fp32 = baseline_metrics['fp32_tflops']

            comparison['fp32'] = {
                'baseline': baseline_fp32,
                'current': current_fp32,
                'gain_tflops': round(current_fp32 - baseline_fp32, 2)
            }

        return comparison

    # ========== Enhanced Benchmarking ==========

    def _benchmark_gemm_sustained(
        self,
        dtype: torch.dtype,
        duration_seconds: int = 60,
        size: int = 4096
    ) -> float:
        """
        Run sustained benchmark to force GPU into P0 state.

        This runs continuously for the specified duration to ensure:
        1. GPU reaches thermal steady-state
        2. Clocks stabilize at boost frequency
        3. More accurate real-world performance measurement

        Args:
            dtype: Data type to test
            duration_seconds: How long to run (default 60s)
            size: Matrix dimension (default 4096 for Tensor Core opt)

        Returns:
            Average TFLOPS over the sustained period
        """
        # Create matrices
        A = torch.randn(size, size, dtype=dtype, device=self.device)
        B = torch.randn(size, size, dtype=dtype, device=self.device)

        # Warmup phase (10 seconds)
        print(f"  Warming up GPU (10 seconds)...")
        warmup_start = time.time()
        while time.time() - warmup_start < 10:
            C = torch.matmul(A, B)
        torch.cuda.synchronize()

        # Sustained measurement phase
        print(f"  Running sustained benchmark ({duration_seconds} seconds)...")
        tflops_samples = []
        start_time = time.time()
        end_time = start_time + duration_seconds

        iteration = 0
        while time.time() < end_time:
            # Measure this window
            iter_start = time.perf_counter()

            for _ in range(10):  # 10 matmuls per window
                C = torch.matmul(A, B)

            torch.cuda.synchronize()
            iter_elapsed = time.perf_counter() - iter_start

            # Calculate TFLOPS
            flops_per_matmul = 2 * size * size * size
            total_flops = flops_per_matmul * 10
            tflops = (total_flops / iter_elapsed) / 1e12
            tflops_samples.append(tflops)

            iteration += 1

        # Clean up
        del A, B, C
        torch.cuda.empty_cache()

        # Return average (sustained performance)
        avg_tflops = sum(tflops_samples) / len(tflops_samples) if tflops_samples else 0
        return avg_tflops
