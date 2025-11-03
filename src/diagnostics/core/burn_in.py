"""
Burn-in Testing Module

Long-running stability tests to validate production readiness.
Designed for homelab users who want to ensure their hardware is stable
before committing to expensive multi-day training runs.

Tests:
- Continuous inference (hours to days)
- Thermal stability under sustained load
- Memory leak detection
- Performance degradation over time
- Error rate tracking

Educational focus: "Will this hardware survive a 48-hour training run?"
"""

import torch
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from .metrics import DiagnosticTest, TestResult, TestMode, TestStatus, MetricType
from .health_monitor import GPUHealthMonitor, GPUHealthSnapshot
from .backend_detector import BackendDetector, ComputeBackend


@dataclass
class BurnInResult:
    """Results from burn-in testing"""
    duration_seconds: float
    total_iterations: int
    successful_iterations: int
    failed_iterations: int
    errors: List[Dict[str, Any]]

    # Performance metrics
    avg_tokens_per_sec: float
    tokens_per_sec_std: float  # Standard deviation (measures stability)

    # Thermal metrics
    peak_temperature: float
    avg_temperature: float
    throttle_events: int

    # Memory metrics
    peak_memory_gb: float
    avg_memory_gb: float
    memory_leak_detected: bool

    # Overall assessment
    passed: bool
    stability_score: float  # 0-100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'duration_hours': round(self.duration_seconds / 3600, 2),
            'total_iterations': self.total_iterations,
            'success_rate_pct': round((self.successful_iterations / self.total_iterations * 100), 2) if self.total_iterations > 0 else 0,
            'avg_tokens_per_sec': round(self.avg_tokens_per_sec, 2),
            'performance_stability': round(100 - min(self.tokens_per_sec_std / self.avg_tokens_per_sec * 100, 100), 1) if self.avg_tokens_per_sec > 0 else 0,
            'peak_temperature': round(self.peak_temperature, 1),
            'avg_temperature': round(self.avg_temperature, 1),
            'throttle_events': self.throttle_events,
            'peak_memory_gb': round(self.peak_memory_gb, 2),
            'memory_leak_detected': self.memory_leak_detected,
            'stability_score': round(self.stability_score, 1),
            'passed': self.passed,
            'error_count': len(self.errors)
        }


class BurnInTester(DiagnosticTest):
    """
    Comprehensive burn-in testing for production readiness validation.

    Usage:
        tester = BurnInTester()
        result = tester.run_burn_in(duration_hours=24, workload='inference')

    Or via CLI:
        python -m src.diagnostics.cli --mode burn-in --duration 24h
    """

    def __init__(self, device: Optional[str] = None):
        super().__init__(
            name="Burn-in Tester",
            description="Long-running stability and stress testing",
            metric_type=MetricType.STABILITY
        )

        backend = BackendDetector.detect()
        self.backend = backend.backend

        if self.backend == ComputeBackend.CPU:
            self.device = "cpu"
        else:
            self.device = device or BackendDetector.get_recommended_device()

        # Health monitoring
        try:
            self.health_monitor = GPUHealthMonitor()
        except Exception as e:
            print(f"Warning: Health monitoring not available: {e}")
            self.health_monitor = None

        # State tracking
        self.is_running = False
        self.should_stop = False

    def run(self, mode: TestMode = TestMode.BURN_IN, duration_hours: float = 1.0) -> TestResult:
        """
        Run burn-in test.

        Args:
            mode: Should be BURN_IN
            duration_hours: How long to run (default 1 hour for testing)

        Returns:
            TestResult with burn-in results
        """
        if mode != TestMode.BURN_IN:
            return self._create_result(
                status=TestStatus.SKIPPED,
                metrics={},
                interpretation="Burn-in test only runs in BURN_IN mode",
                recommendation="Use --mode burn-in to enable"
            )

        try:
            result = self.run_burn_in(duration_hours=duration_hours)

            status = TestStatus.PASSED if result.passed else TestStatus.FAILED

            return self._create_result(
                status=status,
                metrics=result.to_dict(),
                interpretation=self._generate_interpretation(result),
                recommendation=self._generate_recommendations(result)
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._create_result(
                status=TestStatus.FAILED,
                metrics={},
                interpretation="Burn-in test failed",
                recommendation="Check hardware and logs",
                error_message=str(e)
            )

    def run_burn_in(
        self,
        duration_hours: float = 24.0,
        workload: str = 'inference',
        progress_callback: Optional[Callable] = None
    ) -> BurnInResult:
        """
        Run comprehensive burn-in test.

        Args:
            duration_hours: How long to run
            workload: Type of workload ('inference', 'compute', 'memory')
            progress_callback: Optional callback(message, progress_pct)

        Returns:
            BurnInResult with detailed results
        """
        self.is_running = True
        self.should_stop = False

        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)

        # Results tracking
        iterations = 0
        successful = 0
        failed = 0
        errors = []
        performance_samples = []
        memory_samples = []
        health_snapshots = []

        # Start health monitoring
        if self.health_monitor:
            self.health_monitor.start_continuous_monitoring(interval_seconds=60)

        print(f"Starting {duration_hours:.1f} hour burn-in test...")
        print(f"Workload: {workload}")
        print(f"Expected end time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        try:
            while time.time() < end_time and not self.should_stop:
                iterations += 1
                iteration_start = time.time()

                try:
                    # Run workload
                    if workload == 'inference':
                        perf = self._run_inference_iteration()
                    elif workload == 'compute':
                        perf = self._run_compute_iteration()
                    elif workload == 'memory':
                        perf = self._run_memory_iteration()
                    else:
                        raise ValueError(f"Unknown workload: {workload}")

                    successful += 1
                    performance_samples.append(perf)

                    # Track memory
                    if self.backend != ComputeBackend.CPU:
                        mem_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                        memory_samples.append(mem_used)

                    # Health snapshot every minute
                    if self.health_monitor and iterations % 10 == 0:
                        snapshot = self.health_monitor.get_snapshot(0)
                        health_snapshots.append(snapshot)

                except Exception as e:
                    failed += 1
                    errors.append({
                        'iteration': iterations,
                        'time': time.time() - start_time,
                        'error': str(e)
                    })

                # Progress reporting
                elapsed = time.time() - start_time
                progress_pct = (elapsed / (duration_hours * 3600)) * 100

                if progress_callback:
                    progress_callback(
                        f"Iteration {iterations} ({successful} success, {failed} fail)",
                        progress_pct
                    )
                elif iterations % 100 == 0:
                    print(f"[{progress_pct:.1f}%] Iteration {iterations}: {successful} success, {failed} fail")

                # Brief sleep to avoid excessive CPU usage in monitoring
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nBurn-in interrupted by user")
        finally:
            self.is_running = False

        # Calculate results
        duration = time.time() - start_time

        # Performance metrics
        avg_perf = sum(performance_samples) / len(performance_samples) if performance_samples else 0
        import statistics
        perf_std = statistics.stdev(performance_samples) if len(performance_samples) > 1 else 0

        # Thermal metrics
        temps = [s.temperature for s in health_snapshots if s.temperature is not None]
        peak_temp = max(temps) if temps else 0
        avg_temp = sum(temps) / len(temps) if temps else 0
        throttle_count = sum(1 for s in health_snapshots if s.is_throttled)

        # Memory metrics
        peak_mem = max(memory_samples) if memory_samples else 0
        avg_mem = sum(memory_samples) / len(memory_samples) if memory_samples else 0

        # Memory leak detection (memory should be stable, not growing)
        memory_leak = False
        if len(memory_samples) > 100:
            # Compare first 10% vs last 10%
            early_avg = sum(memory_samples[:len(memory_samples)//10]) / (len(memory_samples)//10)
            late_avg = sum(memory_samples[-len(memory_samples)//10:]) / (len(memory_samples)//10)
            if late_avg > early_avg * 1.1:  # 10% growth = potential leak
                memory_leak = True

        # Calculate stability score
        stability_score = self._calculate_stability_score(
            success_rate=successful / iterations if iterations > 0 else 0,
            perf_stability=100 - min(perf_std / avg_perf * 100, 100) if avg_perf > 0 else 0,
            thermal_ok=peak_temp < 85,
            throttle_events=throttle_count,
            memory_leak=memory_leak
        )

        passed = (
            successful / iterations >= 0.99 and  # 99% success rate
            stability_score >= 80 and            # 80% stability
            not memory_leak                       # No memory leaks
        ) if iterations > 0 else False

        return BurnInResult(
            duration_seconds=duration,
            total_iterations=iterations,
            successful_iterations=successful,
            failed_iterations=failed,
            errors=errors,
            avg_tokens_per_sec=avg_perf,
            tokens_per_sec_std=perf_std,
            peak_temperature=peak_temp,
            avg_temperature=avg_temp,
            throttle_events=throttle_count,
            peak_memory_gb=peak_mem,
            avg_memory_gb=avg_mem,
            memory_leak_detected=memory_leak,
            passed=passed,
            stability_score=stability_score
        )

    def _run_inference_iteration(self) -> float:
        """
        Run single inference iteration.

        Returns tokens/sec for this iteration.
        """
        # Use smallest model for burn-in (fastest iterations)
        try:
            from .model_loader import ModelLoader

            # Cache model loader
            if not hasattr(self, '_model') or not hasattr(self, '_tokenizer'):
                loader = ModelLoader()
                model_name = "Qwen2.5-0.5B"  # Smallest model

                if not loader.is_model_cached(model_name):
                    # Download if needed (only happens once)
                    loader.download_model(model_name)

                self._model, self._tokenizer = loader.load_model_and_tokenizer(
                    model_name,
                    device=self.device,
                    dtype=torch.float16 if self.backend != ComputeBackend.CPU else torch.float32
                )

            # Run inference
            prompt = "Test prompt for burn-in testing"
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

            start = time.perf_counter()

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id
                )

            elapsed = time.perf_counter() - start
            tokens_per_sec = 20 / elapsed

            return tokens_per_sec

        except Exception as e:
            # Fallback to simple compute if model fails
            return self._run_compute_iteration()

    def _run_compute_iteration(self) -> float:
        """
        Run compute-heavy iteration (matrix multiplication).

        Returns GFLOPS achieved.
        """
        size = 2048
        A = torch.randn(size, size, dtype=torch.float32, device=self.device)
        B = torch.randn(size, size, dtype=torch.float32, device=self.device)

        start = time.perf_counter()

        for _ in range(10):
            C = torch.mm(A, B)

        if self.device != "cpu":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        # Calculate GFLOPS
        flops = 10 * 2 * (size ** 3)
        gflops = flops / elapsed / 1e9

        del A, B, C
        return gflops

    def _run_memory_iteration(self) -> float:
        """
        Run memory-heavy iteration (bandwidth test).

        Returns GB/s achieved.
        """
        size_mb = 128
        elements = (size_mb * 1024 * 1024) // 4

        data = torch.randn(elements, dtype=torch.float32, device=self.device)

        start = time.perf_counter()

        for _ in range(10):
            _ = data.sum()

        if self.device != "cpu":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        bytes_read = 10 * size_mb * 1024 * 1024
        bandwidth_gbps = (bytes_read / elapsed) / (1024**3)

        del data
        return bandwidth_gbps

    def _calculate_stability_score(
        self,
        success_rate: float,
        perf_stability: float,
        thermal_ok: bool,
        throttle_events: int,
        memory_leak: bool
    ) -> float:
        """Calculate overall stability score (0-100)"""
        score = 0.0

        # Success rate: 40 points
        score += success_rate * 40

        # Performance stability: 30 points
        score += (perf_stability / 100) * 30

        # Thermal: 15 points
        if thermal_ok:
            score += 15
        score -= min(throttle_events, 10)  # -1 point per throttle event (max -10)

        # Memory: 15 points
        if not memory_leak:
            score += 15

        return max(0, min(100, score))

    def _generate_interpretation(self, result: BurnInResult) -> str:
        """Generate human-readable interpretation"""
        lines = []

        hours = result.duration_seconds / 3600
        lines.append(f"Burn-in Duration: {hours:.1f} hours")
        lines.append(f"Iterations: {result.total_iterations:,} ({result.successful_iterations:,} successful)")

        success_rate = (result.successful_iterations / result.total_iterations * 100) if result.total_iterations > 0 else 0
        lines.append(f"Success Rate: {success_rate:.2f}%")

        # Performance
        lines.append(f"Avg Performance: {result.avg_tokens_per_sec:.1f} tokens/sec")
        perf_var = (result.tokens_per_sec_std / result.avg_tokens_per_sec * 100) if result.avg_tokens_per_sec > 0 else 0
        lines.append(f"Performance Variation: ±{perf_var:.1f}%")

        # Thermal
        lines.append(f"Temperature: {result.avg_temperature:.1f}°C avg, {result.peak_temperature:.1f}°C peak")
        if result.throttle_events > 0:
            lines.append(f"⚠ Throttling Events: {result.throttle_events}")

        # Memory
        if result.memory_leak_detected:
            lines.append("⚠ Memory Leak Detected!")

        # Overall
        lines.append(f"Stability Score: {result.stability_score:.1f}/100")
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        lines.append(f"Status: {status}")

        return "\n  ".join(lines)

    def _generate_recommendations(self, result: BurnInResult) -> str:
        """Generate actionable recommendations"""
        recommendations = []

        if result.passed:
            recommendations.append("✅ System is stable and production-ready!")
            recommendations.append("  → Safe for multi-day training runs")
        else:
            recommendations.append("❌ System failed burn-in test")

            success_rate = (result.successful_iterations / result.total_iterations * 100) if result.total_iterations > 0 else 0

            if success_rate < 99:
                recommendations.append(f"  ⚠ Success rate ({success_rate:.1f}%) too low")
                recommendations.append("    → Check for hardware errors in dmesg/logs")

            if result.throttle_events > 5:
                recommendations.append(f"  ⚠ Frequent throttling ({result.throttle_events} events)")
                recommendations.append("    → Improve cooling or reduce power limit")

            if result.memory_leak_detected:
                recommendations.append("  ⚠ Memory leak detected")
                recommendations.append("    → This may be a PyTorch or driver issue")

            if result.peak_temperature > 85:
                recommendations.append(f"  ⚠ High temperature ({result.peak_temperature:.1f}°C)")
                recommendations.append("    → Add cooling or undervolt GPU")

        return "\n  ".join(recommendations) if recommendations else "System performance acceptable."

    def stop(self):
        """Stop burn-in test gracefully"""
        self.should_stop = True


if __name__ == "__main__":
    """Test burn-in with short duration"""
    print("Testing burn-in (30 second test)...")
    print()

    tester = BurnInTester()
    result = tester.run_burn_in(duration_hours=30/3600, workload='compute')  # 30 seconds

    print()
    print("=" * 80)
    print("BURN-IN RESULTS")
    print("=" * 80)
    for key, value in result.to_dict().items():
        print(f"{key}: {value}")
    print("=" * 80)
