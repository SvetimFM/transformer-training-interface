"""
CLI Interface for NCCL Diagnostics & Performance Testing

Usage:
    python -m diagnostics.cli --mode quick
    python -m diagnostics.cli --mode deep --output report.html
    python -m diagnostics.cli --mode burn-in --duration 24h

This tool helps you understand your hardware's capabilities for LLM training and inference.
"""

import argparse
import sys
import os
from typing import List
from datetime import datetime

# Add src to path if running as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from diagnostics.core.metrics import TestMode, TestSuite, TestResult, TestStatus
from diagnostics.core.hardware_detector import HardwareDetector
from diagnostics.core.memory_profiler import MemoryProfiler
from diagnostics.core.compute_profiler import ComputeProfiler
from diagnostics.core.inference_profiler import InferenceProfiler
from diagnostics.core.backend_detector import BackendDetector, ComputeBackend
from diagnostics.core.cpu_profiler import CPUProfiler
from diagnostics.core.health_monitor import GPUHealthMonitor
from diagnostics.core.burn_in import BurnInTester


class DiagnosticsCLI:
    """Command-line interface for running diagnostic tests"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.suite = TestSuite(
            name="Hardware Diagnostics",
            description="Comprehensive hardware capability testing"
        )
        self.burn_in_duration = 1.0  # Default duration for burn-in (hours)

    def run_all_tests(self, mode: TestMode = TestMode.QUICK):
        """Run all available diagnostic tests"""
        # Detect backend
        backend_info = BackendDetector.detect()
        backend = backend_info.backend

        print()
        print("=" * 80)
        print("🔬 HARDWARE DIAGNOSTICS & PERFORMANCE TESTING")
        print("=" * 80)
        print(f"Backend: {backend.value.upper()} ({backend_info.device_count} device(s))")
        print(f"Mode: {mode.value.upper()}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()

        # Test 1: Hardware Detection
        print("=" * 80)
        print("1. HARDWARE DETECTION")
        print("=" * 80)
        result = self._run_test(HardwareDetector(), mode)
        self.suite.add_result(result)
        result.print_summary(verbose=False)
        print()

        # CPU-only mode
        if backend == ComputeBackend.CPU:
            print("=" * 80)
            print("2. CPU PROFILING")
            print("=" * 80)
            print("Running on CPU - GPU-specific tests will be skipped.")
            print("Note: CPU is typically 10-100x slower than GPU for deep learning.")
            print()
            result = self._run_test(CPUProfiler(), mode)
            self.suite.add_result(result)
            result.print_summary(verbose=False)
            print()
            self._print_summary()
            return

        # Check if GPU is available (redundant check for safety)
        import torch
        if not torch.cuda.is_available():
            print("⚠️  No GPU detected. Skipping GPU-specific tests.")
            self._print_summary()
            return

        # Initialize health monitoring for GPU tests
        health_monitor = None
        try:
            health_monitor = GPUHealthMonitor()
            print("✓ GPU health monitoring enabled")
            print()
        except Exception as e:
            print(f"⚠️  Health monitoring unavailable: {e}")
            print()

        # Test 2: Memory Profiling
        print("=" * 80)
        print("2. MEMORY PROFILING")
        print("=" * 80)
        result = self._run_test(MemoryProfiler(), mode)
        self.suite.add_result(result)
        result.print_summary(verbose=False)

        # Show health snapshot
        if health_monitor:
            snapshot = health_monitor.get_snapshot(0)
            print(f"  📊 {snapshot}")
        print()

        # Test 3: Compute Profiling
        print("=" * 80)
        print("3. COMPUTE PROFILING (TFLOPS)")
        print("=" * 80)
        result = self._run_test(ComputeProfiler(), mode)
        self.suite.add_result(result)
        compute_result = result  # Save for comparison later
        result.print_summary(verbose=False)

        # Show health snapshot
        if health_monitor:
            snapshot = health_monitor.get_snapshot(0)
            print(f"  📊 {snapshot}")
        print()

        # Test 4: Real Inference Profiling (Deep mode only)
        if mode in [TestMode.DEEP, TestMode.BURN_IN]:
            print("=" * 80)
            print("4. REAL INFERENCE PROFILING")
            print("=" * 80)
            print("This test downloads a small LLM model (~1-6 GB) to measure real performance.")
            print("Models are cached locally for future use.")
            print()

            try:
                profiler = InferenceProfiler()
                result = self._run_test_with_mode(profiler, mode)
                self.suite.add_result(result)
                result.print_summary(verbose=False)

                # Show comparison to synthetic if both tests passed
                if result.status == TestStatus.PASSED and compute_result.status == TestStatus.PASSED:
                    comparison = profiler.compare_to_synthetic(
                        result.metrics,
                        compute_result.metrics
                    )
                    if comparison:
                        print()
                        print("  📊 Synthetic vs Real Comparison:")
                        print(f"    Synthetic GEMM: {comparison.get('synthetic_gemm_tflops', 0)} TFLOPS")
                        print(f"    Real Inference: {comparison.get('real_inference_tflops', 0)} TFLOPS")
                        print(f"    Efficiency: {comparison.get('efficiency_percent', 0)}%")
                        print(f"    → {comparison.get('explanation', '')}")

            except Exception as e:
                print(f"  ⚠️  Inference profiling skipped: {e}")
                print("     (transformers and huggingface_hub required)")

            print()

        # Test 5: Burn-in Testing (Burn-in mode only)
        if mode == TestMode.BURN_IN:
            print("=" * 80)
            print("5. BURN-IN TESTING")
            print("=" * 80)
            print(f"Running {self.burn_in_duration:.1f} hour stability test...")
            print("This test validates hardware stability under sustained load.")
            print("Press Ctrl+C to stop early.")
            print()

            try:
                tester = BurnInTester()
                # Pass duration directly to the burn-in tester
                result = tester.run(mode=mode, duration_hours=self.burn_in_duration)
                self.suite.add_result(result)
                result.print_summary(verbose=False)
            except KeyboardInterrupt:
                print("\n⚠️  Burn-in test interrupted by user")
            except Exception as e:
                print(f"  ⚠️  Burn-in test failed: {e}")
                import traceback
                traceback.print_exc()

            print()

        # Cleanup health monitor
        if health_monitor:
            try:
                health_monitor.cleanup()
            except:
                pass

        # TODO: Add more tests
        # - NCCL Profiler (multi-GPU)

        # Print summary
        self._print_summary()

    def _run_test(self, test, mode: TestMode) -> TestResult:
        """Run a single test and handle errors"""
        try:
            return test.run(mode=mode)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return TestResult(
                test_name=test.name,
                status=TestStatus.FAILED,
                error_message=str(e)
            )

    def _run_test_with_mode(self, test, mode: TestMode) -> TestResult:
        """Run a test with mode parameter (same as _run_test, for clarity)"""
        return self._run_test(test, mode)

    def _print_summary(self):
        """Print final summary"""
        print("=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)

        summary = self.suite.get_summary()
        print(f"Total Tests: {len(self.suite.tests)}")
        print(f"  ✓ Passed: {summary.get('passed', 0)}")
        print(f"  ⚠ Warning: {summary.get('warning', 0)}")
        print(f"  ✗ Failed: {summary.get('failed', 0)}")
        print()

        # Print key insights
        self._print_key_insights()

        print("=" * 80)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def _print_key_insights(self):
        """Extract and print key insights from all tests"""
        print("🎯 KEY INSIGHTS:")
        print("-" * 80)

        # Find specific test results
        hw_test = next((t for t in self.suite.tests if "Hardware" in t.test_name), None)
        mem_test = next((t for t in self.suite.tests if "Memory" in t.test_name), None)
        compute_test = next((t for t in self.suite.tests if "Compute" in t.test_name), None)

        # GPU Info
        if hw_test and "gpus" in hw_test.metrics:
            gpus = hw_test.metrics["gpus"]
            if gpus:
                gpu = gpus[0]
                print(f"  GPU: {gpu['name']}")
                print(f"       {gpu['total_memory']} VRAM")
                print(f"       Compute Capability {gpu['compute_capability']}")
                print()

        # Memory Insights
        if mem_test:
            usable = mem_test.metrics.get("max_allocatable", "N/A")
            bandwidth = mem_test.metrics.get("device_to_device_bandwidth", "N/A")
            print(f"  Memory: {usable} usable")
            print(f"          {bandwidth} bandwidth")
            print()

        # Compute Insights
        if compute_test:
            fp16 = compute_test.metrics.get("fp16_tflops", 0)
            fp32 = compute_test.metrics.get("fp32_tflops", 0)
            print(f"  Compute: {fp16} TFLOPS (FP16)")
            print(f"           {fp32} TFLOPS (FP32)")
            if fp32 > 0:
                speedup = fp16 / fp32
                print(f"           {speedup:.1f}x speedup with FP16")
            print()

        # Model Size Recommendations
        if mem_test:
            mem_gb = mem_test.metrics.get("max_allocatable_bytes", 0) / (1024**3)
            print("  Recommended Model Sizes:")
            if mem_gb >= 24:
                print("    ✓ Up to 30B parameters (FP16)")
                print("    ✓ Up to 70B parameters (4-bit quantization)")
            elif mem_gb >= 12:
                print("    ✓ Up to 13B parameters (FP16)")
                print("    ✓ Up to 30B parameters (4-bit quantization)")
            elif mem_gb >= 6:
                print("    ✓ Up to 7B parameters (FP16)")
                print("    ✓ Up to 13B parameters (4-bit quantization)")
            else:
                print("    ⚠ Smaller models recommended (<7B)")
                print("    ⚠ Use 4-bit quantization for larger models")
            print()


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Hardware Diagnostics & Performance Testing for LLM Workloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick check (1-5 minutes)
  python -m diagnostics.cli --mode quick

  # Deep benchmarking (20-30 minutes)
  python -m diagnostics.cli --mode deep

  # Burn-in testing (hours)
  python -m diagnostics.cli --mode burn-in --duration 24h

For hobbyists: This tool helps you understand if your GPU can run specific LLM models!
        """
    )

    parser.add_argument(
        "--mode",
        choices=["quick", "deep", "burn-in"],
        default="quick",
        help="Test mode: quick (1-5 min), deep (20-30 min), or burn-in (hours)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output report file (html, json, or md)"
    )

    parser.add_argument(
        "--duration",
        type=str,
        help="Duration for burn-in mode (e.g., 24h, 48h)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to test (default: cuda:0)"
    )

    args = parser.parse_args()

    # Map mode string to enum
    mode_map = {
        "quick": TestMode.QUICK,
        "deep": TestMode.DEEP,
        "burn-in": TestMode.BURN_IN
    }
    mode = mode_map[args.mode]

    # Parse duration for burn-in mode
    duration_hours = 1.0  # Default 1 hour
    if args.duration:
        duration_str = args.duration.lower()
        try:
            if duration_str.endswith('h'):
                duration_hours = float(duration_str[:-1])
            elif duration_str.endswith('m'):
                duration_hours = float(duration_str[:-1]) / 60
            else:
                duration_hours = float(duration_str)
        except ValueError:
            print(f"⚠️  Invalid duration format: {args.duration}")
            print("   Use format like: 24h, 2.5h, 90m")
            sys.exit(1)

    # Run diagnostics
    cli = DiagnosticsCLI()

    # For burn-in mode, pass duration to the test
    if mode == TestMode.BURN_IN:
        # We'll need to pass duration through - let's modify run_all_tests
        cli.burn_in_duration = duration_hours

    cli.run_all_tests(mode=mode)

    # TODO: Generate report if --output specified
    if args.output:
        print(f"\n⚠️  Report generation not yet implemented")
        print(f"    Planned output: {args.output}")


if __name__ == "__main__":
    main()
