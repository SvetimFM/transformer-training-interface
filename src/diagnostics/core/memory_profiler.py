"""
Memory Profiler

Tests:
1. Memory Bandwidth - How fast can we move data?
2. Allocatable Memory - How much VRAM can we actually use?
3. Allocation Patterns - Simulating model loading scenarios

Why this matters:
- Memory bandwidth affects token generation speed
- Knowing max allocatable memory helps size models correctly
- Understanding fragmentation helps with memory planning
"""

import torch
import time
import gc
from typing import Dict, Any, List, Tuple
from .metrics import (
    DiagnosticTest, TestResult, TestMode, TestStatus, MetricType,
    format_bytes, format_percentage, ProgressTracker
)


class MemoryProfiler(DiagnosticTest):
    """
    Comprehensive GPU memory profiling.

    Measures:
    - Memory bandwidth (device-to-device, host-to-device, device-to-host)
    - Maximum allocatable memory
    - Fragmentation characteristics
    - Allocation/deallocation performance
    """

    def __init__(self, device: str = "cuda:0"):
        super().__init__(
            name="Memory Profiler",
            description="Test GPU memory bandwidth, capacity, and allocation patterns",
            metric_type=MetricType.MEMORY
        )
        self.device = device

        # Check if device is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.device_props = torch.cuda.get_device_properties(device)

    def run(self, mode: TestMode = TestMode.QUICK) -> TestResult:
        """Run memory profiling tests"""
        try:
            metrics = {}

            # Always run: Basic capacity check
            self._report_progress("Checking memory capacity...", 10)
            capacity_metrics = self._test_capacity()
            metrics.update(capacity_metrics)

            # Always run: Bandwidth test
            self._report_progress("Testing memory bandwidth...", 40)
            bandwidth_metrics = self._test_bandwidth(mode)
            metrics.update(bandwidth_metrics)

            if mode in [TestMode.DEEP, TestMode.BURN_IN]:
                # Detailed allocation pattern test
                self._report_progress("Testing allocation patterns...", 70)
                allocation_metrics = self._test_allocation_patterns()
                metrics.update(allocation_metrics)

            if mode == TestMode.BURN_IN:
                # Long-running stability test
                self._report_progress("Running memory stability test...", 85)
                stability_metrics = self._test_stability()
                metrics.update(stability_metrics)

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
                interpretation="Memory profiling failed",
                recommendation="Check GPU availability and drivers",
                error_message=str(e)
            )

    def _test_capacity(self) -> Dict[str, Any]:
        """Test memory capacity and find max allocatable"""
        metrics = {}

        # Total and free memory
        total_memory = self.device_props.total_memory
        reserved = torch.cuda.memory_reserved(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        free = total_memory - reserved

        metrics["total_memory_bytes"] = total_memory
        metrics["total_memory"] = format_bytes(total_memory)
        metrics["free_memory_bytes"] = free
        metrics["free_memory"] = format_bytes(free)
        metrics["reserved_bytes"] = reserved
        metrics["allocated_bytes"] = allocated

        # Find maximum allocatable (with binary search)
        max_allocatable = self._find_max_allocatable()
        metrics["max_allocatable_bytes"] = max_allocatable
        metrics["max_allocatable"] = format_bytes(max_allocatable)
        metrics["usable_percentage"] = (max_allocatable / total_memory) * 100

        return metrics

    def _find_max_allocatable(self) -> int:
        """Binary search to find maximum allocatable memory"""
        # Clear cache first
        torch.cuda.empty_cache()
        gc.collect()

        total_memory = self.device_props.total_memory
        low, high = 0, int(total_memory * 0.95)  # Start at 95% of total

        max_size = 0

        while low <= high:
            mid = (low + high) // 2
            # Try to allocate this amount
            try:
                tensor = torch.empty(mid // 4, dtype=torch.float32, device=self.device)
                del tensor
                torch.cuda.synchronize()
                max_size = mid
                low = mid + 1  # Try larger
            except RuntimeError:
                high = mid - 1  # Try smaller

        # Clean up
        torch.cuda.empty_cache()
        gc.collect()

        return max_size

    def _test_bandwidth(self, mode: TestMode) -> Dict[str, Any]:
        """Test memory bandwidth for various operations"""
        metrics = {}

        # Determine test sizes based on mode
        if mode == TestMode.QUICK:
            sizes_mb = [100, 500, 1000]  # Quick test with smaller sizes
        elif mode == TestMode.DEEP:
            sizes_mb = [10, 50, 100, 500, 1000, 2000]  # More comprehensive
        else:  # BURN_IN
            sizes_mb = [100, 500, 1000, 2000]  # Representative sizes

        # Device-to-device bandwidth (most important for inference)
        d2d_bandwidth = self._measure_device_to_device_bandwidth(sizes_mb)
        metrics["device_to_device_bandwidth_gbps"] = d2d_bandwidth
        metrics["device_to_device_bandwidth"] = f"{d2d_bandwidth:.2f} GB/s"

        # Host-to-device bandwidth (affects model loading)
        h2d_bandwidth = self._measure_host_to_device_bandwidth(sizes_mb)
        metrics["host_to_device_bandwidth_gbps"] = h2d_bandwidth
        metrics["host_to_device_bandwidth"] = f"{h2d_bandwidth:.2f} GB/s"

        # Device-to-host bandwidth (affects copying results back)
        d2h_bandwidth = self._measure_device_to_host_bandwidth(sizes_mb)
        metrics["device_to_host_bandwidth_gbps"] = d2h_bandwidth
        metrics["device_to_host_bandwidth"] = f"{d2h_bandwidth:.2f} GB/s"

        # Theoretical peak (for comparison)
        # Most GPUs have similar read/write bandwidth
        theoretical_peak = self._estimate_theoretical_bandwidth()
        if theoretical_peak:
            metrics["theoretical_bandwidth_gbps"] = theoretical_peak
            metrics["theoretical_bandwidth"] = f"{theoretical_peak:.2f} GB/s"
            metrics["bandwidth_efficiency"] = format_percentage((d2d_bandwidth / theoretical_peak) * 100)

        return metrics

    def _measure_device_to_device_bandwidth(self, sizes_mb: List[int]) -> float:
        """Measure device-to-device copy bandwidth"""
        bandwidths = []

        for size_mb in sizes_mb:
            num_elements = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
            src = torch.randn(num_elements, dtype=torch.float32, device=self.device)
            dst = torch.empty_like(src)

            # Warm-up
            dst.copy_(src)
            torch.cuda.synchronize()

            # Measure
            num_iterations = 10
            start = time.perf_counter()
            for _ in range(num_iterations):
                dst.copy_(src)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            bytes_transferred = num_elements * 4 * num_iterations
            bandwidth_gbps = (bytes_transferred / elapsed) / (1024**3)
            bandwidths.append(bandwidth_gbps)

            # Clean up
            del src, dst

        # Return average of larger sizes (more stable)
        return sum(bandwidths[-3:]) / min(3, len(bandwidths))

    def _measure_host_to_device_bandwidth(self, sizes_mb: List[int]) -> float:
        """Measure host-to-device copy bandwidth"""
        bandwidths = []

        for size_mb in sizes_mb:
            num_elements = (size_mb * 1024 * 1024) // 4
            host_tensor = torch.randn(num_elements, dtype=torch.float32)
            device_tensor = torch.empty(num_elements, dtype=torch.float32, device=self.device)

            # Warm-up
            device_tensor.copy_(host_tensor)
            torch.cuda.synchronize()

            # Measure
            num_iterations = 5
            start = time.perf_counter()
            for _ in range(num_iterations):
                device_tensor.copy_(host_tensor)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            bytes_transferred = num_elements * 4 * num_iterations
            bandwidth_gbps = (bytes_transferred / elapsed) / (1024**3)
            bandwidths.append(bandwidth_gbps)

            del host_tensor, device_tensor

        return sum(bandwidths[-3:]) / min(3, len(bandwidths))

    def _measure_device_to_host_bandwidth(self, sizes_mb: List[int]) -> float:
        """Measure device-to-host copy bandwidth"""
        bandwidths = []

        for size_mb in sizes_mb:
            num_elements = (size_mb * 1024 * 1024) // 4
            device_tensor = torch.randn(num_elements, dtype=torch.float32, device=self.device)
            host_tensor = torch.empty(num_elements, dtype=torch.float32)

            # Warm-up
            host_tensor.copy_(device_tensor)
            torch.cuda.synchronize()

            # Measure
            num_iterations = 5
            start = time.perf_counter()
            for _ in range(num_iterations):
                host_tensor.copy_(device_tensor)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            bytes_transferred = num_elements * 4 * num_iterations
            bandwidth_gbps = (bytes_transferred / elapsed) / (1024**3)
            bandwidths.append(bandwidth_gbps)

            del device_tensor, host_tensor

        return sum(bandwidths[-3:]) / min(3, len(bandwidths))

    def _estimate_theoretical_bandwidth(self) -> float:
        """Estimate theoretical peak bandwidth based on GPU specs"""
        # This is approximate - actual values vary by GPU model
        # Most modern GPUs achieve 80-95% of theoretical peak in practice

        gpu_name = self.device_props.name.lower()

        # Known theoretical bandwidths (GB/s)
        known_specs = {
            "3090": 936,
            "3080": 760,
            "3070": 448,
            "4090": 1008,
            "4080": 716,
            "4070": 504,
            "a100": 1555,
            "a40": 696,
            "v100": 900,
            "h100": 3350,
        }

        for model, bandwidth in known_specs.items():
            if model in gpu_name:
                return bandwidth

        # If unknown, return None
        return None

    def _test_allocation_patterns(self) -> Dict[str, Any]:
        """Test various allocation patterns that simulate model loading"""
        metrics = {}

        # Pattern 1: Large single allocation (e.g., loading a big model)
        large_alloc_time, large_alloc_size = self._test_large_allocation()
        metrics["large_allocation_time_ms"] = large_alloc_time * 1000
        metrics["large_allocation_size"] = format_bytes(large_alloc_size)

        # Pattern 2: Many small allocations (e.g., many model parameters)
        small_alloc_time, small_alloc_count = self._test_many_small_allocations()
        metrics["small_allocations_time_ms"] = small_alloc_time * 1000
        metrics["small_allocations_count"] = small_alloc_count

        # Pattern 3: Fragmentation test
        frag_score = self._test_fragmentation()
        metrics["fragmentation_score"] = frag_score

        return metrics

    def _test_large_allocation(self) -> Tuple[float, int]:
        """Test single large allocation"""
        torch.cuda.empty_cache()

        # Try to allocate 80% of free memory
        free_memory = self.device_props.total_memory - torch.cuda.memory_allocated(self.device)
        alloc_size = int(free_memory * 0.8)

        start = time.perf_counter()
        try:
            tensor = torch.empty(alloc_size // 4, dtype=torch.float32, device=self.device)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            del tensor
            return elapsed, alloc_size
        except RuntimeError:
            return 0.0, 0

    def _test_many_small_allocations(self) -> Tuple[float, int]:
        """Test many small allocations"""
        torch.cuda.empty_cache()

        # Allocate 100 small tensors (simulating model parameters)
        alloc_size = 1024 * 1024  # 1MB each
        count = 100

        start = time.perf_counter()
        tensors = []
        for _ in range(count):
            tensors.append(torch.empty(alloc_size // 4, dtype=torch.float32, device=self.device))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Clean up
        del tensors
        torch.cuda.empty_cache()

        return elapsed, count

    def _test_fragmentation(self) -> float:
        """Test memory fragmentation by alternating allocations and deallocations"""
        torch.cuda.empty_cache()

        # Create fragmentation
        tensors = []
        for i in range(10):
            size = (i + 1) * 10 * 1024 * 1024  # 10MB, 20MB, ..., 100MB
            tensors.append(torch.empty(size // 4, dtype=torch.float32, device=self.device))

        # Free every other tensor (set to None to release GPU memory)
        for i in range(0, len(tensors), 2):
            tensors[i] = None

        # Try to allocate a large chunk
        try:
            large = torch.empty(400 * 1024 * 1024 // 4, dtype=torch.float32, device=self.device)
            del large
            frag_score = 0.0  # No fragmentation
        except RuntimeError:
            frag_score = 1.0  # Severe fragmentation

        # Clean up
        del tensors
        torch.cuda.empty_cache()

        return frag_score

    def _test_stability(self) -> Dict[str, Any]:
        """Long-running memory stability test for burn-in mode"""
        # TODO: Implement burn-in specific tests
        # For now, return placeholder
        return {
            "stability_test": "Not yet implemented",
            "duration_minutes": 0
        }

    def _generate_interpretation(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable interpretation"""
        lines = []

        # Memory capacity
        total = metrics.get("total_memory", "N/A")
        usable = metrics.get("max_allocatable", "N/A")
        usable_pct = metrics.get("usable_percentage", 0)
        lines.append(f"Total Memory: {total}, Usable: {usable} ({usable_pct:.1f}%)")

        # Bandwidth
        d2d_bw = metrics.get("device_to_device_bandwidth_gbps", 0)
        lines.append(f"Memory Bandwidth: {d2d_bw:.1f} GB/s (device-to-device)")

        # Efficiency
        if "bandwidth_efficiency" in metrics:
            eff = metrics["bandwidth_efficiency"]
            lines.append(f"  Bandwidth Efficiency: {eff} of theoretical peak")

        return "\n  ".join(lines)

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> str:
        """Generate recommendations based on results"""
        recommendations = []

        # Bandwidth recommendations
        d2d_bw = metrics.get("device_to_device_bandwidth_gbps", 0)
        if d2d_bw > 500:
            recommendations.append("✓ Excellent memory bandwidth - great for large models")
        elif d2d_bw > 300:
            recommendations.append("✓ Good memory bandwidth - suitable for most workloads")
        elif d2d_bw > 100:
            recommendations.append("⚠ Moderate bandwidth - may bottleneck on very large models")
        else:
            recommendations.append("⚠ Low bandwidth - consider upgrading GPU")

        # Efficiency recommendations
        if "theoretical_bandwidth_gbps" in metrics:
            efficiency = metrics["device_to_device_bandwidth_gbps"] / metrics["theoretical_bandwidth_gbps"]
            if efficiency < 0.7:
                recommendations.append("⚠ Bandwidth below 70% of theoretical - check for thermal throttling")
            elif efficiency > 0.9:
                recommendations.append("✓ Excellent bandwidth efficiency!")

        # Usable memory recommendations
        usable_pct = metrics.get("usable_percentage", 0)
        if usable_pct < 85:
            recommendations.append("⚠ Less than 85% memory usable - some overhead from system/PyTorch")

        return "\n  ".join(recommendations) if recommendations else "Memory performance looks good!"
