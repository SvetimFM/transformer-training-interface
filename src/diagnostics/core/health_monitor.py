"""
GPU Health Monitoring Module

Provides real-time GPU health metrics via NVML (NVIDIA) or ROCm SMI (AMD).
Critical for:
- Burn-in testing (detect thermal throttling over hours)
- Performance debugging (power limits, clock drift)
- Production monitoring (catch hardware issues early)

Uses:
- NVIDIA: pynvml library (Python bindings for NVML)
- AMD: subprocess calls to rocm-smi
- CPU: psutil for temperature/power if available
"""

import time
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from .backend_detector import BackendDetector, ComputeBackend


@dataclass
class GPUHealthSnapshot:
    """Single point-in-time GPU health measurement"""
    timestamp: float  # Unix timestamp
    gpu_id: int

    # Temperature (Celsius)
    temperature: Optional[float] = None
    temperature_threshold: Optional[float] = None  # Max safe temp

    # Power (Watts)
    power_draw: Optional[float] = None
    power_limit: Optional[float] = None

    # Clocks (MHz)
    clock_sm: Optional[int] = None          # Streaming Multiprocessor (core) clock
    clock_memory: Optional[int] = None
    clock_sm_max: Optional[int] = None      # Maximum possible
    clock_memory_max: Optional[int] = None

    # Utilization (%)
    utilization_gpu: Optional[int] = None
    utilization_memory: Optional[int] = None

    # Memory (bytes)
    memory_used: Optional[int] = None
    memory_total: Optional[int] = None

    # Fan (%)
    fan_speed: Optional[int] = None

    # Status flags
    is_throttled: bool = False
    throttle_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'timestamp': self.timestamp,
            'gpu_id': self.gpu_id,
            'temperature': self.temperature,
            'power_draw': self.power_draw,
            'clock_sm': self.clock_sm,
            'utilization_gpu': self.utilization_gpu,
            'memory_used_gb': round(self.memory_used / (1024**3), 2) if self.memory_used else None,
            'is_throttled': self.is_throttled,
        }

        if self.throttle_reason:
            result['throttle_reason'] = self.throttle_reason

        return result

    def __str__(self) -> str:
        """Human-readable string"""
        parts = [
            f"GPU {self.gpu_id}:",
            f"Temp: {self.temperature:.0f}°C" if self.temperature else "Temp: N/A",
            f"Power: {self.power_draw:.1f}W" if self.power_draw else "Power: N/A",
            f"Clock: {self.clock_sm}MHz" if self.clock_sm else "Clock: N/A",
            f"Util: {self.utilization_gpu}%" if self.utilization_gpu is not None else "Util: N/A",
        ]

        if self.is_throttled:
            parts.append(f"⚠ THROTTLED ({self.throttle_reason})")

        return " ".join(parts)


class GPUHealthMonitor:
    """
    Unified GPU health monitoring for NVIDIA, AMD, and CPU.

    Usage:
        monitor = GPUHealthMonitor()
        snapshot = monitor.get_snapshot(gpu_id=0)
        print(f"Temperature: {snapshot.temperature}°C")

    For continuous monitoring:
        monitor.start_logging(interval_seconds=60, duration_hours=24)
        # ... run workload ...
        report = monitor.stop_logging()
    """

    def __init__(self):
        """Initialize health monitor based on detected backend"""
        self.backend_info = BackendDetector.detect()
        self.backend = self.backend_info.backend

        # NVIDIA-specific initialization
        if self.backend == ComputeBackend.CUDA_NVIDIA:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                self.handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(self.backend_info.device_count)
                ]
            except ImportError:
                raise ImportError(
                    "pynvml not installed. Install with: pip install nvidia-ml-py3"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize NVML: {e}")

        # AMD-specific initialization
        elif self.backend == ComputeBackend.ROCM_AMD:
            # Check if rocm-smi is available
            try:
                result = subprocess.run(
                    ['rocm-smi', '--help'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode != 0:
                    raise RuntimeError("rocm-smi not found or not working")
            except FileNotFoundError:
                raise RuntimeError("rocm-smi not found. Is ROCm installed?")

        # CPU monitoring
        elif self.backend == ComputeBackend.CPU:
            try:
                import psutil
                self.psutil = psutil
            except ImportError:
                raise ImportError("psutil not installed. Install with: pip install psutil")

        # Logging state
        self.is_logging = False
        self.log_history: List[GPUHealthSnapshot] = []

    def get_snapshot(self, gpu_id: int = 0) -> GPUHealthSnapshot:
        """
        Get current health metrics for specified GPU.

        Args:
            gpu_id: GPU index (0 for first GPU)

        Returns:
            GPUHealthSnapshot with current metrics
        """
        snapshot = GPUHealthSnapshot(
            timestamp=time.time(),
            gpu_id=gpu_id
        )

        if self.backend == ComputeBackend.CUDA_NVIDIA:
            self._fill_nvidia_snapshot(snapshot, gpu_id)
        elif self.backend == ComputeBackend.ROCM_AMD:
            self._fill_amd_snapshot(snapshot, gpu_id)
        elif self.backend == ComputeBackend.CPU:
            self._fill_cpu_snapshot(snapshot)

        return snapshot

    def _fill_nvidia_snapshot(self, snapshot: GPUHealthSnapshot, gpu_id: int):
        """Fill snapshot with NVIDIA NVML data"""
        handle = self.handles[gpu_id]
        nvml = self.nvml

        try:
            # Temperature
            try:
                snapshot.temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            except:
                pass

            try:
                # Try different threshold constants (varies by pynvml version)
                if hasattr(nvml, 'NVML_TEMPERATURE_THRESHOLD_GPU_MAX'):
                    snapshot.temperature_threshold = nvml.nvmlDeviceGetTemperatureThreshold(
                        handle, nvml.NVML_TEMPERATURE_THRESHOLD_GPU_MAX
                    )
                elif hasattr(nvml, 'NVML_TEMPERATURE_THRESHOLD_SHUTDOWN'):
                    snapshot.temperature_threshold = nvml.nvmlDeviceGetTemperatureThreshold(
                        handle, nvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN
                    )
            except:
                pass

            # Power
            try:
                snapshot.power_draw = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            except:
                pass

            try:
                snapshot.power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except:
                pass

            # Clocks
            try:
                snapshot.clock_sm = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
            except:
                pass

            try:
                snapshot.clock_memory = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
            except:
                pass

            try:
                snapshot.clock_sm_max = nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_SM)
            except:
                pass

            try:
                snapshot.clock_memory_max = nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_MEM)
            except:
                pass

            # Utilization
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                snapshot.utilization_gpu = util.gpu
                snapshot.utilization_memory = util.memory
            except:
                pass

            # Memory
            try:
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                snapshot.memory_used = mem_info.used
                snapshot.memory_total = mem_info.total
            except:
                pass

            # Fan speed (if available)
            try:
                snapshot.fan_speed = nvml.nvmlDeviceGetFanSpeed(handle)
            except:
                pass  # Not all GPUs have fan speed reporting

            # Throttle detection
            try:
                perf_state = nvml.nvmlDeviceGetPerformanceState(handle)
                if perf_state > 0:  # P0 is max performance, higher numbers = throttled
                    snapshot.is_throttled = True
                    snapshot.throttle_reason = f"Performance state P{perf_state}"
            except:
                pass

            # Check if clocks are significantly below max
            if snapshot.clock_sm and snapshot.clock_sm_max:
                if snapshot.clock_sm < snapshot.clock_sm_max * 0.9:  # More than 10% below max
                    snapshot.is_throttled = True
                    if not snapshot.throttle_reason:
                        # Determine reason
                        if snapshot.temperature and snapshot.temperature > 80:
                            snapshot.throttle_reason = "Thermal"
                        elif snapshot.power_draw and snapshot.power_limit:
                            if snapshot.power_draw >= snapshot.power_limit * 0.95:
                                snapshot.throttle_reason = "Power limit"
                        else:
                            snapshot.throttle_reason = "Unknown"

        except Exception as e:
            # Log error but don't fail entirely - some metrics may still be valid
            if not snapshot.temperature:  # Only set error if we got nothing
                snapshot.throttle_reason = f"Error reading NVML: {e}"

    def _fill_amd_snapshot(self, snapshot: GPUHealthSnapshot, gpu_id: int):
        """Fill snapshot with AMD ROCm SMI data"""
        try:
            # Run rocm-smi for this GPU
            result = subprocess.run(
                ['rocm-smi', '-d', str(gpu_id), '--showtemp', '--showpower', '--showclocks', '--showuse'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                snapshot.throttle_reason = "rocm-smi failed"
                return

            # Parse output (rocm-smi output format is not as structured as NVML)
            output = result.stdout

            # Temperature
            for line in output.split('\n'):
                if 'Temperature' in line:
                    try:
                        temp_str = line.split(':')[-1].strip().replace('c', '').strip()
                        snapshot.temperature = float(temp_str)
                    except:
                        pass

                # Power
                if 'Average Graphics Package Power' in line or 'Power' in line:
                    try:
                        power_str = line.split(':')[-1].strip().replace('W', '').strip()
                        snapshot.power_draw = float(power_str)
                    except:
                        pass

                # Clocks
                if 'sclk' in line.lower() or 'GPU Clock' in line:
                    try:
                        clock_str = line.split(':')[-1].strip().replace('MHz', '').strip()
                        snapshot.clock_sm = int(clock_str)
                    except:
                        pass

                if 'mclk' in line.lower() or 'Memory Clock' in line:
                    try:
                        clock_str = line.split(':')[-1].strip().replace('MHz', '').strip()
                        snapshot.clock_memory = int(clock_str)
                    except:
                        pass

                # Utilization
                if 'GPU use' in line or 'GPU Utilization' in line:
                    try:
                        util_str = line.split(':')[-1].strip().replace('%', '').strip()
                        snapshot.utilization_gpu = int(util_str)
                    except:
                        pass

        except Exception as e:
            snapshot.throttle_reason = f"Error reading rocm-smi: {e}"

    def _fill_cpu_snapshot(self, snapshot: GPUHealthSnapshot):
        """Fill snapshot with CPU metrics (if available)"""
        try:
            # CPU temperature (if available)
            if hasattr(self.psutil, 'sensors_temperatures'):
                temps = self.psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature sensor
                    for name, entries in temps.items():
                        if entries:
                            snapshot.temperature = entries[0].current
                            break

            # CPU utilization
            snapshot.utilization_gpu = int(self.psutil.cpu_percent(interval=0.1))

            # Memory
            mem = self.psutil.virtual_memory()
            snapshot.memory_used = mem.used
            snapshot.memory_total = mem.total

        except Exception as e:
            snapshot.throttle_reason = f"Error reading CPU metrics: {e}"

    def start_continuous_monitoring(self, interval_seconds: int = 60):
        """
        Start continuous monitoring in background.

        Args:
            interval_seconds: How often to sample (default 60s = 1 minute)
        """
        self.is_logging = True
        self.log_history = []
        self.log_interval = interval_seconds

    def record_snapshot(self, gpu_id: int = 0):
        """
        Record a snapshot to the log history.
        Call this periodically during long-running tests.
        """
        if self.is_logging:
            snapshot = self.get_snapshot(gpu_id)
            self.log_history.append(snapshot)

    def get_monitoring_report(self) -> Dict[str, Any]:
        """
        Generate report from logged snapshots.

        Returns summary statistics and detected issues.
        """
        if not self.log_history:
            return {"error": "No monitoring data collected"}

        report = {
            "duration_seconds": self.log_history[-1].timestamp - self.log_history[0].timestamp,
            "sample_count": len(self.log_history),
            "gpu_id": self.log_history[0].gpu_id,
        }

        # Temperature stats
        temps = [s.temperature for s in self.log_history if s.temperature is not None]
        if temps:
            report["temperature"] = {
                "min": round(min(temps), 1),
                "max": round(max(temps), 1),
                "avg": round(sum(temps) / len(temps), 1),
            }

        # Power stats
        powers = [s.power_draw for s in self.log_history if s.power_draw is not None]
        if powers:
            report["power"] = {
                "min": round(min(powers), 1),
                "max": round(max(powers), 1),
                "avg": round(sum(powers) / len(powers), 1),
            }

        # Throttle events
        throttle_events = [s for s in self.log_history if s.is_throttled]
        report["throttle_events"] = len(throttle_events)

        if throttle_events:
            reasons = {}
            for event in throttle_events:
                reason = event.throttle_reason or "Unknown"
                reasons[reason] = reasons.get(reason, 0) + 1
            report["throttle_reasons"] = reasons

        # Clock stats
        clocks = [s.clock_sm for s in self.log_history if s.clock_sm is not None]
        if clocks:
            report["clock_sm"] = {
                "min": min(clocks),
                "max": max(clocks),
                "avg": round(sum(clocks) / len(clocks)),
            }

        return report

    def cleanup(self):
        """Cleanup resources (important for NVML)"""
        if self.backend == ComputeBackend.CUDA_NVIDIA and hasattr(self, 'nvml'):
            try:
                self.nvml.nvmlShutdown()
            except:
                pass


if __name__ == "__main__":
    """Test health monitoring"""
    print("=" * 80)
    print("GPU HEALTH MONITORING TEST")
    print("=" * 80)
    print()

    try:
        monitor = GPUHealthMonitor()
        print(f"Backend: {monitor.backend.value}")
        print(f"Device count: {monitor.backend_info.device_count}")
        print()

        # Get snapshot for each GPU
        for gpu_id in range(monitor.backend_info.device_count):
            snapshot = monitor.get_snapshot(gpu_id)
            print(snapshot)
            print()

        print("Detailed snapshot (GPU 0):")
        snapshot = monitor.get_snapshot(0)
        for key, value in snapshot.to_dict().items():
            print(f"  {key}: {value}")

        monitor.cleanup()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)
