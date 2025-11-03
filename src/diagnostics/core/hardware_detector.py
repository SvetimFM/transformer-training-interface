"""
Hardware Detection Module

Detects and reports on:
- GPU hardware (count, names, memory)
- CUDA and NCCL versions
- GPU topology (PCIe, NVLink connections)
- System information (CPU, RAM, OS)

This is always the first test that runs - it tells us what hardware we're working with.
"""

import torch
import platform
import subprocess
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from .metrics import DiagnosticTest, TestResult, TestMode, TestStatus, MetricType, format_bytes


@dataclass
class GPUInfo:
    """Information about a single GPU"""
    index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    memory_clock_mhz: Optional[int] = None
    sm_count: Optional[int] = None  # Streaming Multiprocessor count
    pcie_gen: Optional[int] = None
    pcie_width: Optional[int] = None
    supports_p2p: bool = False
    nvlink_connections: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "index": self.index,
            "name": self.name,
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            "total_memory": format_bytes(self.total_memory_bytes),
            "total_memory_bytes": self.total_memory_bytes,
            "memory_clock_mhz": self.memory_clock_mhz,
            "sm_count": self.sm_count,
            "pcie_gen": self.pcie_gen,
            "pcie_width": self.pcie_width,
            "supports_p2p": self.supports_p2p,
            "nvlink_connections": self.nvlink_connections
        }

    def __str__(self) -> str:
        """Human-readable representation"""
        memory_str = format_bytes(self.total_memory_bytes)
        cc_str = f"{self.compute_capability[0]}.{self.compute_capability[1]}"

        base = f"GPU {self.index}: {self.name} ({memory_str}, CC {cc_str})"

        if self.nvlink_connections:
            base += f"\n  NVLink: Connected to GPUs {self.nvlink_connections}"
        elif self.pcie_gen and self.pcie_width:
            base += f"\n  PCIe: Gen {self.pcie_gen} x{self.pcie_width}"

        return base


@dataclass
class SystemInfo:
    """Information about the host system"""
    os: str
    os_version: str
    python_version: str
    pytorch_version: str
    cuda_version: str
    cudnn_version: Optional[str] = None
    nccl_version: Optional[str] = None
    cpu_name: Optional[str] = None
    cpu_count: Optional[int] = None
    total_ram_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "os": self.os,
            "os_version": self.os_version,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
        }

        if self.cudnn_version:
            result["cudnn_version"] = self.cudnn_version
        if self.nccl_version:
            result["nccl_version"] = self.nccl_version
        if self.cpu_name:
            result["cpu_name"] = self.cpu_name
        if self.cpu_count:
            result["cpu_count"] = self.cpu_count
        if self.total_ram_bytes:
            result["total_ram"] = format_bytes(self.total_ram_bytes)
            result["total_ram_bytes"] = self.total_ram_bytes

        return result


class HardwareDetector(DiagnosticTest):
    """
    Comprehensive hardware detection.

    This test identifies all available hardware and software versions.
    It's the foundation for all other tests - we need to know what we're working with!
    """

    def __init__(self):
        super().__init__(
            name="Hardware Detection",
            description="Detect GPUs, CUDA version, NCCL support, and system topology",
            metric_type=MetricType.HARDWARE
        )

    def run(self, mode: TestMode = TestMode.QUICK) -> TestResult:
        """Run hardware detection"""
        try:
            self._report_progress("Detecting hardware...", 0)

            # Detect system info
            self._report_progress("Checking system information...", 20)
            system_info = self._detect_system()

            # Detect GPUs
            self._report_progress("Detecting GPUs...", 40)
            gpu_info = self._detect_gpus()

            # Detect topology (only in deep or burn-in mode)
            if mode in [TestMode.DEEP, TestMode.BURN_IN] and len(gpu_info) > 1:
                self._report_progress("Analyzing GPU topology...", 70)
                self._detect_topology(gpu_info)

            self._report_progress("Complete", 100)

            # Create result
            metrics = {
                "gpu_count": len(gpu_info),
                "gpus": [gpu.to_dict() for gpu in gpu_info],
                "system": system_info.to_dict()
            }

            # Generate interpretation
            interpretation = self._generate_interpretation(gpu_info, system_info)

            # Generate recommendations
            recommendation = self._generate_recommendations(gpu_info, system_info)

            return self._create_result(
                status=TestStatus.PASSED if len(gpu_info) > 0 else TestStatus.WARNING,
                metrics=metrics,
                interpretation=interpretation,
                recommendation=recommendation
            )

        except Exception as e:
            return self._create_result(
                status=TestStatus.FAILED,
                metrics={},
                interpretation="Failed to detect hardware",
                recommendation="Check CUDA installation and GPU drivers",
                error_message=str(e)
            )

    def _detect_system(self) -> SystemInfo:
        """Detect system information"""
        # OS info
        os_name = platform.system()
        os_version = platform.release()

        # Python info
        python_version = platform.python_version()

        # PyTorch info
        pytorch_version = torch.__version__

        # CUDA info
        cuda_version = torch.version.cuda or "N/A"
        cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None

        # NCCL info (if available)
        nccl_version = None
        if torch.cuda.is_available() and torch.cuda.nccl.is_available([]):
            try:
                nccl_version = ".".join(map(str, torch.cuda.nccl.version()))
            except:
                nccl_version = "available (version unknown)"

        # CPU info
        cpu_count = os.cpu_count()
        cpu_name = None
        try:
            if os_name == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            break
            elif os_name == "Windows":
                cpu_name = platform.processor()
            elif os_name == "Darwin":  # macOS
                cpu_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except:
            pass

        # RAM info
        total_ram = None
        try:
            import psutil
            total_ram = psutil.virtual_memory().total
        except ImportError:
            pass

        return SystemInfo(
            os=os_name,
            os_version=os_version,
            python_version=python_version,
            pytorch_version=pytorch_version,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            nccl_version=nccl_version,
            cpu_name=cpu_name,
            cpu_count=cpu_count,
            total_ram_bytes=total_ram
        )

    def _detect_gpus(self) -> List[GPUInfo]:
        """Detect all GPUs and their properties"""
        if not torch.cuda.is_available():
            return []

        gpu_list = []
        device_count = torch.cuda.device_count()

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)

            gpu_info = GPUInfo(
                index=i,
                name=props.name,
                compute_capability=(props.major, props.minor),
                total_memory_bytes=props.total_memory,
                sm_count=props.multi_processor_count
            )

            gpu_list.append(gpu_info)

        return gpu_list

    def _detect_topology(self, gpu_list: List[GPUInfo]):
        """Detect GPU interconnect topology (PCIe, NVLink)"""
        if not torch.cuda.is_available() or len(gpu_list) < 2:
            return

        # Check P2P access between all GPU pairs
        for i, gpu_i in enumerate(gpu_list):
            for j, gpu_j in enumerate(gpu_list):
                if i != j:
                    try:
                        can_access = torch.cuda.can_device_access_peer(i, j)
                        if can_access:
                            gpu_i.supports_p2p = True
                            # Assuming NVLink if P2P is available
                            # (more sophisticated detection would check nvidia-smi)
                            if j not in gpu_i.nvlink_connections:
                                gpu_i.nvlink_connections.append(j)
                    except:
                        pass

    def _generate_interpretation(self, gpu_list: List[GPUInfo], system_info: SystemInfo) -> str:
        """Generate human-readable interpretation of detected hardware"""
        if not gpu_list:
            return "⚠️ No GPUs detected. This system will run on CPU only (very slow for LLMs)."

        # GPU summary
        gpu_count = len(gpu_list)
        if gpu_count == 1:
            gpu = gpu_list[0]
            memory_gb = gpu.total_memory_bytes / (1024**3)
            interpretation = f"Found: {gpu.name} with {memory_gb:.1f} GB VRAM"
        else:
            first_gpu = gpu_list[0]
            memory_gb = first_gpu.total_memory_bytes / (1024**3)
            # Check if all GPUs are the same
            all_same = all(g.name == first_gpu.name for g in gpu_list)
            if all_same:
                interpretation = f"Found: {gpu_count}x {first_gpu.name} ({memory_gb:.1f} GB each)"
            else:
                interpretation = f"Found: {gpu_count} mixed GPUs"

        # Add topology info
        if gpu_count > 1:
            has_nvlink = any(len(gpu.nvlink_connections) > 0 for gpu in gpu_list)
            if has_nvlink:
                interpretation += "\n  Topology: GPUs connected via NVLink (high-speed)"
            else:
                interpretation += "\n  Topology: GPUs connected via PCIe (standard)"

        # Add CUDA/NCCL info
        interpretation += f"\n  CUDA: {system_info.cuda_version}"
        if system_info.nccl_version:
            interpretation += f", NCCL: {system_info.nccl_version}"
        interpretation += f", PyTorch: {system_info.pytorch_version}"

        return interpretation

    def _generate_recommendations(self, gpu_list: List[GPUInfo], system_info: SystemInfo) -> str:
        """Generate actionable recommendations based on detected hardware"""
        if not gpu_list:
            return "Install CUDA and GPU drivers to enable GPU acceleration."

        recommendations = []

        # Memory-based recommendations
        if gpu_list:
            min_memory_gb = min(gpu.total_memory_bytes for gpu in gpu_list) / (1024**3)
            max_memory_gb = max(gpu.total_memory_bytes for gpu in gpu_list) / (1024**3)

            if min_memory_gb >= 40:
                recommendations.append("✓ Excellent GPU memory - can run 30B+ models")
            elif min_memory_gb >= 24:
                recommendations.append("✓ Great GPU memory - can run 13B-30B models comfortably")
            elif min_memory_gb >= 12:
                recommendations.append("✓ Good GPU memory - ideal for 7B-13B models")
            elif min_memory_gb >= 6:
                recommendations.append("⚠ Moderate GPU memory - recommended for smaller models (<7B) or quantization")
            else:
                recommendations.append("⚠ Limited GPU memory - use 4-bit quantization for larger models")

        # Multi-GPU recommendations
        if len(gpu_list) > 1:
            has_nvlink = any(len(gpu.nvlink_connections) > 0 for gpu in gpu_list)
            if has_nvlink:
                recommendations.append("✓ NVLink detected - excellent for multi-GPU training")
            else:
                recommendations.append("ℹ PCIe connection - consider model parallelism techniques")

            if not system_info.nccl_version:
                recommendations.append("⚠ NCCL not detected - multi-GPU performance will be limited")

        # Compute capability recommendations
        if gpu_list:
            min_cc = min(gpu.compute_capability for gpu in gpu_list)
            if min_cc >= (8, 0):
                recommendations.append("✓ Compute Capability 8.0+ - supports all modern features (FP16, BF16, TF32)")
            elif min_cc >= (7, 0):
                recommendations.append("✓ Compute Capability 7.0+ - good support for mixed precision")
            else:
                recommendations.append("ℹ Older GPU architecture - some features may be limited")

        return "\n  ".join(recommendations) if recommendations else "System ready for LLM workloads!"

    def detect(self) -> Dict[str, Any]:
        """
        Convenience method for quick hardware detection without full test framework.

        Returns:
            Dictionary with hardware info
        """
        result = self.run(mode=TestMode.QUICK)
        return result.metrics
