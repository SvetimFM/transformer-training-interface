"""
Backend Detection Module

Detects the compute backend (CUDA/ROCm/CPU) and provides unified interface
for hardware-agnostic diagnostic code.

This is the foundation for cross-platform support.
"""

import torch
import platform
import subprocess
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class ComputeBackend(Enum):
    """Supported compute backends"""
    CUDA_NVIDIA = "cuda"      # NVIDIA GPUs via CUDA
    ROCM_AMD = "rocm"         # AMD GPUs via ROCm
    CPU = "cpu"               # CPU-only mode
    METAL_APPLE = "mps"       # Apple Silicon (future)


@dataclass
class BackendInfo:
    """Information about detected compute backend"""
    backend: ComputeBackend
    version: str
    device_count: int
    device_name: Optional[str] = None
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "backend": self.backend.value,
            "version": self.version,
            "device_count": self.device_count,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version
        }

    def __str__(self) -> str:
        """Human-readable representation"""
        if self.backend == ComputeBackend.CPU:
            return f"CPU-only mode (PyTorch {self.version})"

        devices = "device" if self.device_count == 1 else "devices"
        return f"{self.backend.value.upper()}: {self.device_count} {devices} ({self.device_name})"


class BackendDetector:
    """
    Detects compute backend and provides unified interface.

    Usage:
        detector = BackendDetector()
        backend = detector.detect()

        if backend.backend == ComputeBackend.CUDA_NVIDIA:
            # Use CUDA-specific code
        elif backend.backend == ComputeBackend.ROCM_AMD:
            # Use ROCm-specific code
        else:
            # CPU fallback
    """

    @staticmethod
    def detect() -> BackendInfo:
        """
        Detect the compute backend available on this system.

        Detection priority:
        1. Check for CUDA (NVIDIA)
        2. Check for ROCm (AMD)
        3. Check for MPS (Apple Silicon)
        4. Fallback to CPU

        Returns:
            BackendInfo with details about detected backend
        """
        # Check for CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            # Distinguish between CUDA and ROCm
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                return BackendDetector._detect_rocm()
            else:
                return BackendDetector._detect_cuda()

        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return BackendDetector._detect_mps()

        # Fallback to CPU
        return BackendDetector._detect_cpu()

    @staticmethod
    def _detect_cuda() -> BackendInfo:
        """Detect NVIDIA CUDA backend"""
        device_count = torch.cuda.device_count()

        info = BackendInfo(
            backend=ComputeBackend.CUDA_NVIDIA,
            version=torch.version.cuda or "Unknown",
            device_count=device_count
        )

        if device_count > 0:
            # Get info about first device
            props = torch.cuda.get_device_properties(0)
            info.device_name = props.name
            info.compute_capability = f"{props.major}.{props.minor}"

            # Try to get driver version via nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    info.driver_version = result.stdout.strip().split('\n')[0]
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        return info

    @staticmethod
    def _detect_rocm() -> BackendInfo:
        """Detect AMD ROCm backend"""
        device_count = torch.cuda.device_count()  # ROCm uses same API

        info = BackendInfo(
            backend=ComputeBackend.ROCM_AMD,
            version=torch.version.hip or "Unknown",
            device_count=device_count
        )

        if device_count > 0:
            # Get device info
            props = torch.cuda.get_device_properties(0)
            info.device_name = props.name
            info.compute_capability = f"gfx{props.gcnArchName}" if hasattr(props, 'gcnArchName') else "Unknown"

            # Try to get ROCm version via rocm-smi
            try:
                result = subprocess.run(
                    ['rocm-smi', '--showdriverversion'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Parse ROCm version from output
                    for line in result.stdout.split('\n'):
                        if 'Driver version' in line or 'ROCm version' in line:
                            info.driver_version = line.split(':')[-1].strip()
                            break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        return info

    @staticmethod
    def _detect_mps() -> BackendInfo:
        """Detect Apple Metal Performance Shaders backend"""
        return BackendInfo(
            backend=ComputeBackend.METAL_APPLE,
            version=torch.__version__,
            device_count=1,  # Apple Silicon has one unified memory GPU
            device_name="Apple Silicon GPU"
        )

    @staticmethod
    def _detect_cpu() -> BackendInfo:
        """Detect CPU-only mode"""
        import psutil

        cpu_name = None
        try:
            # Try to get CPU model name
            if platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            cpu_name = line.split(':')[1].strip()
                            break
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    cpu_name = result.stdout.strip()
            elif platform.system() == "Windows":
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    cpu_name = result.stdout.split('\n')[1].strip()
        except:
            pass

        return BackendInfo(
            backend=ComputeBackend.CPU,
            version=torch.__version__,
            device_count=psutil.cpu_count(logical=False) or 1,  # Physical cores
            device_name=cpu_name or f"{platform.processor()} ({psutil.cpu_count(logical=False)} cores)"
        )

    @staticmethod
    def get_backend_features(backend: ComputeBackend) -> Dict[str, bool]:
        """
        Get capabilities of a specific backend.

        Returns dict of feature flags for conditional code paths.
        """
        features = {
            'supports_fp16': False,
            'supports_bf16': False,
            'supports_tf32': False,
            'supports_p2p': False,
            'supports_nvlink': False,
            'supports_multi_gpu': False,
            'supports_nccl': False,
            'has_health_monitoring': False,  # NVML/ROCm SMI
        }

        if backend == ComputeBackend.CUDA_NVIDIA:
            features.update({
                'supports_fp16': True,
                'supports_bf16': True,
                'supports_tf32': True,
                'supports_p2p': torch.cuda.device_count() > 1,
                'supports_nvlink': True,  # May or may not have NVLink, but API exists
                'supports_multi_gpu': torch.cuda.device_count() > 1,
                'supports_nccl': True,
                'has_health_monitoring': True,  # NVML
            })

        elif backend == ComputeBackend.ROCM_AMD:
            features.update({
                'supports_fp16': True,
                'supports_bf16': True,
                'supports_tf32': False,  # AMD doesn't have TF32
                'supports_p2p': torch.cuda.device_count() > 1,
                'supports_nvlink': False,  # AMD has Infinity Fabric instead
                'supports_multi_gpu': torch.cuda.device_count() > 1,
                'supports_nccl': True,  # RCCL is API-compatible
                'has_health_monitoring': True,  # rocm-smi
            })

        elif backend == ComputeBackend.METAL_APPLE:
            features.update({
                'supports_fp16': True,
                'supports_bf16': False,
                'supports_multi_gpu': False,  # Apple Silicon is single GPU
            })

        # CPU has minimal features
        return features

    @staticmethod
    def get_recommended_device() -> str:
        """
        Get recommended PyTorch device string based on backend.

        Returns:
            "cuda:0", "cpu", "mps", etc.
        """
        backend = BackendDetector.detect()

        if backend.backend == ComputeBackend.CUDA_NVIDIA:
            return "cuda:0"
        elif backend.backend == ComputeBackend.ROCM_AMD:
            return "cuda:0"  # ROCm uses same API
        elif backend.backend == ComputeBackend.METAL_APPLE:
            return "mps"
        else:
            return "cpu"


def main():
    """CLI tool for testing backend detection"""
    print("=" * 80)
    print("COMPUTE BACKEND DETECTION")
    print("=" * 80)
    print()

    backend = BackendDetector.detect()
    print(f"Detected: {backend}")
    print()

    print("Backend Details:")
    for key, value in backend.to_dict().items():
        print(f"  {key}: {value}")
    print()

    features = BackendDetector.get_backend_features(backend.backend)
    print("Backend Features:")
    for feature, supported in features.items():
        status = "✓" if supported else "✗"
        print(f"  {status} {feature}")
    print()

    print(f"Recommended device string: {BackendDetector.get_recommended_device()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
