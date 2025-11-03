"""
Unit tests for HardwareDetector.

Tests GPU detection, system info, and interpretation logic.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diagnostics.core.hardware_detector import HardwareDetector, GPUInfo, SystemInfo
from diagnostics.core.metrics import TestMode, TestStatus


class TestGPUInfo:
    """Test suite for GPUInfo dataclass"""

    def test_gpu_info_creation(self):
        """Test creating a GPUInfo instance"""
        gpu = GPUInfo(
            index=0,
            name="NVIDIA RTX 3070",
            compute_capability=(8, 6),
            total_memory_bytes=8 * 1024**3
        )

        assert gpu.index == 0
        assert gpu.name == "NVIDIA RTX 3070"
        assert gpu.compute_capability == (8, 6)

    def test_gpu_info_to_dict(self):
        """Test GPU info serialization to dict"""
        gpu = GPUInfo(
            index=0,
            name="NVIDIA RTX 3070",
            compute_capability=(8, 6),
            total_memory_bytes=8 * 1024**3,
            sm_count=48
        )

        gpu_dict = gpu.to_dict()

        assert gpu_dict["index"] == 0
        assert gpu_dict["name"] == "NVIDIA RTX 3070"
        assert gpu_dict["compute_capability"] == "8.6"
        assert "total_memory" in gpu_dict
        assert gpu_dict["sm_count"] == 48

    def test_gpu_info_str_representation(self):
        """Test string representation of GPU"""
        gpu = GPUInfo(
            index=0,
            name="NVIDIA RTX 3070",
            compute_capability=(8, 6),
            total_memory_bytes=8 * 1024**3
        )

        str_repr = str(gpu)
        assert "GPU 0" in str_repr
        assert "NVIDIA RTX 3070" in str_repr
        assert "8.6" in str_repr  # Compute capability

    def test_gpu_info_with_nvlink(self):
        """Test GPU info with NVLink connections"""
        gpu = GPUInfo(
            index=0,
            name="NVIDIA A100",
            compute_capability=(8, 0),
            total_memory_bytes=40 * 1024**3,
            nvlink_connections=[1, 2, 3]
        )

        str_repr = str(gpu)
        assert "NVLink" in str_repr
        assert "[1, 2, 3]" in str_repr

    def test_gpu_info_with_pcie(self):
        """Test GPU info with PCIe details"""
        gpu = GPUInfo(
            index=0,
            name="NVIDIA RTX 3070",
            compute_capability=(8, 6),
            total_memory_bytes=8 * 1024**3,
            pcie_gen=4,
            pcie_width=16
        )

        str_repr = str(gpu)
        assert "PCIe" in str_repr
        assert "Gen 4" in str_repr
        assert "x16" in str_repr


class TestSystemInfo:
    """Test suite for SystemInfo dataclass"""

    def test_system_info_creation(self):
        """Test creating a SystemInfo instance"""
        sys_info = SystemInfo(
            os="Linux",
            os_version="5.15.0",
            python_version="3.10.0",
            pytorch_version="2.0.0",
            cuda_version="12.1"
        )

        assert sys_info.os == "Linux"
        assert sys_info.cuda_version == "12.1"

    def test_system_info_to_dict(self):
        """Test system info serialization"""
        sys_info = SystemInfo(
            os="Linux",
            os_version="5.15.0",
            python_version="3.10.0",
            pytorch_version="2.0.0",
            cuda_version="12.1",
            nccl_version="2.15.5",
            cpu_count=16
        )

        sys_dict = sys_info.to_dict()

        assert sys_dict["os"] == "Linux"
        assert sys_dict["cuda_version"] == "12.1"
        assert sys_dict["nccl_version"] == "2.15.5"
        assert sys_dict["cpu_count"] == 16

    def test_system_info_optional_fields(self):
        """Test system info with optional fields"""
        sys_info = SystemInfo(
            os="Linux",
            os_version="5.15.0",
            python_version="3.10.0",
            pytorch_version="2.0.0",
            cuda_version="12.1"
        )

        sys_dict = sys_info.to_dict()

        # Optional fields should not be in dict if None
        assert "cudnn_version" not in sys_dict or sys_dict.get("cudnn_version") is not None


class TestHardwareDetector:
    """Test suite for HardwareDetector class"""

    def test_initialization(self):
        """Test HardwareDetector initialization"""
        detector = HardwareDetector()

        assert detector.name == "Hardware Detection"
        assert detector.description
        assert detector.metric_type.value == "hardware"

    @pytest.mark.skipif(not hasattr(sys.modules.get('torch', None), 'cuda'),
                       reason="PyTorch with CUDA not available")
    def test_detect_system(self):
        """Test system information detection"""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        detector = HardwareDetector()
        sys_info = detector._detect_system()

        # Check required fields
        assert sys_info.os
        assert sys_info.os_version
        assert sys_info.python_version
        assert sys_info.pytorch_version
        assert sys_info.cuda_version

        # Check types
        assert isinstance(sys_info.os, str)
        assert isinstance(sys_info.python_version, str)

    @pytest.mark.skipif(not hasattr(sys.modules.get('torch', None), 'cuda'),
                       reason="PyTorch with CUDA not available")
    def test_detect_gpus(self):
        """Test GPU detection"""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        detector = HardwareDetector()
        gpu_list = detector._detect_gpus()

        if torch.cuda.device_count() > 0:
            assert len(gpu_list) > 0

            # Check first GPU
            gpu = gpu_list[0]
            assert isinstance(gpu, GPUInfo)
            assert gpu.index == 0
            assert gpu.name
            assert gpu.compute_capability
            assert gpu.total_memory_bytes > 0

    def test_detect_gpus_no_cuda(self):
        """Test GPU detection when CUDA is not available"""
        detector = HardwareDetector()

        with patch('torch.cuda.is_available', return_value=False):
            gpu_list = detector._detect_gpus()
            assert gpu_list == []

    def test_generate_interpretation_no_gpu(self):
        """Test interpretation generation with no GPUs"""
        detector = HardwareDetector()

        sys_info = SystemInfo(
            os="Linux",
            os_version="5.15.0",
            python_version="3.10.0",
            pytorch_version="2.0.0",
            cuda_version="N/A"
        )

        interpretation = detector._generate_interpretation([], sys_info)

        assert "No GPUs detected" in interpretation or "CPU only" in interpretation

    def test_generate_interpretation_single_gpu(self):
        """Test interpretation with single GPU"""
        detector = HardwareDetector()

        gpu = GPUInfo(
            index=0,
            name="NVIDIA RTX 3070 Ti",
            compute_capability=(8, 6),
            total_memory_bytes=8 * 1024**3
        )

        sys_info = SystemInfo(
            os="Linux",
            os_version="5.15.0",
            python_version="3.10.0",
            pytorch_version="2.0.0",
            cuda_version="12.1"
        )

        interpretation = detector._generate_interpretation([gpu], sys_info)

        assert "RTX 3070 Ti" in interpretation
        assert "8.0 GB" in interpretation or "8 GB" in interpretation
        assert "CUDA" in interpretation

    def test_generate_interpretation_multi_gpu(self):
        """Test interpretation with multiple GPUs"""
        detector = HardwareDetector()

        gpus = [
            GPUInfo(
                index=0,
                name="NVIDIA RTX 3090",
                compute_capability=(8, 6),
                total_memory_bytes=24 * 1024**3
            ),
            GPUInfo(
                index=1,
                name="NVIDIA RTX 3090",
                compute_capability=(8, 6),
                total_memory_bytes=24 * 1024**3
            )
        ]

        sys_info = SystemInfo(
            os="Linux",
            os_version="5.15.0",
            python_version="3.10.0",
            pytorch_version="2.0.0",
            cuda_version="12.1",
            nccl_version="2.15.5"
        )

        interpretation = detector._generate_interpretation(gpus, sys_info)

        assert "2x" in interpretation or "2 " in interpretation
        assert "RTX 3090" in interpretation

    def test_generate_recommendations_high_memory(self):
        """Test recommendations for high-memory GPU"""
        detector = HardwareDetector()

        gpus = [
            GPUInfo(
                index=0,
                name="NVIDIA A100",
                compute_capability=(8, 0),
                total_memory_bytes=40 * 1024**3
            )
        ]

        sys_info = SystemInfo(
            os="Linux",
            os_version="5.15.0",
            python_version="3.10.0",
            pytorch_version="2.0.0",
            cuda_version="12.1"
        )

        recommendations = detector._generate_recommendations(gpus, sys_info)

        assert "30B" in recommendations or "Excellent" in recommendations

    def test_generate_recommendations_low_memory(self):
        """Test recommendations for low-memory GPU"""
        detector = HardwareDetector()

        gpus = [
            GPUInfo(
                index=0,
                name="NVIDIA GTX 1060",
                compute_capability=(6, 1),
                total_memory_bytes=6 * 1024**3
            )
        ]

        sys_info = SystemInfo(
            os="Linux",
            os_version="5.15.0",
            python_version="3.10.0",
            pytorch_version="2.0.0",
            cuda_version="11.8"
        )

        recommendations = detector._generate_recommendations(gpus, sys_info)

        assert "quantization" in recommendations.lower() or "smaller" in recommendations.lower()

    def test_generate_recommendations_multi_gpu_no_nccl(self):
        """Test recommendations for multi-GPU without NCCL"""
        detector = HardwareDetector()

        gpus = [
            GPUInfo(index=0, name="RTX 3090", compute_capability=(8, 6), total_memory_bytes=24 * 1024**3),
            GPUInfo(index=1, name="RTX 3090", compute_capability=(8, 6), total_memory_bytes=24 * 1024**3)
        ]

        sys_info = SystemInfo(
            os="Linux",
            os_version="5.15.0",
            python_version="3.10.0",
            pytorch_version="2.0.0",
            cuda_version="12.1",
            nccl_version=None  # No NCCL
        )

        recommendations = detector._generate_recommendations(gpus, sys_info)

        assert "NCCL" in recommendations

    @pytest.mark.skipif(not hasattr(sys.modules.get('torch', None), 'cuda'),
                       reason="PyTorch with CUDA not available")
    def test_run_quick_mode(self):
        """Test running detector in quick mode"""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        detector = HardwareDetector()
        result = detector.run(mode=TestMode.QUICK)

        assert result.test_name == "Hardware Detection"
        assert result.status in [TestStatus.PASSED, TestStatus.WARNING]
        assert "gpu_count" in result.metrics
        assert "system" in result.metrics

    def test_detect_convenience_method(self):
        """Test the convenience detect() method"""
        detector = HardwareDetector()

        with patch.object(detector, 'run') as mock_run:
            # Mock the result
            from diagnostics.core.metrics import TestResult
            mock_result = TestResult(
                test_name="Hardware Detection",
                status=TestStatus.PASSED,
                metrics={"gpu_count": 1, "test": "data"}
            )
            mock_run.return_value = mock_result

            metrics = detector.detect()

            # Should call run() with QUICK mode
            mock_run.assert_called_once_with(mode=TestMode.QUICK)

            # Should return just the metrics
            assert metrics == {"gpu_count": 1, "test": "data"}

    def test_run_handles_errors_gracefully(self):
        """Test that run() handles errors gracefully"""
        detector = HardwareDetector()

        with patch.object(detector, '_detect_system', side_effect=Exception("Test error")):
            result = detector.run(mode=TestMode.QUICK)

            assert result.status == TestStatus.FAILED
            assert result.error_message == "Test error"

    def test_compute_capability_recommendations(self):
        """Test recommendations based on compute capability"""
        detector = HardwareDetector()

        # Modern GPU (CC 8.0+)
        modern_gpu = [GPUInfo(
            index=0, name="RTX 3090", compute_capability=(8, 0),
            total_memory_bytes=24 * 1024**3
        )]
        sys_info = SystemInfo(os="Linux", os_version="", python_version="",
                              pytorch_version="", cuda_version="12.1")

        recommendations = detector._generate_recommendations(modern_gpu, sys_info)
        assert "8.0+" in recommendations or "modern" in recommendations.lower()

        # Older GPU (CC 6.x)
        old_gpu = [GPUInfo(
            index=0, name="GTX 1080", compute_capability=(6, 1),
            total_memory_bytes=8 * 1024**3
        )]

        recommendations = detector._generate_recommendations(old_gpu, sys_info)
        assert "limited" in recommendations.lower() or "older" in recommendations.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
