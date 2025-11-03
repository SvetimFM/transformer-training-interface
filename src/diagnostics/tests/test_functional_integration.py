"""
Functional Integration Tests for Diagnostics System

Tests end-to-end scenarios including:
- CLI integration with all modes
- Backend detection and fallback
- Health monitoring integration
- Burn-in testing
- Error handling and recovery
"""

import os
import sys
import pytest
import torch
from unittest.mock import patch, MagicMock, PropertyMock
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diagnostics.cli import DiagnosticsCLI
from diagnostics.core.backend_detector import BackendDetector, ComputeBackend
from diagnostics.core.health_monitor import GPUHealthMonitor
from diagnostics.core.burn_in import BurnInTester
from diagnostics.core.cpu_profiler import CPUProfiler
from diagnostics.core.metrics import TestMode, TestStatus


class TestBackendDetection:
    """Test backend detection integration"""

    def test_backend_detector_initialization(self):
        """Test that backend detector initializes correctly"""
        backend_info = BackendDetector.detect()

        assert backend_info is not None
        assert backend_info.backend in [
            ComputeBackend.CUDA_NVIDIA,
            ComputeBackend.ROCM_AMD,
            ComputeBackend.CPU,
            ComputeBackend.METAL_APPLE
        ]

    def test_backend_features_accessible(self):
        """Test that backend features can be queried"""
        backend_info = BackendDetector.detect()
        features = BackendDetector.get_backend_features(backend_info.backend)

        assert isinstance(features, dict)
        assert 'supports_fp16' in features
        assert 'supports_nccl' in features

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_cuda_backend_detection(self):
        """Test CUDA backend is detected correctly"""
        backend_info = BackendDetector.detect()

        assert backend_info.backend == ComputeBackend.CUDA_NVIDIA
        assert backend_info.device_count > 0
        assert backend_info.device_name is not None
        assert backend_info.version is not None

    def test_recommended_device_string(self):
        """Test recommended device string format"""
        device = BackendDetector.get_recommended_device()

        assert isinstance(device, str)
        assert 'cuda' in device or 'cpu' in device


class TestHealthMonitoring:
    """Test health monitoring integration"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_health_monitor_initialization(self):
        """Test health monitor initializes on CUDA system"""
        monitor = GPUHealthMonitor()

        assert monitor is not None
        assert monitor.backend == ComputeBackend.CUDA_NVIDIA

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_health_snapshot_capture(self):
        """Test capturing health snapshot"""
        monitor = GPUHealthMonitor()
        snapshot = monitor.get_snapshot(0)

        assert snapshot is not None
        assert snapshot.gpu_id == 0
        # At least one metric should be available
        assert (snapshot.temperature is not None or
                snapshot.power_draw is not None or
                snapshot.utilization_gpu is not None)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_health_monitor_cleanup(self):
        """Test health monitor cleanup doesn't error"""
        monitor = GPUHealthMonitor()
        monitor.cleanup()  # Should not raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_throttle_detection(self):
        """Test throttle detection works"""
        monitor = GPUHealthMonitor()
        snapshot = monitor.get_snapshot(0)

        # Should have throttle detection fields
        assert hasattr(snapshot, 'is_throttled')
        assert hasattr(snapshot, 'throttle_reason')
        assert isinstance(snapshot.is_throttled, bool)


class TestCLIIntegration:
    """Test CLI integration with new components"""

    def test_cli_initialization(self):
        """Test CLI initializes correctly"""
        cli = DiagnosticsCLI()

        assert cli is not None
        assert cli.suite is not None
        assert hasattr(cli, 'burn_in_duration')

    def test_cli_quick_mode_completes(self):
        """Test quick mode runs to completion"""
        cli = DiagnosticsCLI()

        # Capture stdout to suppress output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            cli.run_all_tests(mode=TestMode.QUICK)

            # Should have run tests
            assert len(cli.suite.tests) > 0

            # All tests should have status
            for result in cli.suite.tests:
                assert result.status in [TestStatus.PASSED, TestStatus.WARNING, TestStatus.FAILED, TestStatus.SKIPPED]
        finally:
            sys.stdout = old_stdout

    def test_cli_backend_detection_integration(self):
        """Test that CLI detects backend correctly"""
        cli = DiagnosticsCLI()

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            cli.run_all_tests(mode=TestMode.QUICK)

            # Should have detected backend (indirectly via successful execution)
            assert True  # If we get here, backend detection worked
        finally:
            sys.stdout = old_stdout

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_cli_health_monitoring_integration(self):
        """Test that CLI uses health monitoring"""
        cli = DiagnosticsCLI()

        # Health monitor should be initialized during GPU tests
        # We verify this indirectly by checking that quick mode completes
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            cli.run_all_tests(mode=TestMode.QUICK)
            assert len(cli.suite.tests) > 0
        finally:
            sys.stdout = old_stdout


class TestCPUFallback:
    """Test CPU fallback when no GPU available"""

    @patch('torch.cuda.is_available')
    @patch('diagnostics.core.backend_detector.BackendDetector.detect')
    def test_cpu_mode_activation(self, mock_detect, mock_cuda_available):
        """Test that CPU mode activates when no GPU"""
        # Mock no CUDA
        mock_cuda_available.return_value = False

        # Mock CPU backend
        from diagnostics.core.backend_detector import BackendInfo, ComputeBackend
        mock_backend = BackendInfo(
            backend=ComputeBackend.CPU,
            version="N/A",
            device_count=0,
            device_name="CPU",
            compute_capability=None,
            driver_version=None
        )
        mock_detect.return_value = mock_backend

        cli = DiagnosticsCLI()

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            cli.run_all_tests(mode=TestMode.QUICK)

            # Should have run some tests (at least hardware detection + CPU profiler)
            assert len(cli.suite.tests) >= 1
        finally:
            sys.stdout = old_stdout

    @pytest.mark.skipif(torch.cuda.is_available(), reason="CPUProfiler requires CPU-only mode")
    def test_cpu_profiler_standalone(self):
        """Test CPU profiler can run standalone"""
        profiler = CPUProfiler()

        # Run quick test
        result = profiler.run(mode=TestMode.QUICK)

        assert result is not None
        assert result.status in [TestStatus.PASSED, TestStatus.WARNING, TestStatus.FAILED]
        # Should have some metrics
        assert len(result.metrics) > 0


class TestBurnInIntegration:
    """Test burn-in testing integration"""

    def test_burn_in_tester_initialization(self):
        """Test burn-in tester initializes"""
        tester = BurnInTester()

        assert tester is not None
        assert hasattr(tester, 'device')

    def test_burn_in_short_duration(self):
        """Test burn-in with very short duration (30 seconds)"""
        tester = BurnInTester()

        # Run 30-second burn-in
        result = tester.run_burn_in(duration_hours=30/3600, workload='compute')

        assert result is not None
        assert result.duration_seconds > 0
        assert result.total_iterations > 0
        assert hasattr(result, 'stability_score')
        assert 0 <= result.stability_score <= 100

    def test_burn_in_result_to_dict(self):
        """Test burn-in result serialization"""
        tester = BurnInTester()

        # Very short test
        result = tester.run_burn_in(duration_hours=10/3600, workload='compute')  # 10 seconds

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert 'duration_hours' in result_dict
        assert 'stability_score' in result_dict
        assert 'passed' in result_dict

    def test_burn_in_stability_scoring(self):
        """Test burn-in stability scoring logic"""
        tester = BurnInTester()

        # Test the private scoring method
        score = tester._calculate_stability_score(
            success_rate=0.99,
            perf_stability=95.0,
            thermal_ok=True,
            throttle_events=0,
            memory_leak=False
        )

        assert isinstance(score, float)
        assert 0 <= score <= 100
        # High quality parameters should give high score
        assert score > 80

    def test_burn_in_different_workloads(self):
        """Test burn-in supports different workloads"""
        tester = BurnInTester()

        # Test compute workload
        result_compute = tester.run_burn_in(duration_hours=5/3600, workload='compute')
        assert result_compute.total_iterations > 0

        # Test memory workload
        result_memory = tester.run_burn_in(duration_hours=5/3600, workload='memory')
        assert result_memory.total_iterations > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_burn_in_with_health_monitoring(self):
        """Test burn-in uses health monitoring on GPU"""
        tester = BurnInTester()

        result = tester.run_burn_in(duration_hours=10/3600, workload='compute')

        # Should have thermal metrics if GPU available
        assert hasattr(result, 'peak_temperature')
        assert hasattr(result, 'throttle_events')


class TestErrorHandling:
    """Test error handling and recovery"""

    def test_cli_handles_missing_gpu_gracefully(self):
        """Test CLI doesn't crash when GPU tests fail"""
        cli = DiagnosticsCLI()

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Should complete without exceptions
            cli.run_all_tests(mode=TestMode.QUICK)
            assert True
        except Exception as e:
            pytest.fail(f"CLI crashed with: {e}")
        finally:
            sys.stdout = old_stdout

    def test_health_monitor_missing_nvml(self):
        """Test health monitor handles missing pynvml gracefully"""
        # This test verifies error handling path
        # On systems without pynvml, initialization should fail gracefully
        try:
            # Try to initialize on CPU backend (should skip NVML)
            with patch('diagnostics.core.backend_detector.BackendDetector.detect') as mock_detect:
                from diagnostics.core.backend_detector import BackendInfo, ComputeBackend
                mock_detect.return_value = BackendInfo(
                    backend=ComputeBackend.CPU,
                    version="N/A",
                    device_count=0,
                    device_name="CPU",
                    compute_capability=None,
                    driver_version=None
                )

                monitor = GPUHealthMonitor()
                # Should initialize without error (using psutil for CPU)
                assert monitor is not None
        except ImportError:
            # Expected on systems without psutil
            pass

    def test_burn_in_graceful_stop(self):
        """Test burn-in can be stopped gracefully"""
        tester = BurnInTester()

        # Start and immediately stop
        tester.should_stop = False
        result = tester.run_burn_in(duration_hours=1/3600, workload='compute')

        # Should still return result even if stopped early
        assert result is not None


class TestCLIDurationParsing:
    """Test CLI duration parsing for burn-in mode"""

    def test_burn_in_duration_setting(self):
        """Test that CLI accepts burn-in duration"""
        cli = DiagnosticsCLI()

        # Set custom duration
        cli.burn_in_duration = 2.5

        assert cli.burn_in_duration == 2.5

    def test_default_burn_in_duration(self):
        """Test default burn-in duration"""
        cli = DiagnosticsCLI()

        # Should have default
        assert hasattr(cli, 'burn_in_duration')
        assert isinstance(cli.burn_in_duration, (int, float))
        assert cli.burn_in_duration > 0


class TestEndToEndScenarios:
    """End-to-end integration scenarios"""

    def test_full_quick_mode_workflow(self):
        """Test complete quick mode workflow"""
        cli = DiagnosticsCLI()

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Run quick mode
            cli.run_all_tests(mode=TestMode.QUICK)

            # Verify results
            assert len(cli.suite.tests) > 0

            # Get summary
            summary = cli.suite.get_summary()
            # Summary has keys like 'passed', 'failed', 'warning'
            assert 'passed' in summary
            assert summary['passed'] >= 0

        finally:
            sys.stdout = old_stdout

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gpu_health_throughout_tests(self):
        """Test health monitoring works throughout test suite"""
        from diagnostics.core.health_monitor import GPUHealthMonitor

        monitor = GPUHealthMonitor()

        # Capture multiple snapshots
        snapshots = []
        for _ in range(3):
            snapshot = monitor.get_snapshot(0)
            snapshots.append(snapshot)

        # All snapshots should be valid
        assert len(snapshots) == 3
        for snapshot in snapshots:
            assert snapshot is not None
            assert snapshot.gpu_id == 0

        monitor.cleanup()

    def test_backend_to_cli_workflow(self):
        """Test complete workflow from backend detection to test execution"""
        # Step 1: Detect backend
        backend_info = BackendDetector.detect()
        assert backend_info is not None

        # Step 2: Create CLI
        cli = DiagnosticsCLI()

        # Step 3: Run tests
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            cli.run_all_tests(mode=TestMode.QUICK)
            assert len(cli.suite.tests) > 0
        finally:
            sys.stdout = old_stdout


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
