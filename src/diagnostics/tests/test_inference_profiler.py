"""
Unit tests for InferenceProfiler.

Tests real inference profiling, throughput measurement, and latency analysis.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diagnostics.core.inference_profiler import InferenceProfiler
from diagnostics.core.metrics import TestMode, TestStatus


class TestInferenceProfilerInit:
    """Test suite for InferenceProfiler initialization"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_initialization_with_cuda(self):
        """Test InferenceProfiler initialization when CUDA is available"""
        profiler = InferenceProfiler(device="cuda:0")

        assert profiler.name == "Real Inference Profiler"
        assert profiler.description
        assert profiler.metric_type.value == "inference"
        assert profiler.device == "cuda:0"
        assert profiler.device_props is not None
        assert profiler.model_loader is not None

    def test_initialization_without_cuda(self):
        """Test InferenceProfiler raises error when CUDA unavailable"""
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                InferenceProfiler()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_initialization_custom_device(self):
        """Test initialization with custom device"""
        profiler = InferenceProfiler(device="cuda:0")
        assert profiler.device == "cuda:0"


class TestWarmup:
    """Test suite for model warmup functionality"""

    @pytest.fixture
    def profiler(self):
        """Create profiler with mocked CUDA"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties'):
                profiler = InferenceProfiler(device="cuda:0")
                return profiler

    def test_warmup_calls_generate(self, profiler):
        """Test warmup runs model.generate correct number of times"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock tokenizer to return an object with .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = mock_inputs
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])

        with patch('torch.cuda.synchronize'):
            profiler._warmup(mock_model, mock_tokenizer, num_iterations=3)

        # Should call generate 3 times
        assert mock_model.generate.call_count == 3

    def test_warmup_synchronizes(self, profiler):
        """Test warmup synchronizes CUDA"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock tokenizer to return an object with .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = mock_inputs

        with patch('torch.cuda.synchronize') as mock_sync:
            profiler._warmup(mock_model, mock_tokenizer)

        # Should call synchronize at end
        mock_sync.assert_called()


class TestThroughputBenchmark:
    """Test suite for throughput benchmarking"""

    @pytest.fixture
    def profiler(self):
        """Create profiler with mocked CUDA"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties'):
                return InferenceProfiler(device="cuda:0")

    def test_throughput_quick_mode(self, profiler):
        """Test throughput measures batch size 1 only in quick mode"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(to=lambda x: {"input_ids": torch.tensor([[1, 2, 3]])})
        mock_model.generate.return_value = torch.tensor([[1] * 100])

        with patch('torch.cuda.synchronize'):
            with patch('time.perf_counter', side_effect=[0.0, 1.0] * 10):  # 1 second per run
                metrics = profiler._benchmark_throughput(mock_model, mock_tokenizer, TestMode.QUICK)

        # Quick mode should only test batch size 1
        assert "throughput_batch1_tokens_per_sec" in metrics
        assert "throughput_batch4_tokens_per_sec" not in metrics
        assert "throughput_batch8_tokens_per_sec" not in metrics

    def test_throughput_deep_mode(self, profiler):
        """Test throughput measures multiple batch sizes in deep mode"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(to=lambda x: {"input_ids": torch.tensor([[1, 2, 3]])})
        mock_model.generate.return_value = torch.tensor([[1] * 100])

        with patch('torch.cuda.synchronize'):
            with patch('time.perf_counter', side_effect=[0.0, 1.0] * 50):  # Mock timing
                metrics = profiler._benchmark_throughput(mock_model, mock_tokenizer, TestMode.DEEP)

        # Deep mode should test batch sizes 1, 4, 8
        assert "throughput_batch1_tokens_per_sec" in metrics
        assert "throughput_batch4_tokens_per_sec" in metrics
        assert "throughput_batch8_tokens_per_sec" in metrics

    def test_throughput_calculation(self, profiler):
        """Test throughput calculation is correct"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(to=lambda x: {"input_ids": torch.tensor([[1, 2, 3]])})
        mock_model.generate.return_value = torch.tensor([[1] * 100])

        # Mock timing: each run takes exactly 1 second
        timing_sequence = []
        for _ in range(5 + 1):  # 5 runs + 1 warmup
            timing_sequence.extend([0.0, 1.0])

        with patch('torch.cuda.synchronize'):
            with patch('time.perf_counter', side_effect=timing_sequence):
                metrics = profiler._benchmark_throughput(mock_model, mock_tokenizer, TestMode.QUICK)

        # 100 tokens per run, 5 runs, 1 second each = 100 tokens/sec
        assert metrics["tokens_per_sec"] == 100.0

    def test_throughput_latency_per_token(self, profiler):
        """Test latency per token is calculated"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(to=lambda x: {"input_ids": torch.tensor([[1, 2, 3]])})
        mock_model.generate.return_value = torch.tensor([[1] * 100])

        with patch('torch.cuda.synchronize'):
            with patch('time.perf_counter', side_effect=[0.0, 1.0] * 10):
                metrics = profiler._benchmark_throughput(mock_model, mock_tokenizer, TestMode.QUICK)

        # Should calculate ms per token
        assert "latency_batch1_ms_per_token" in metrics
        assert metrics["latency_batch1_ms_per_token"] > 0


class TestLatencyBenchmark:
    """Test suite for latency benchmarking"""

    @pytest.fixture
    def profiler(self):
        """Create profiler with mocked CUDA"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties'):
                return InferenceProfiler(device="cuda:0")

    def test_latency_measures_ttft(self, profiler):
        """Test latency benchmark measures time to first token"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock tokenizer to return input_ids with shape
        mock_inputs = MagicMock()
        mock_inputs.input_ids.shape = (1, 10)  # 10 tokens in prompt
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2]])

        # Mock timing: 5 TTFT measurements (10 calls) + 3 decode measurements (6 calls) = 16 total
        # TTFT: 0.05s each, Decode: 2.5s each (50 tokens * 0.05s)
        timing_sequence = (
            [0.0, 0.05] * 5 +  # 5 TTFT measurements
            [0.0, 2.5] * 3     # 3 decode measurements
        )

        with patch('torch.cuda.synchronize'):
            with patch('time.perf_counter', side_effect=timing_sequence):
                metrics = profiler._benchmark_latency(mock_model, mock_tokenizer)

        assert "time_to_first_token_ms" in metrics
        assert metrics["time_to_first_token_ms"] == 50.0  # 0.05 seconds = 50 ms

    def test_latency_measures_decode(self, profiler):
        """Test latency benchmark measures decode latency"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.input_ids.shape = (1, 10)
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2]])

        # TTFT: 0.05s, Decode: 2.5s (total) - 0.05s (TTFT) = 2.45s for 50 tokens
        timing_sequence = (
            [0.0, 0.05] * 5 +  # 5 TTFT measurements
            [0.0, 2.5] * 3     # 3 decode measurements
        )

        with patch('torch.cuda.synchronize'):
            with patch('time.perf_counter', side_effect=timing_sequence):
                metrics = profiler._benchmark_latency(mock_model, mock_tokenizer)

        assert "decode_latency_ms_per_token" in metrics
        assert metrics["decode_latency_ms_per_token"] > 0

    def test_latency_calculates_prefill_tokens_per_sec(self, profiler):
        """Test latency benchmark calculates prefill throughput"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.input_ids.shape = (1, 10)  # 10 tokens
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2]])

        # TTFT = 0.05s, 10 tokens = 200 tokens/sec
        timing_sequence = (
            [0.0, 0.05] * 5 +  # 5 TTFT measurements
            [0.0, 2.5] * 3     # 3 decode measurements
        )

        with patch('torch.cuda.synchronize'):
            with patch('time.perf_counter', side_effect=timing_sequence):
                metrics = profiler._benchmark_latency(mock_model, mock_tokenizer)

        assert "prefill_tokens_per_sec" in metrics
        assert metrics["prefill_tokens_per_sec"] == 200.0  # 10 tokens / 0.05s

    def test_latency_tracks_prompt_length(self, profiler):
        """Test latency benchmark tracks prompt length"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.input_ids.shape = (1, 15)  # 15 tokens
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2]])

        timing_sequence = (
            [0.0, 0.05] * 5 +  # 5 TTFT measurements
            [0.0, 2.5] * 3     # 3 decode measurements
        )

        with patch('torch.cuda.synchronize'):
            with patch('time.perf_counter', side_effect=timing_sequence):
                metrics = profiler._benchmark_latency(mock_model, mock_tokenizer)

        assert "prompt_length_tokens" in metrics
        assert metrics["prompt_length_tokens"] == 15


class TestMemoryBenchmark:
    """Test suite for memory usage benchmarking"""

    @pytest.fixture
    def profiler(self):
        """Create profiler with mocked CUDA"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties'):
                return InferenceProfiler(device="cuda:0")

    def test_memory_default_sequence_lengths(self, profiler):
        """Test memory benchmark uses default sequence lengths"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.input_ids.shape = (1, 128)
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2]])

        with patch('torch.cuda.reset_peak_memory_stats'):
            with patch('torch.cuda.memory_allocated', return_value=1000 * 1024**2):  # 1000 MB
                with patch('torch.cuda.max_memory_allocated', return_value=1200 * 1024**2):  # 1200 MB
                    with patch('torch.cuda.empty_cache'):
                        metrics = profiler._benchmark_memory_usage(mock_model, mock_tokenizer)

        # Should test default lengths: 128, 512, 1024, 2048
        assert "memory_seq128_peak_mb" in metrics
        assert "memory_seq512_peak_mb" in metrics
        assert "memory_seq1024_peak_mb" in metrics
        assert "memory_seq2048_peak_mb" in metrics

    def test_memory_custom_sequence_lengths(self, profiler):
        """Test memory benchmark with custom sequence lengths"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.input_ids.shape = (1, 64)
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2]])

        with patch('torch.cuda.reset_peak_memory_stats'):
            with patch('torch.cuda.memory_allocated', return_value=500 * 1024**2):
                with patch('torch.cuda.max_memory_allocated', return_value=600 * 1024**2):
                    with patch('torch.cuda.empty_cache'):
                        metrics = profiler._benchmark_memory_usage(
                            mock_model, mock_tokenizer, sequence_lengths=[64, 256]
                        )

        assert "memory_seq64_peak_mb" in metrics
        assert "memory_seq256_peak_mb" in metrics
        assert "memory_seq1024_peak_mb" not in metrics

    def test_memory_tracks_peak_and_active(self, profiler):
        """Test memory benchmark tracks peak and active memory"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.input_ids.shape = (1, 128)
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2]])

        with patch('torch.cuda.reset_peak_memory_stats'):
            with patch('torch.cuda.memory_allocated', return_value=1000 * 1024**2):
                with patch('torch.cuda.max_memory_allocated', return_value=1200 * 1024**2):
                    with patch('torch.cuda.empty_cache'):
                        metrics = profiler._benchmark_memory_usage(
                            mock_model, mock_tokenizer, sequence_lengths=[128]
                        )

        assert "memory_seq128_peak_mb" in metrics
        assert "memory_seq128_active_mb" in metrics
        assert metrics["memory_seq128_peak_mb"] == 1200.0
        assert metrics["memory_seq128_active_mb"] == 1000.0

    def test_memory_handles_oom(self, profiler):
        """Test memory benchmark handles OOM gracefully"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.input_ids.shape = (1, 128)
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        # First sequence works, second OOMs
        mock_model.generate.side_effect = [
            torch.tensor([[1, 2]]),
            RuntimeError("CUDA out of memory")
        ]

        with patch('torch.cuda.reset_peak_memory_stats'):
            with patch('torch.cuda.memory_allocated', return_value=1000 * 1024**2):
                with patch('torch.cuda.max_memory_allocated', return_value=1200 * 1024**2):
                    with patch('torch.cuda.empty_cache'):
                        metrics = profiler._benchmark_memory_usage(
                            mock_model, mock_tokenizer, sequence_lengths=[128, 2048]
                        )

        # First should succeed
        assert "memory_seq128_peak_mb" in metrics
        # Second should show OOM
        assert metrics.get("memory_seq2048_status") == "OOM"


class TestInterpretation:
    """Test suite for interpretation generation"""

    @pytest.fixture
    def profiler(self):
        """Create profiler with mocked CUDA"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties'):
                return InferenceProfiler(device="cuda:0")

    def test_interpretation_includes_model_info(self, profiler):
        """Test interpretation includes model name and parameters"""
        metrics = {
            "model_name": "Qwen2.5-3B",
            "model_params_billions": 3.1,
            "tokens_per_sec": 45.5
        }

        interpretation = profiler._generate_interpretation(metrics)

        assert "Qwen2.5-3B" in interpretation
        assert "3.1B" in interpretation

    def test_interpretation_categorizes_performance(self, profiler):
        """Test interpretation provides performance categories"""
        # Excellent performance
        metrics = {"model_name": "Test", "model_params_billions": 1.0, "tokens_per_sec": 60.0}
        interpretation = profiler._generate_interpretation(metrics)
        assert "Excellent" in interpretation or "instant" in interpretation.lower()

        # Good performance
        metrics = {"model_name": "Test", "model_params_billions": 1.0, "tokens_per_sec": 35.0}
        interpretation = profiler._generate_interpretation(metrics)
        assert "Great" in interpretation or "interactive" in interpretation.lower()

        # Adequate performance
        metrics = {"model_name": "Test", "model_params_billions": 1.0, "tokens_per_sec": 10.0}
        interpretation = profiler._generate_interpretation(metrics)
        assert "Adequate" in interpretation or "basic" in interpretation.lower()

    def test_interpretation_includes_latency(self, profiler):
        """Test interpretation includes latency information"""
        metrics = {
            "model_name": "Test",
            "model_params_billions": 1.0,
            "tokens_per_sec": 30.0,
            "time_to_first_token_ms": 45.0
        }

        interpretation = profiler._generate_interpretation(metrics)

        assert "45 ms" in interpretation or "Time to First Token" in interpretation


class TestRecommendations:
    """Test suite for recommendation generation"""

    @pytest.fixture
    def profiler(self):
        """Create profiler with mocked CUDA"""
        with patch('torch.cuda.is_available', return_value=True):
            mock_props = MagicMock()
            mock_props.total_memory = 24 * 1024**3  # 24GB
            with patch('torch.cuda.get_device_properties', return_value=mock_props):
                return InferenceProfiler(device="cuda:0")

    def test_recommendations_excellent_performance(self, profiler):
        """Test recommendations for excellent performance"""
        metrics = {"tokens_per_sec": 50.0, "model_name": "Qwen2.5-3B"}

        recommendations = profiler._generate_recommendations(metrics)

        assert "Excellent" in recommendations or "✓" in recommendations

    def test_recommendations_suggest_improvements(self, profiler):
        """Test recommendations suggest improvements for low performance"""
        metrics = {"tokens_per_sec": 15.0, "model_name": "Qwen2.5-3B"}

        recommendations = profiler._generate_recommendations(metrics)

        assert "compile" in recommendations.lower() or "throttling" in recommendations.lower()

    def test_recommendations_suggest_larger_models(self, profiler):
        """Test recommendations suggest larger models when using small ones"""
        metrics = {"tokens_per_sec": 30.0, "model_name": "Qwen2.5-0.5B"}

        recommendations = profiler._generate_recommendations(metrics)

        assert "larger" in recommendations.lower() or "3B" in recommendations or "7B" in recommendations

    def test_recommendations_based_on_vram(self, profiler):
        """Test recommendations consider available VRAM"""
        metrics = {"tokens_per_sec": 30.0, "model_name": "Qwen2.5-3B"}

        # Profiler has 24GB VRAM (from fixture)
        recommendations = profiler._generate_recommendations(metrics)

        assert "7B" in recommendations or "13B" in recommendations


class TestSyntheticComparison:
    """Test suite for synthetic benchmark comparison"""

    @pytest.fixture
    def profiler(self):
        """Create profiler with mocked CUDA"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties'):
                return InferenceProfiler(device="cuda:0")

    def test_comparison_calculates_real_tflops(self, profiler):
        """Test comparison calculates real TFLOPS from inference"""
        inference_metrics = {
            "tokens_per_sec": 30.0,
            "model_params_billions": 3.0
        }
        compute_metrics = {
            "fp16_tflops": 40.0
        }

        result = profiler.compare_to_synthetic(inference_metrics, compute_metrics)

        assert "real_inference_tflops" in result
        assert result["real_inference_tflops"] > 0

    def test_comparison_calculates_efficiency(self, profiler):
        """Test comparison calculates efficiency percentage"""
        inference_metrics = {
            "tokens_per_sec": 30.0,
            "model_params_billions": 3.0
        }
        compute_metrics = {
            "fp16_tflops": 40.0
        }

        result = profiler.compare_to_synthetic(inference_metrics, compute_metrics)

        assert "efficiency_percent" in result
        assert 0 < result["efficiency_percent"] <= 100

    def test_comparison_provides_explanation(self, profiler):
        """Test comparison provides educational explanation"""
        inference_metrics = {
            "tokens_per_sec": 30.0,
            "model_params_billions": 3.0
        }
        compute_metrics = {
            "fp16_tflops": 40.0
        }

        result = profiler.compare_to_synthetic(inference_metrics, compute_metrics)

        assert "explanation" in result
        assert len(result["explanation"]) > 20  # Substantial explanation

    def test_comparison_handles_missing_metrics(self, profiler):
        """Test comparison handles missing metrics gracefully"""
        inference_metrics = {}
        compute_metrics = {}

        result = profiler.compare_to_synthetic(inference_metrics, compute_metrics)

        # Should return empty or minimal result, not crash
        assert isinstance(result, dict)


class TestRunIntegration:
    """Integration tests for full run() workflow"""

    @pytest.fixture
    def profiler(self):
        """Create profiler with mocked CUDA"""
        with patch('torch.cuda.is_available', return_value=True):
            mock_props = MagicMock()
            mock_props.total_memory = 8 * 1024**3  # 8GB
            with patch('torch.cuda.get_device_properties', return_value=mock_props):
                return InferenceProfiler(device="cuda:0")

    def test_run_skips_on_insufficient_vram(self, profiler):
        """Test run() skips when insufficient VRAM"""
        # Mock auto_select to return None (insufficient VRAM)
        with patch.object(profiler.model_loader, 'auto_select_model', return_value=None):
            result = profiler.run(mode=TestMode.QUICK, model_name=None)

        assert result.status == TestStatus.SKIPPED
        assert "Insufficient VRAM" in result.interpretation

    def test_run_uses_cached_model(self, profiler):
        """Test run() uses cached model without download"""
        # Mock everything
        with patch.object(profiler.model_loader, 'auto_select_model', return_value="Qwen2.5-1.5B"):
            with patch.object(profiler.model_loader, 'get_model_info') as mock_info:
                mock_info.return_value = MagicMock(
                    params_billions=1.5, size_gb=3.0, name="Qwen2.5-1.5B"
                )
                with patch.object(profiler.model_loader, 'is_model_cached', return_value=True):
                    with patch.object(profiler.model_loader, 'load_model_and_tokenizer') as mock_load:
                        mock_model = MagicMock()
                        mock_tokenizer = MagicMock()
                        mock_tokenizer.eos_token_id = 0

                        # Create mock inputs object
                        mock_inputs = MagicMock()
                        mock_inputs.input_ids.shape = (1, 10)
                        mock_inputs.to.return_value = mock_inputs
                        mock_tokenizer.return_value = mock_inputs

                        mock_model.generate.return_value = torch.tensor([[1, 2]])
                        mock_load.return_value = (mock_model, mock_tokenizer)

                        with patch.object(profiler.model_loader, 'download_model') as mock_download:
                            with patch('torch.cuda.synchronize'):
                                with patch('torch.cuda.empty_cache'):
                                    # Create realistic timing sequence:
                                    # Warmup: 3 runs * 2 = 6 calls
                                    # Throughput: 6 runs (1 warmup + 5 measure) * 2 = 12 calls
                                    # Latency TTFT: 5 runs * 2 = 10 calls
                                    # Latency decode: 3 runs * 2 = 6 calls
                                    # Total: 34 calls
                                    timing_sequence = (
                                        [0.0, 0.5] * 3 +    # Warmup
                                        [0.0, 1.0] * 6 +    # Throughput
                                        [0.0, 0.05] * 5 +   # TTFT (50ms)
                                        [0.0, 2.5] * 3      # Decode (2.5s total, 2.45s after TTFT)
                                    )

                                    with patch('time.perf_counter', side_effect=timing_sequence):
                                        with patch('torch.cuda.reset_peak_memory_stats'):
                                            with patch('torch.cuda.memory_allocated', return_value=1000 * 1024**2):
                                                with patch('torch.cuda.max_memory_allocated', return_value=1200 * 1024**2):
                                                    result = profiler.run(mode=TestMode.QUICK, ask_permission=False)

        # Should not download if cached
        mock_download.assert_not_called()

        # Should succeed
        assert result.status == TestStatus.PASSED

    def test_run_handles_errors(self, profiler):
        """Test run() handles errors gracefully"""
        # Mock model selection to raise error
        with patch.object(profiler.model_loader, 'auto_select_model', side_effect=Exception("Test error")):
            result = profiler.run(mode=TestMode.QUICK)

        assert result.status == TestStatus.FAILED
        assert result.error_message == "Test error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
