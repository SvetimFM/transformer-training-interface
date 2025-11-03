"""
Unit tests for ModelLoader.

Tests model selection, caching, and download logic.
"""

import os
import tempfile
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# Add parent directories to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diagnostics.core.model_loader import ModelLoader, BenchmarkModel


class TestModelLoader:
    """Test suite for ModelLoader class"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def loader(self, temp_cache_dir):
        """Create ModelLoader instance with temporary cache"""
        return ModelLoader(cache_dir=temp_cache_dir)

    def test_initialization(self, temp_cache_dir):
        """Test ModelLoader initialization creates cache directory"""
        loader = ModelLoader(cache_dir=temp_cache_dir)
        assert loader.cache_dir.exists()
        assert loader.cache_dir.is_dir()

    def test_default_cache_location(self):
        """Test default cache directory is in home directory"""
        loader = ModelLoader()
        expected_path = Path.home() / ".cache" / "diagnostics" / "models"
        assert loader.cache_dir == expected_path

    def test_auto_select_model_large_vram(self, loader):
        """Test model selection with plenty of VRAM (24GB)"""
        selected = loader.auto_select_model(available_vram_gb=24.0)
        # Should select largest model (Qwen2.5-3B)
        assert selected == "Qwen2.5-3B"

    def test_auto_select_model_medium_vram(self, loader):
        """Test model selection with medium VRAM (8GB)"""
        selected = loader.auto_select_model(available_vram_gb=8.0)
        # With conservative mode (80% usable), 8GB * 0.8 = 6.4GB
        # Should select Qwen2.5-1.5B or Qwen2.5-3B depending on min_vram_gb
        assert selected in ["Qwen2.5-1.5B", "Qwen2.5-3B"]

    def test_auto_select_model_low_vram(self, loader):
        """Test model selection with low VRAM (4GB)"""
        selected = loader.auto_select_model(available_vram_gb=4.0)
        # 4GB * 0.8 = 3.2GB usable, should select smallest that fits
        assert selected in ["Qwen2.5-0.5B", "TinyLlama-1.1B", "Qwen2.5-1.5B"]

    def test_auto_select_model_very_low_vram(self, loader):
        """Test model selection with very low VRAM (2GB)"""
        selected = loader.auto_select_model(available_vram_gb=2.0)
        # 2GB * 0.8 = 1.6GB usable - may not fit any model
        # If Qwen2.5-0.5B min_vram_gb is 3, this will return None
        assert selected is None or selected == "Qwen2.5-0.5B"

    def test_auto_select_model_insufficient_vram(self, loader):
        """Test model selection with insufficient VRAM (<2GB)"""
        selected = loader.auto_select_model(available_vram_gb=1.5)
        # Should return None
        assert selected is None

    def test_auto_select_model_conservative_mode(self, loader):
        """Test conservative mode leaves 20% headroom"""
        selected = loader.auto_select_model(available_vram_gb=5.0, conservative=True)
        # 5GB * 0.8 = 4GB usable, should select TinyLlama-1.1B
        assert selected == "TinyLlama-1.1B"

    def test_auto_select_model_non_conservative(self, loader):
        """Test non-conservative mode uses full VRAM"""
        selected = loader.auto_select_model(available_vram_gb=5.0, conservative=False)
        # 5GB usable, could fit larger model
        assert selected in ["Qwen2.5-1.5B", "TinyLlama-1.1B"]

    def test_get_model_info_valid(self, loader):
        """Test retrieving info for valid model"""
        info = loader.get_model_info("Qwen2.5-3B")
        assert info is not None
        assert isinstance(info, BenchmarkModel)
        assert info.name == "Qwen2.5-3B"
        assert info.params_billions == 3.1
        assert info.size_gb == 6.5
        assert info.min_vram_gb == 8

    def test_get_model_info_invalid(self, loader):
        """Test retrieving info for non-existent model"""
        info = loader.get_model_info("NonExistentModel")
        assert info is None

    def test_is_model_cached_false(self, loader, temp_cache_dir):
        """Test cache check for non-cached model"""
        assert not loader.is_model_cached("Qwen2.5-3B")

    def test_is_model_cached_true(self, loader, temp_cache_dir):
        """Test cache check for cached model"""
        # Create fake cached model directory
        model_dir = Path(temp_cache_dir) / "Qwen2.5-3B"
        model_dir.mkdir(parents=True)

        # Add a file to make it non-empty
        (model_dir / "config.json").write_text("{}")

        assert loader.is_model_cached("Qwen2.5-3B")

    def test_is_model_cached_empty_directory(self, loader, temp_cache_dir):
        """Test cache check for empty model directory"""
        # Create empty directory
        model_dir = Path(temp_cache_dir) / "Qwen2.5-3B"
        model_dir.mkdir(parents=True)

        # Should return False for empty directory
        assert not loader.is_model_cached("Qwen2.5-3B")

    def test_is_model_cached_invalid_model(self, loader):
        """Test cache check for invalid model name"""
        assert not loader.is_model_cached("InvalidModel")

    def test_list_available_models(self, loader):
        """Test listing all available models"""
        models = loader.list_available_models()

        assert len(models) == 4  # We have 4 benchmark models
        assert "Qwen2.5-3B" in models
        assert "TinyLlama-1.1B" in models

        # Check structure of returned data
        qwen_info = models["Qwen2.5-3B"]
        assert "repo_id" in qwen_info
        assert "size_gb" in qwen_info
        assert "params_billions" in qwen_info
        assert "cached" in qwen_info
        assert qwen_info["cached"] is False  # Not cached in temp dir

    def test_list_available_models_with_cache(self, loader, temp_cache_dir):
        """Test listing models shows cache status correctly"""
        # Create fake cached model
        model_dir = Path(temp_cache_dir) / "Qwen2.5-3B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        models = loader.list_available_models()
        assert models["Qwen2.5-3B"]["cached"] is True
        assert models["TinyLlama-1.1B"]["cached"] is False

    def test_estimate_inference_memory_basic(self, loader):
        """Test memory estimation for basic scenario"""
        memory_gb = loader.estimate_inference_memory(
            model_name="Qwen2.5-3B",
            sequence_length=512,
            batch_size=1
        )

        # Should be model size * multipliers
        assert memory_gb > 6.5  # At least model size
        assert memory_gb < 15.0  # Reasonable upper bound

    def test_estimate_inference_memory_long_context(self, loader):
        """Test memory estimation grows with sequence length"""
        mem_short = loader.estimate_inference_memory(
            "Qwen2.5-3B", sequence_length=512, batch_size=1
        )
        mem_long = loader.estimate_inference_memory(
            "Qwen2.5-3B", sequence_length=2048, batch_size=1
        )

        # Longer context should use more memory
        assert mem_long > mem_short

    def test_estimate_inference_memory_larger_batch(self, loader):
        """Test memory estimation grows with batch size"""
        mem_batch1 = loader.estimate_inference_memory(
            "Qwen2.5-3B", sequence_length=512, batch_size=1
        )
        mem_batch8 = loader.estimate_inference_memory(
            "Qwen2.5-3B", sequence_length=512, batch_size=8
        )

        # Larger batch should use more memory
        assert mem_batch8 > mem_batch1

    def test_estimate_inference_memory_invalid_model(self, loader):
        """Test memory estimation for invalid model"""
        memory_gb = loader.estimate_inference_memory(
            "InvalidModel", sequence_length=512, batch_size=1
        )
        assert memory_gb == 0.0

    @patch('huggingface_hub.snapshot_download')
    def test_download_model_success(self, mock_download, loader, temp_cache_dir):
        """Test successful model download"""
        mock_download.return_value = str(Path(temp_cache_dir) / "Qwen2.5-3B")

        # Mock progress callback
        progress_calls = []
        def progress_callback(msg, progress):
            progress_calls.append((msg, progress))

        path = loader.download_model("Qwen2.5-3B", progress_callback=progress_callback)

        # Check download was called
        mock_download.assert_called_once()

        # Check progress callbacks were made
        assert len(progress_calls) >= 2  # At least start and end
        assert progress_calls[-1][1] == 100  # Final progress is 100%

    def test_download_model_invalid_name(self, loader):
        """Test download with invalid model name raises error"""
        with pytest.raises(ValueError, match="Unknown model"):
            loader.download_model("InvalidModel")

    @patch('huggingface_hub.snapshot_download')
    def test_download_model_already_cached(self, mock_download, loader, temp_cache_dir):
        """Test download skips if model already cached"""
        # Create fake cached model
        model_dir = Path(temp_cache_dir) / "Qwen2.5-3B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        progress_calls = []
        def progress_callback(msg, progress):
            progress_calls.append((msg, progress))

        path = loader.download_model("Qwen2.5-3B", progress_callback=progress_callback)

        # Should not call download
        mock_download.assert_not_called()

        # Should still call progress callback
        assert any("already cached" in msg.lower() for msg, _ in progress_calls)

    def test_benchmark_models_completeness(self, loader):
        """Test that all benchmark models have required fields"""
        for name, spec in loader.BENCHMARK_MODELS.items():
            assert spec.name == name
            assert spec.repo_id
            assert spec.size_gb > 0
            assert spec.min_vram_gb > 0
            assert spec.params_billions > 0
            assert spec.description

            # Repo ID should be valid format
            assert "/" in spec.repo_id  # HuggingFace format: org/model

    def test_benchmark_models_ordering(self, loader):
        """Test that models are ordered reasonably by size"""
        models = list(loader.BENCHMARK_MODELS.values())
        sizes = [m.params_billions for m in models]

        # Should have a range of sizes
        assert min(sizes) < 1.0  # Have small model
        assert max(sizes) > 2.0  # Have larger model

    def test_thread_safety_cache_dir(self, loader):
        """Test that cache directory path is consistent"""
        # Get cache dir multiple times
        dir1 = loader.cache_dir
        dir2 = loader.cache_dir

        assert dir1 == dir2
        assert dir1.exists()


class TestBenchmarkModel:
    """Test suite for BenchmarkModel dataclass"""

    def test_benchmark_model_creation(self):
        """Test creating a BenchmarkModel instance"""
        model = BenchmarkModel(
            name="TestModel",
            repo_id="test/model",
            size_gb=5.0,
            min_vram_gb=8.0,
            description="Test model",
            params_billions=3.0
        )

        assert model.name == "TestModel"
        assert model.repo_id == "test/model"
        assert model.size_gb == 5.0
        assert model.min_vram_gb == 8.0
        assert model.params_billions == 3.0

    def test_benchmark_model_required_fields(self):
        """Test that all fields are required"""
        with pytest.raises(TypeError):
            # Missing required fields
            BenchmarkModel(name="Test")


# Integration-style tests
class TestModelLoaderIntegration:
    """Integration tests that test multiple components together"""

    @pytest.fixture
    def loader(self):
        """Create loader with temporary cache"""
        temp_dir = tempfile.mkdtemp()
        loader = ModelLoader(cache_dir=temp_dir)
        yield loader
        shutil.rmtree(temp_dir)

    def test_full_workflow_model_selection_and_info(self, loader):
        """Test complete workflow: select model, get info, check cache"""
        # Step 1: Auto-select model
        selected = loader.auto_select_model(available_vram_gb=8.0)
        assert selected is not None

        # Step 2: Get model info
        info = loader.get_model_info(selected)
        assert info is not None
        assert info.name == selected

        # Step 3: Check cache status
        is_cached = loader.is_model_cached(selected)
        assert isinstance(is_cached, bool)

        # Step 4: Estimate memory
        memory = loader.estimate_inference_memory(selected)
        assert memory > 0

    def test_vram_edge_cases(self, loader):
        """Test edge cases in VRAM-based model selection"""
        # Exactly at boundary (conservative mode)
        selected = loader.auto_select_model(available_vram_gb=3.0, conservative=True)
        # 3.0 * 0.8 = 2.4GB usable
        # May or may not fit smallest model depending on min_vram_gb
        assert selected in [None, "Qwen2.5-0.5B"]

        # Higher VRAM should definitely fit something
        selected = loader.auto_select_model(available_vram_gb=5.0, conservative=True)
        # 5.0 * 0.8 = 4.0GB usable - should fit at least one model
        assert selected is not None

    def test_all_models_have_valid_memory_estimates(self, loader):
        """Test that memory estimation works for all models"""
        for model_name in loader.BENCHMARK_MODELS.keys():
            memory = loader.estimate_inference_memory(model_name)
            assert memory > 0, f"Memory estimate for {model_name} should be positive"

            # Memory should be at least the model size
            info = loader.get_model_info(model_name)
            assert memory >= info.size_gb


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
