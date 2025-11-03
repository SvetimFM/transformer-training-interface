"""
Model Loader for Inference Profiling

Downloads and caches small transformer models for realistic benchmarking.
Uses HuggingFace Hub for model access.
"""

import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BenchmarkModel:
    """Specification for a benchmark model"""
    name: str
    repo_id: str
    size_gb: float
    min_vram_gb: float
    description: str
    params_billions: float


class ModelLoader:
    """
    Download and load transformer models for benchmarking.

    Automatically selects appropriate model based on available VRAM.
    Caches models locally to avoid re-downloading.
    """

    # Curated list of models for benchmarking
    BENCHMARK_MODELS = {
        "TinyLlama-1.1B": BenchmarkModel(
            name="TinyLlama-1.1B",
            repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            size_gb=2.2,
            min_vram_gb=4,
            description="Smallest realistic LLM, fast testing",
            params_billions=1.1
        ),
        "Qwen2.5-0.5B": BenchmarkModel(
            name="Qwen2.5-0.5B",
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            size_gb=1.0,
            min_vram_gb=3,
            description="Tiny model for constrained hardware",
            params_billions=0.5
        ),
        "Qwen2.5-1.5B": BenchmarkModel(
            name="Qwen2.5-1.5B",
            repo_id="Qwen/Qwen2.5-1.5B-Instruct",
            size_gb=3.1,
            min_vram_gb=6,
            description="Small but capable model",
            params_billions=1.5
        ),
        "Qwen2.5-3B": BenchmarkModel(
            name="Qwen2.5-3B",
            repo_id="Qwen/Qwen2.5-3B-Instruct",
            size_gb=6.5,
            min_vram_gb=8,
            description="Mid-size model, good baseline",
            params_billions=3.1
        ),
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model loader.

        Args:
            cache_dir: Where to cache downloaded models.
                      Defaults to ~/.cache/diagnostics/models
        """
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), ".cache", "diagnostics", "models")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def auto_select_model(self, available_vram_gb: float, conservative: bool = True) -> Optional[str]:
        """
        Automatically select the largest model that fits in available VRAM.

        Args:
            available_vram_gb: Available VRAM in gigabytes
            conservative: If True, leave 20% headroom for KV cache and activations

        Returns:
            Model name, or None if no model fits
        """
        if conservative:
            # Leave 20% headroom for KV cache, activations, etc.
            usable_vram = available_vram_gb * 0.8
        else:
            usable_vram = available_vram_gb

        # Sort models by size (largest first)
        sorted_models = sorted(
            self.BENCHMARK_MODELS.items(),
            key=lambda x: x[1].params_billions,
            reverse=True
        )

        for name, spec in sorted_models:
            if usable_vram >= spec.min_vram_gb:
                return name

        return None

    def get_model_info(self, model_name: str) -> Optional[BenchmarkModel]:
        """Get information about a specific model"""
        return self.BENCHMARK_MODELS.get(model_name)

    def is_model_cached(self, model_name: str) -> bool:
        """Check if model is already downloaded"""
        spec = self.BENCHMARK_MODELS.get(model_name)
        if not spec:
            return False

        # Check if model directory exists in cache
        model_path = self.cache_dir / model_name
        return model_path.exists() and any(model_path.iterdir())

    def download_model(self, model_name: str, progress_callback=None) -> str:
        """
        Download model from HuggingFace Hub.

        Args:
            model_name: Name of the benchmark model
            progress_callback: Optional callback(message: str, progress: float 0-100)

        Returns:
            Path to downloaded model

        Raises:
            ValueError: If model name not found
            ImportError: If transformers not installed
        """
        spec = self.BENCHMARK_MODELS.get(model_name)
        if not spec:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.BENCHMARK_MODELS.keys())}")

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed. "
                "Install with: pip install huggingface_hub transformers"
            )

        model_path = self.cache_dir / model_name

        # Check if already downloaded
        if self.is_model_cached(model_name):
            if progress_callback:
                progress_callback(f"Model {model_name} already cached", 100)
            return str(model_path)

        # Download
        if progress_callback:
            progress_callback(f"Downloading {model_name} (~{spec.size_gb:.1f} GB)...", 0)

        try:
            downloaded_path = snapshot_download(
                repo_id=spec.repo_id,
                cache_dir=str(self.cache_dir),
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
            )

            if progress_callback:
                progress_callback(f"Download complete: {model_name}", 100)

            return downloaded_path

        except Exception as e:
            if progress_callback:
                progress_callback(f"Download failed: {str(e)}", 0)
            raise

    def load_model_and_tokenizer(self, model_name: str, device: str = "cuda", dtype=torch.float16):
        """
        Load model and tokenizer for inference.

        Args:
            model_name: Name of the benchmark model
            device: Device to load model on
            dtype: Data type for model weights

        Returns:
            (model, tokenizer) tuple
        """
        try:
            # Temporarily filter out paths that might contain conflicting 'tokenizers' module
            import sys
            original_path = sys.path.copy()
            sys.path = [p for p in sys.path if 'src' not in p or 'site-packages' in p]
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            finally:
                sys.path = original_path
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers"
            )

        spec = self.BENCHMARK_MODELS.get(model_name)
        if not spec:
            raise ValueError(f"Unknown model: {model_name}")

        # Download if needed
        if not self.is_model_cached(model_name):
            self.download_model(model_name)

        model_path = self.cache_dir / model_name

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )

        # Clean up generation config to avoid warnings
        # Some models have default temperature/top_p/top_k that aren't compatible with do_sample=False
        if hasattr(model, 'generation_config'):
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None

        model.eval()  # Set to evaluation mode

        return model, tokenizer

    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available benchmark models.

        Returns:
            Dictionary of model specs with cache status
        """
        result = {}

        for name, spec in self.BENCHMARK_MODELS.items():
            result[name] = {
                "repo_id": spec.repo_id,
                "size_gb": spec.size_gb,
                "min_vram_gb": spec.min_vram_gb,
                "params_billions": spec.params_billions,
                "description": spec.description,
                "cached": self.is_model_cached(name)
            }

        return result

    def estimate_inference_memory(self, model_name: str, sequence_length: int = 512, batch_size: int = 1) -> float:
        """
        Estimate total memory needed for inference.

        Args:
            model_name: Name of the benchmark model
            sequence_length: Input sequence length
            batch_size: Batch size

        Returns:
            Estimated memory in GB
        """
        spec = self.BENCHMARK_MODELS.get(model_name)
        if not spec:
            return 0.0

        # Model weights
        model_memory_gb = spec.size_gb

        # KV cache: 2 (K and V) * layers * heads * seq_len * head_dim * batch * 2 bytes (fp16)
        # Rough estimate: ~2x model size for typical sequence lengths
        kv_cache_multiplier = 1 + (sequence_length / 1024)

        # Activations: depends on batch size
        activation_multiplier = 1 + (batch_size * 0.1)

        total_memory = model_memory_gb * kv_cache_multiplier * activation_multiplier

        return total_memory
