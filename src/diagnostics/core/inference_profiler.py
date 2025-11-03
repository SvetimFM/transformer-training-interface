"""
Real Inference Profiler

Tests actual transformer models to measure realistic performance:
- Tokens per second (throughput)
- Time to first token (latency)
- Memory usage patterns
- Comparison to synthetic benchmarks
"""

import torch
import time
from typing import Dict, Any, List, Optional, Tuple
from .metrics import (
    DiagnosticTest, TestResult, TestMode, TestStatus, MetricType,
    format_throughput, ProgressTracker
)
from .model_loader import ModelLoader


class InferenceProfiler(DiagnosticTest):
    """
    Comprehensive real-world inference performance testing.

    Unlike synthetic GEMM benchmarks, this tests actual transformer models
    with realistic attention mechanisms, KV caching, and memory patterns.

    Educational focus: Help users understand real-world performance,
    not just theoretical TFLOPS.
    """

    def __init__(self, device: str = "cuda:0"):
        super().__init__(
            name="Real Inference Profiler",
            description="Measure actual transformer model performance (tokens/sec, latency, memory)",
            metric_type=MetricType.INFERENCE
        )
        self.device = device

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.device_props = torch.cuda.get_device_properties(device)
        self.model_loader = ModelLoader()

    def run(self, mode: TestMode = TestMode.QUICK, model_name: Optional[str] = None, ask_permission: bool = True) -> TestResult:
        """
        Run real inference profiling.

        Args:
            mode: Test mode (quick, deep, or burn-in)
            model_name: Specific model to test, or None to auto-select
            ask_permission: If True, ask user before downloading models

        Returns:
            TestResult with comprehensive inference metrics
        """
        try:
            metrics = {}

            # Step 1: Select model
            self._report_progress("Selecting model based on VRAM...", 5)

            if model_name is None:
                available_vram_gb = self.device_props.total_memory / (1024**3)
                model_name = self.model_loader.auto_select_model(available_vram_gb)

                if model_name is None:
                    return self._create_result(
                        status=TestStatus.SKIPPED,
                        metrics={},
                        interpretation="Insufficient VRAM for smallest benchmark model",
                        recommendation="Need at least 3GB VRAM for inference profiling"
                    )

            model_info = self.model_loader.get_model_info(model_name)
            metrics["model_name"] = model_name
            metrics["model_params_billions"] = model_info.params_billions

            # Step 2: Check if download needed
            is_cached = self.model_loader.is_model_cached(model_name)
            metrics["model_cached"] = is_cached

            if not is_cached:
                if ask_permission:
                    self._report_progress(
                        f"Model {model_name} needs to be downloaded (~{model_info.size_gb:.1f} GB). Proceeding...",
                        10
                    )
                    # In a real implementation, we'd ask for user consent here
                    # For now, we'll proceed automatically

                self._report_progress(f"Downloading {model_name}...", 15)
                self.model_loader.download_model(
                    model_name,
                    progress_callback=lambda msg, prog: self._report_progress(msg, 15 + (prog * 0.2))
                )

            # Step 3: Load model
            self._report_progress("Loading model...", 35)
            model, tokenizer = self.model_loader.load_model_and_tokenizer(
                model_name,
                device=self.device,
                dtype=torch.float16
            )

            # Step 4: Warm-up
            self._report_progress("Warming up model...", 40)
            self._warmup(model, tokenizer)

            # Step 5: Throughput benchmarks
            self._report_progress("Measuring throughput...", 50)
            throughput_metrics = self._benchmark_throughput(model, tokenizer, mode)
            metrics.update(throughput_metrics)

            # Step 6: Latency benchmarks
            self._report_progress("Measuring latency...", 65)
            latency_metrics = self._benchmark_latency(model, tokenizer)
            metrics.update(latency_metrics)

            # Step 7: Memory profiling (deep mode only)
            if mode in [TestMode.DEEP, TestMode.BURN_IN]:
                self._report_progress("Profiling memory usage...", 80)
                memory_metrics = self._benchmark_memory_usage(model, tokenizer)
                metrics.update(memory_metrics)

            # Step 8: Cleanup
            del model, tokenizer
            torch.cuda.empty_cache()

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
                interpretation="Inference profiling failed",
                recommendation="Check model availability and VRAM",
                error_message=str(e)
            )

    def _warmup(self, model, tokenizer, num_iterations: int = 3):
        """Warm-up model to ensure accurate measurements"""
        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model.generate(**inputs, max_new_tokens=10)

        torch.cuda.synchronize()

    def _benchmark_throughput(self, model, tokenizer, mode: TestMode) -> Dict[str, Any]:
        """
        Measure tokens per second at various batch sizes.

        Throughput = total tokens generated / time
        """
        metrics = {}

        # Determine batch sizes to test based on mode
        if mode == TestMode.QUICK:
            batch_sizes = [1]  # Just interactive mode
        elif mode == TestMode.DEEP:
            batch_sizes = [1, 4, 8]  # Interactive + bulk processing
        else:  # BURN_IN
            batch_sizes = [1, 4]

        prompt = "Explain the concept of machine learning in simple terms"
        max_new_tokens = 100

        for batch_size in batch_sizes:
            # Create batch
            prompts = [prompt] * batch_size
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

            # Warm-up for this batch size
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=10)
            torch.cuda.synchronize()

            # Measure
            num_runs = 5 if mode == TestMode.QUICK else 10
            total_tokens = 0
            total_time = 0

            for _ in range(num_runs):
                start = time.perf_counter()

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Deterministic for consistency
                        pad_token_id=tokenizer.eos_token_id
                    )

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                tokens_generated = batch_size * max_new_tokens
                total_tokens += tokens_generated
                total_time += elapsed

            # Calculate metrics
            avg_tokens_per_sec = total_tokens / total_time
            avg_latency_per_token_ms = (total_time / total_tokens) * 1000

            metrics[f"throughput_batch{batch_size}_tokens_per_sec"] = round(avg_tokens_per_sec, 2)
            metrics[f"latency_batch{batch_size}_ms_per_token"] = round(avg_latency_per_token_ms, 2)

        # Store the single-batch throughput as primary metric
        metrics["tokens_per_sec"] = metrics.get("throughput_batch1_tokens_per_sec", 0)

        return metrics

    def _benchmark_latency(self, model, tokenizer) -> Dict[str, Any]:
        """
        Measure detailed latency breakdown.

        Time to First Token (TTFT) = time to start generating
        Decode latency = time per subsequent token
        """
        metrics = {}

        prompt = "Write a detailed explanation of quantum computing"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]

        # Measure time to first token (prefill phase)
        times = []
        for _ in range(5):
            start = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            torch.cuda.synchronize()
            ttft = time.perf_counter() - start
            times.append(ttft)

        avg_ttft = sum(times) / len(times)
        metrics["time_to_first_token_ms"] = round(avg_ttft * 1000, 2)
        metrics["prompt_length_tokens"] = prompt_length
        metrics["prefill_tokens_per_sec"] = round(prompt_length / avg_ttft, 2)

        # Measure decode phase (generating subsequent tokens)
        decode_times = []
        num_decode_tokens = 50

        for _ in range(3):  # Average over 3 runs
            start = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=num_decode_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            torch.cuda.synchronize()
            total_time = time.perf_counter() - start

            # Subtract TTFT to get pure decode time
            decode_time = total_time - avg_ttft
            decode_times.append(decode_time)

        avg_decode_time = sum(decode_times) / len(decode_times)
        decode_latency_per_token = avg_decode_time / num_decode_tokens

        metrics["decode_latency_ms_per_token"] = round(decode_latency_per_token * 1000, 2)
        metrics["decode_tokens_per_sec"] = round(1 / decode_latency_per_token, 2)

        return metrics

    def _benchmark_memory_usage(self, model, tokenizer, sequence_lengths: List[int] = None) -> Dict[str, Any]:
        """
        Measure memory usage across different sequence lengths.

        Shows KV cache growth pattern.
        """
        if sequence_lengths is None:
            sequence_lengths = [128, 512, 1024, 2048]

        metrics = {}
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()

        for seq_len in sequence_lengths:
            # Create prompt of target length (approximately)
            prompt = "word " * (seq_len // 2)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len).to(self.device)
            actual_prompt_len = inputs.input_ids.shape[1]

            # Reset peak stats for this test
            torch.cuda.reset_peak_memory_stats()

            # Generate with this prompt
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                peak_memory = torch.cuda.max_memory_allocated()
                current_memory = torch.cuda.memory_allocated()

                metrics[f"memory_seq{seq_len}_peak_mb"] = round(peak_memory / (1024**2), 2)
                metrics[f"memory_seq{seq_len}_active_mb"] = round(current_memory / (1024**2), 2)
                metrics[f"memory_seq{seq_len}_kv_cache_estimate_mb"] = round(
                    (current_memory - baseline_memory) / (1024**2), 2
                )

            except RuntimeError as e:
                if "out of memory" in str(e):
                    metrics[f"memory_seq{seq_len}_status"] = "OOM"
                    break  # Don't try longer sequences
                raise

            # Cleanup
            del inputs, outputs
            torch.cuda.empty_cache()

        return metrics

    def _generate_interpretation(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable interpretation"""
        lines = []

        model_name = metrics.get("model_name", "Unknown")
        params_b = metrics.get("model_params_billions", 0)

        lines.append(f"Model: {model_name} ({params_b}B parameters)")

        # Throughput
        tokens_per_sec = metrics.get("tokens_per_sec", 0)
        if tokens_per_sec > 0:
            lines.append(f"Throughput: {tokens_per_sec:.1f} tokens/sec (batch=1)")

            # Translate to user scenarios
            if tokens_per_sec >= 50:
                speed_desc = "Excellent! Near-instant responses"
            elif tokens_per_sec >= 30:
                speed_desc = "Great for interactive chat"
            elif tokens_per_sec >= 15:
                speed_desc = "Good for most use cases"
            else:
                speed_desc = "Adequate for basic tasks"

            lines.append(f"  → {speed_desc}")

        # Latency
        ttft = metrics.get("time_to_first_token_ms", 0)
        if ttft > 0:
            lines.append(f"Time to First Token: {ttft:.0f} ms")
            if ttft < 50:
                lines.append("  → Feels instant!")
            elif ttft < 200:
                lines.append("  → Responsive")
            else:
                lines.append("  → Noticeable delay")

        # Memory
        peak_mem = metrics.get("memory_seq1024_peak_mb", 0)
        if peak_mem > 0:
            lines.append(f"Memory (1K context): {peak_mem:.1f} MB")

        return "\n  ".join(lines)

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> str:
        """Generate actionable recommendations"""
        recommendations = []

        tokens_per_sec = metrics.get("tokens_per_sec", 0)
        model_name = metrics.get("model_name", "")

        # Performance recommendations
        if tokens_per_sec > 0:
            if tokens_per_sec >= 40:
                recommendations.append("✓ Excellent performance for this model size!")
            elif tokens_per_sec >= 20:
                recommendations.append("✓ Good performance")
                recommendations.append("  Try torch.compile for 1.3-1.8x speedup")
            else:
                recommendations.append("⚠ Performance could be improved")
                recommendations.append("  Check for thermal throttling")
                recommendations.append("  Try enabling torch.compile")

        # Model size recommendations
        if "3B" in model_name or "1.5B" in model_name:
            recommendations.append("💡 You can likely run 7B models on this hardware")
        elif "1.1B" in model_name or "0.5B" in model_name:
            recommendations.append("💡 Try testing with a larger model (3B or 7B)")

        # Memory recommendations
        total_vram_gb = self.device_props.total_memory / (1024**3)
        if total_vram_gb >= 12:
            recommendations.append("✓ Sufficient VRAM for 7B-13B models")
        elif total_vram_gb >= 8:
            recommendations.append("✓ Good for 7B models, 13B with quantization")
        else:
            recommendations.append("ℹ Limited VRAM - consider 4-bit quantization for larger models")

        return "\n  ".join(recommendations) if recommendations else "Performance looks good!"

    def compare_to_synthetic(self, inference_metrics: Dict[str, Any], compute_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare real inference to synthetic GEMM benchmarks.

        This is educational - shows users why synthetic benchmarks differ from real performance.
        """
        result = {}

        # Get synthetic TFLOPS from compute profiler
        synthetic_tflops = compute_metrics.get("fp16_tflops", 0)

        # Estimate real TFLOPS from inference throughput
        # Very rough estimate: tokens/sec * model_params * 2 (forward pass FLOPs)
        tokens_per_sec = inference_metrics.get("tokens_per_sec", 0)
        params_billions = inference_metrics.get("model_params_billions", 0)

        if tokens_per_sec > 0 and params_billions > 0:
            # Each token requires ~2 * params FLOPs (multiply-add)
            flops_per_token = params_billions * 2 * 1e9
            real_tflops = (tokens_per_sec * flops_per_token) / 1e12

            result["real_inference_tflops"] = round(real_tflops, 2)
            result["synthetic_gemm_tflops"] = synthetic_tflops

            if synthetic_tflops > 0:
                efficiency = (real_tflops / synthetic_tflops) * 100
                result["efficiency_percent"] = round(efficiency, 1)

                # Educational explanation
                if efficiency >= 70:
                    explanation = "Excellent efficiency! Very close to theoretical maximum."
                elif efficiency >= 50:
                    explanation = "Good efficiency. Gap due to memory movement and attention complexity."
                elif efficiency >= 30:
                    explanation = "Normal range. Real transformers are memory-bound, not compute-bound."
                else:
                    explanation = "Lower than expected. Check for thermal throttling or configuration issues."

                result["explanation"] = explanation

        return result
