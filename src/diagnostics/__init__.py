"""
NCCL Diagnostics & Performance Testing Module

This module helps hobbyists understand their hardware capabilities for LLM training and inference.
It provides comprehensive benchmarking for:
- Memory capacity and bandwidth
- Compute throughput (TFLOPS)
- Inference performance (tokens/second)
- Multi-GPU communication (NCCL)
- Long-term stability (burn-in tests)

Usage:
    # CLI
    python -m diagnostics.cli --mode quick

    # Programmatic
    from diagnostics import HardwareDetector, MemoryProfiler
    detector = HardwareDetector()
    info = detector.detect()
"""

__version__ = "0.1.0"

from .core.hardware_detector import HardwareDetector
from .core.metrics import DiagnosticTest, TestResult, TestMode

__all__ = [
    "HardwareDetector",
    "DiagnosticTest",
    "TestResult",
    "TestMode",
]
