"""
Base classes for diagnostic tests and metric collection.

This module provides the foundation for all performance tests in the diagnostics suite.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import time
import json
from datetime import datetime


class TestMode(Enum):
    """Test execution modes with different levels of thoroughness"""
    QUICK = "quick"          # 1-5 minutes: Basic sanity check
    DEEP = "deep"            # 20-30 minutes: Comprehensive benchmarking
    BURN_IN = "burn_in"      # Hours/days: Long-running stability test


class MetricType(Enum):
    """Categories of metrics we measure"""
    HARDWARE = "hardware"           # GPU count, CUDA version, etc.
    MEMORY = "memory"               # Capacity, bandwidth, allocation
    COMPUTE = "compute"             # TFLOPS, GEMM performance
    INFERENCE = "inference"         # Tokens/sec, latency
    COMMUNICATION = "communication" # NCCL bandwidth, all-reduce time
    STABILITY = "stability"         # Temperature, power, errors


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class TestResult:
    """Result from a single diagnostic test

    This class stores both the raw measurements and human-readable interpretations.
    It's designed to be educational - helping users understand what the numbers mean.
    """
    test_name: str
    status: TestStatus

    # Measurements
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Educational context
    description: str = ""              # What this test does
    interpretation: str = ""           # What the results mean
    recommendation: str = ""           # What action to take

    # Metadata
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None

    # Visual indicators for reports
    emoji: str = "ℹ️"  # 🔍 ✅ ⚠️ ❌

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "test_name": self.test_name,
            "status": self.status.value,
            "metrics": self.metrics,
            "description": self.description,
            "interpretation": self.interpretation,
            "recommendation": self.recommendation,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "error_message": self.error_message,
            "emoji": self.emoji
        }

    def to_json(self) -> str:
        """Export as JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def print_summary(self, verbose: bool = False):
        """Print human-readable summary to console"""
        status_symbols = {
            TestStatus.PASSED: "✓",
            TestStatus.FAILED: "✗",
            TestStatus.WARNING: "⚠",
            TestStatus.SKIPPED: "○",
            TestStatus.RUNNING: "→",
            TestStatus.PENDING: "…"
        }

        symbol = status_symbols.get(self.status, "?")
        print(f"{self.emoji} {symbol} {self.test_name}")

        if self.interpretation:
            print(f"  {self.interpretation}")

        if verbose and self.metrics:
            print("  Metrics:")
            for key, value in self.metrics.items():
                print(f"    • {key}: {value}")

        if self.recommendation and self.status in [TestStatus.WARNING, TestStatus.FAILED]:
            print(f"  💡 {self.recommendation}")

        if self.error_message:
            print(f"  ❌ Error: {self.error_message}")


@dataclass
class TestSuite:
    """Collection of related tests"""
    name: str
    description: str
    tests: List[TestResult] = field(default_factory=list)

    def add_result(self, result: TestResult):
        """Add a test result to this suite"""
        self.tests.append(result)

    def get_summary(self) -> Dict[str, int]:
        """Get count of tests by status"""
        summary = {status: 0 for status in TestStatus}
        for test in self.tests:
            summary[test.status] += 1
        return {k.value: v for k, v in summary.items()}

    def all_passed(self) -> bool:
        """Check if all tests passed"""
        return all(t.status == TestStatus.PASSED for t in self.tests)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "summary": self.get_summary(),
            "tests": [t.to_dict() for t in self.tests]
        }


class DiagnosticTest(ABC):
    """
    Abstract base class for all diagnostic tests.

    All profilers (memory, compute, NCCL, etc.) inherit from this class.
    It provides:
    - Progress reporting hooks
    - Timing infrastructure
    - Educational annotations
    - Consistent result formatting
    """

    def __init__(self, name: str, description: str, metric_type: MetricType):
        self.name = name
        self.description = description
        self.metric_type = metric_type
        self.progress_callback: Optional[Callable[[str, float], None]] = None

    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set callback for progress updates

        Args:
            callback: Function(message: str, progress: float 0-100)
        """
        self.progress_callback = callback

    def _report_progress(self, message: str, progress: float = 0.0):
        """Report progress to callback if set"""
        if self.progress_callback:
            self.progress_callback(message, progress)
        else:
            # Default: print to console
            print(f"  [{progress:.0f}%] {message}")

    @abstractmethod
    def run(self, mode: TestMode = TestMode.QUICK) -> TestResult:
        """
        Run the diagnostic test.

        Args:
            mode: Test mode (quick, deep, or burn-in)

        Returns:
            TestResult with measurements and interpretation
        """
        pass

    def _create_result(
        self,
        status: TestStatus,
        metrics: Dict[str, Any],
        interpretation: str = "",
        recommendation: str = "",
        error_message: Optional[str] = None
    ) -> TestResult:
        """Helper to create standardized test results"""

        # Choose emoji based on status
        emoji_map = {
            TestStatus.PASSED: "✅",
            TestStatus.FAILED: "❌",
            TestStatus.WARNING: "⚠️",
            TestStatus.SKIPPED: "⏭️",
            TestStatus.RUNNING: "🔄",
            TestStatus.PENDING: "⏳"
        }

        return TestResult(
            test_name=self.name,
            status=status,
            metrics=metrics,
            description=self.description,
            interpretation=interpretation,
            recommendation=recommendation,
            error_message=error_message,
            emoji=emoji_map.get(status, "ℹ️")
        )

    def _timed_execution(self, func: Callable) -> tuple[Any, float]:
        """Execute a function and measure its duration

        Returns:
            (result, duration_seconds)
        """
        start = time.perf_counter()
        result = func()
        duration = time.perf_counter() - start
        return result, duration

    def explain_metric(self, metric_name: str, value: Any, context: str = "") -> str:
        """
        Generate educational explanation for a metric.

        This is where we make the diagnostics educational!
        Override in subclasses for metric-specific explanations.

        Args:
            metric_name: Name of the metric
            value: Measured value
            context: Additional context (e.g., "peak" value for comparison)

        Returns:
            Human-readable explanation
        """
        return f"{metric_name}: {value}"


class ProgressTracker:
    """Helper class for tracking multi-step test progress"""

    def __init__(self, total_steps: int, callback: Optional[Callable[[str, float], None]] = None):
        self.total_steps = total_steps
        self.current_step = 0
        self.callback = callback

    def update(self, message: str, increment: int = 1):
        """Update progress"""
        self.current_step += increment
        progress = (self.current_step / self.total_steps) * 100

        if self.callback:
            self.callback(message, progress)
        else:
            print(f"  [{progress:.0f}%] {message}")

    def complete(self, message: str = "Complete"):
        """Mark as 100% complete"""
        if self.callback:
            self.callback(message, 100.0)
        else:
            print(f"  [100%] {message}")


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_throughput(value: float, unit: str = "tokens/sec") -> str:
    """Format throughput with appropriate precision"""
    if value >= 1000:
        return f"{value:,.0f} {unit}"
    elif value >= 10:
        return f"{value:.1f} {unit}"
    else:
        return f"{value:.2f} {unit}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format as percentage"""
    return f"{value:.{decimals}f}%"
