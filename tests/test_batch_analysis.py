"""
Tests for the batch analysis pipeline (Phase A, Step 3).

Verifies that headless analysis produces sensible results without
requiring the GUI. These are integration tests that load real ABF files.

Run: python -m pytest tests/test_batch_analysis.py -v
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Fixtures ─────────────────────────────────────────────────────

EXAMPLES = ROOT / "examples"
SAMPLE_ABF = EXAMPLES / "25121004.abf"


@pytest.fixture
def sample_abf():
    """Path to a sample ABF file for testing."""
    if not SAMPLE_ABF.exists():
        pytest.skip(f"Sample ABF not found: {SAMPLE_ABF}")
    return SAMPLE_ABF


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory."""
    return tmp_path


# ── Test: Config dataclasses ─────────────────────────────────────


class TestConfigModels:
    def test_filter_config_roundtrip(self):
        from core.domain.analysis.models import FilterConfig

        fc = FilterConfig(use_low=True, low_hz=50.0, notch_lower=58, notch_upper=62)
        d = fc.to_dict()
        fc2 = FilterConfig.from_dict(d)
        assert fc2.use_low is True
        assert fc2.low_hz == 50.0
        assert fc2.notch_lower == 58

    def test_peak_config_roundtrip(self):
        from core.domain.analysis.models import PeakDetectionConfig

        pc = PeakDetectionConfig(prominence=0.5, min_dist_sec=0.1)
        d = pc.to_dict()
        pc2 = PeakDetectionConfig.from_dict(d)
        assert pc2.prominence == 0.5
        assert pc2.min_dist_sec == 0.1

    def test_analysis_config_roundtrip(self):
        from core.domain.analysis.models import AnalysisConfig, FilterConfig

        ac = AnalysisConfig(filter=FilterConfig(use_zscore=False))
        d = ac.to_dict()
        ac2 = AnalysisConfig.from_dict(d)
        assert ac2.filter.use_zscore is False


# ── Test: Signal processing ──────────────────────────────────────


class TestSignalProcessing:
    def test_passthrough_no_filters(self):
        from core.services.analysis_service import get_processed_signal
        from core.domain.analysis.models import FilterConfig

        y = np.sin(np.linspace(0, 2 * np.pi, 1000))
        fc = FilterConfig(use_zscore=False)
        result = get_processed_signal(y, 1000.0, fc)
        np.testing.assert_allclose(result, y, atol=1e-10)

    def test_zscore_normalises(self):
        from core.services.analysis_service import get_processed_signal
        from core.domain.analysis.models import FilterConfig

        y = np.random.randn(10000) * 5 + 100  # mean=100, std=5
        fc = FilterConfig(use_zscore=True)
        result = get_processed_signal(y, 1000.0, fc)
        assert abs(np.mean(result)) < 0.1
        assert abs(np.std(result) - 1.0) < 0.1

    def test_notch_filter(self):
        from core.services.analysis_service import get_processed_signal
        from core.domain.analysis.models import FilterConfig

        sr = 1000.0
        t = np.linspace(0, 1, int(sr))
        # 60 Hz noise + 5 Hz signal
        y = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
        fc = FilterConfig(notch_lower=58, notch_upper=62, use_zscore=False)
        result = get_processed_signal(y, sr, fc)
        # 60 Hz component should be attenuated
        fft_before = np.abs(np.fft.rfft(y))
        fft_after = np.abs(np.fft.rfft(result))
        freq = np.fft.rfftfreq(len(y), 1 / sr)
        idx_60 = np.argmin(np.abs(freq - 60))
        assert fft_after[idx_60] < fft_before[idx_60] * 0.1


# ── Test: Auto-threshold ─────────────────────────────────────────


class TestAutoThreshold:
    def test_detects_threshold(self):
        from core.services.analysis_service import auto_detect_threshold

        sr = 1000.0
        t = np.linspace(0, 5, int(sr * 5))
        # Clear bimodal: noise peaks ~0.05, signal peaks ~0.5
        y = np.random.randn(len(t)) * 0.02
        for i in range(0, len(t), 300):
            if i + 50 < len(t):
                y[i : i + 50] += np.sin(np.linspace(0, np.pi, 50)) * 0.5
        threshold = auto_detect_threshold(y, sr)
        assert threshold is not None
        assert 0.01 < threshold < 0.4  # between noise and signal


# ── Test: Full pipeline (integration) ────────────────────────────


class TestAnalyzeFile:
    def test_analyze_produces_csv(self, sample_abf, tmp_output):
        from core.services.analysis_service import analyze_file
        from core.domain.analysis.models import AnalysisConfig

        result = analyze_file(sample_abf, AnalysisConfig(), tmp_output)

        assert result.success, f"Analysis failed: {result.error}"
        assert result.n_sweeps > 0
        assert result.n_breaths_total > 0
        assert result.results_path is not None
        assert result.results_path.exists()

        # Verify CSV has content
        import csv

        with open(result.results_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == result.n_breaths_total
        assert "file" in reader.fieldnames
        assert "sweep" in reader.fieldnames
        assert "peak_time" in reader.fieldnames

    def test_analyze_with_custom_config(self, sample_abf, tmp_output):
        from core.services.analysis_service import analyze_file
        from core.domain.analysis.models import AnalysisConfig, FilterConfig

        config = AnalysisConfig(filter=FilterConfig(use_zscore=False))
        result = analyze_file(sample_abf, config, tmp_output)

        assert result.success
        assert result.n_breaths_total > 0

    def test_analyze_nonexistent_file(self, tmp_output):
        from core.services.analysis_service import analyze_file
        from core.domain.analysis.models import AnalysisConfig

        result = analyze_file(
            Path("/nonexistent/file.abf"), AnalysisConfig(), tmp_output
        )
        assert not result.success
        assert result.error is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
