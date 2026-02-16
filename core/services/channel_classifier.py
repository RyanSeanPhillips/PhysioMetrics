"""
Channel classifier — FFT-based channel type detection for recording files.

Pure Python service (no PyQt6 imports). Analyzes channels in ABF/SMRX/EDF
files to classify them as pleth, stim, noise, or empty. Generates optional
thumbnail PNG for visual confirmation.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def classify_channels(
    file_path: Path,
    sample_seconds: float = 5.0,
) -> Dict[str, Any]:
    """
    Analyze all channels in a recording file and classify each one.

    Loads only the first `sample_seconds` of data for speed.

    Args:
        file_path: Path to recording file (.abf, .smrx, .edf).
        sample_seconds: How many seconds of data to analyze (default 5).

    Returns:
        Dict with:
            channels: List of channel classification dicts.
            file_name: str
            file_type: str
            sample_rate: float
    """
    import numpy as np

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".abf":
        return _classify_abf(file_path, sample_seconds)
    elif suffix == ".smrx":
        return _classify_smrx(file_path, sample_seconds)
    elif suffix == ".edf":
        return _classify_edf(file_path, sample_seconds)
    else:
        return {"error": f"Unsupported file type: {suffix}", "channels": []}


def _classify_abf(file_path: Path, sample_seconds: float) -> Dict[str, Any]:
    """Classify channels in an ABF file."""
    import pyabf
    import numpy as np

    abf = pyabf.ABF(str(file_path))
    sample_rate = abf.sampleRate
    results = []

    for ch in range(abf.channelCount):
        abf.setSweep(0, channel=ch)
        data = abf.sweepY

        # Only use first N seconds
        n_samples = min(len(data), int(sample_seconds * sample_rate))
        data = data[:n_samples]

        name = abf.adcNames[ch] if ch < len(abf.adcNames) else f"AD{ch}"
        units = abf.adcUnits[ch] if ch < len(abf.adcUnits) else ""

        features = _compute_features(data, sample_rate)
        classification, confidence = _classify_from_features(features, name, units)

        results.append({
            "index": ch,
            "name": name,
            "units": units,
            "classification": classification,
            "confidence": round(confidence, 3),
            "features": features,
        })

    return {
        "file_name": file_path.name,
        "file_type": "abf",
        "sample_rate": sample_rate,
        "channel_count": abf.channelCount,
        "sweep_count": abf.sweepCount,
        "channels": results,
    }


def _classify_smrx(file_path: Path, sample_seconds: float) -> Dict[str, Any]:
    """Classify channels in an SMRX file."""
    import numpy as np

    try:
        from core.io.son64_dll_loader import SON64Loader
    except ImportError:
        return {"error": "SON64 DLL not available", "channels": []}

    loader = SON64Loader()
    loader.open(str(file_path))

    try:
        all_channels = loader.get_all_channels()
        waveform_channels = [ch for ch in all_channels if ch["kind"] in [1, 7]]

        results = []
        sample_rate = None

        for ch_info in waveform_channels:
            ch_num = ch_info["channel"]
            try:
                data, sr = loader.read_channel(ch_num)
                if sample_rate is None:
                    sample_rate = sr

                n_samples = min(len(data), int(sample_seconds * sr))
                data = data[:n_samples]

                name = ch_info.get("title") or f"Channel_{ch_num}"
                units = ch_info.get("units", "")

                features = _compute_features(data, sr)
                classification, confidence = _classify_from_features(features, name, units)

                results.append({
                    "index": ch_num,
                    "name": name,
                    "units": units,
                    "classification": classification,
                    "confidence": round(confidence, 3),
                    "features": features,
                })
            except Exception as e:
                results.append({
                    "index": ch_num,
                    "name": ch_info.get("title", f"Channel_{ch_num}"),
                    "classification": "error",
                    "error": str(e),
                })

        return {
            "file_name": file_path.name,
            "file_type": "smrx",
            "sample_rate": sample_rate or 0,
            "channel_count": len(waveform_channels),
            "channels": results,
        }
    finally:
        loader.close()


def _classify_edf(file_path: Path, sample_seconds: float) -> Dict[str, Any]:
    """Classify channels in an EDF file."""
    import numpy as np

    try:
        import pyedflib
    except ImportError:
        return {"error": "pyedflib not installed", "channels": []}

    f = pyedflib.EdfReader(str(file_path))
    try:
        n_channels = f.signals_in_file
        results = []
        sample_rate = None

        for ch in range(n_channels):
            label = f.getSignalLabels()[ch].strip()
            if label in ("EDF Annotations", ""):
                continue

            sr = f.getSampleFrequency(ch)
            if sr <= 0:
                continue

            if sample_rate is None:
                sample_rate = sr

            n_samples = min(f.getNSamples()[ch], int(sample_seconds * sr))
            data = f.readSignal(ch, 0, n_samples)

            units = f.getPhysicalDimension(ch).strip()
            features = _compute_features(data, sr)
            classification, confidence = _classify_from_features(features, label, units)

            results.append({
                "index": ch,
                "name": label,
                "units": units,
                "classification": classification,
                "confidence": round(confidence, 3),
                "features": features,
            })

        return {
            "file_name": file_path.name,
            "file_type": "edf",
            "sample_rate": sample_rate or 0,
            "channel_count": len(results),
            "channels": results,
        }
    finally:
        f.close()


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def _compute_features(data, sample_rate: float) -> Dict[str, Any]:
    """
    Compute signal features for classification.

    Features:
        dominant_freq_hz: Frequency with highest PSD (Welch method).
        periodicity_score: Autocorrelation-based periodicity (0-1).
        amplitude_range: [min, max] of the signal.
        rms: Root mean square amplitude.
        snr_db: Signal-to-noise ratio in dB.
        is_digital: Whether signal looks binary (two main levels).
        duty_cycle: For digital signals, fraction of time "on".
        pulse_count: For digital signals, number of rising edges.
    """
    import numpy as np

    data = np.asarray(data, dtype=np.float64)

    if len(data) < 100:
        return {
            "dominant_freq_hz": None,
            "periodicity_score": 0.0,
            "amplitude_range": [0.0, 0.0],
            "rms": 0.0,
            "snr_db": 0.0,
            "is_digital": False,
        }

    # Basic stats
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    data_range = data_max - data_min
    rms = float(np.sqrt(np.mean(data ** 2)))

    # Check for flat/empty signal
    if data_range < 1e-10:
        return {
            "dominant_freq_hz": None,
            "periodicity_score": 0.0,
            "amplitude_range": [data_min, data_max],
            "rms": rms,
            "snr_db": 0.0,
            "is_digital": False,
        }

    # Dominant frequency via Welch PSD
    dominant_freq = _welch_dominant_freq(data, sample_rate)

    # Periodicity score via autocorrelation
    periodicity = _periodicity_score(data, sample_rate)

    # SNR estimation
    snr_db = _estimate_snr(data)

    # Digital detection (binary signal)
    is_digital, duty_cycle, pulse_count = _detect_digital(data, sample_rate)

    features = {
        "dominant_freq_hz": round(dominant_freq, 2) if dominant_freq else None,
        "periodicity_score": round(periodicity, 3),
        "amplitude_range": [round(data_min, 4), round(data_max, 4)],
        "rms": round(rms, 4),
        "snr_db": round(snr_db, 1),
        "is_digital": is_digital,
    }

    if is_digital:
        features["duty_cycle"] = round(duty_cycle, 3) if duty_cycle else None
        features["pulse_count"] = pulse_count

    return features


def _welch_dominant_freq(data, sample_rate: float) -> Optional[float]:
    """Find dominant frequency using Welch PSD."""
    import numpy as np
    from scipy.signal import welch

    # Use a reasonable segment length
    nperseg = min(len(data), int(sample_rate * 2))  # 2-second windows
    if nperseg < 64:
        nperseg = min(len(data), 64)

    try:
        freqs, psd = welch(data, fs=sample_rate, nperseg=nperseg)

        # Ignore DC and very low frequencies (< 0.5 Hz)
        mask = freqs > 0.5
        if not np.any(mask):
            return None

        freqs = freqs[mask]
        psd = psd[mask]

        if len(psd) == 0:
            return None

        peak_idx = np.argmax(psd)
        dominant_freq = float(freqs[peak_idx])

        # Check if peak is significantly above noise floor
        median_psd = np.median(psd)
        if median_psd > 0 and psd[peak_idx] / median_psd < 3.0:
            return None  # No clear dominant frequency

        return dominant_freq
    except Exception:
        return None


def _periodicity_score(data, sample_rate: float) -> float:
    """
    Compute periodicity score (0-1) using autocorrelation.

    High score = quasi-periodic (pleth) or perfectly periodic (stim).
    Low score = noise or flat.
    """
    import numpy as np

    # Normalize
    data = data - np.mean(data)
    std = np.std(data)
    if std < 1e-10:
        return 0.0

    data = data / std

    # Compute autocorrelation for lags up to 2 seconds
    max_lag = min(len(data) // 2, int(sample_rate * 2))
    if max_lag < 10:
        return 0.0

    # Use numpy correlate (faster than scipy for short signals)
    autocorr = np.correlate(data[:max_lag * 2], data[:max_lag * 2], mode="full")
    autocorr = autocorr[len(autocorr) // 2:]  # Keep positive lags
    autocorr = autocorr / autocorr[0]  # Normalize

    # Find the highest peak after the initial decay
    # Skip first 10% of lags (initial correlation decay)
    start_lag = max(10, int(max_lag * 0.05))
    if start_lag >= len(autocorr):
        return 0.0

    search_region = autocorr[start_lag:]
    if len(search_region) == 0:
        return 0.0

    peak_value = float(np.max(search_region))

    # Score: clamp to [0, 1]
    return max(0.0, min(1.0, peak_value))


def _estimate_snr(data) -> float:
    """Estimate signal-to-noise ratio in dB."""
    import numpy as np

    # Simple estimation: ratio of signal variance to noise variance
    # Use median filter to estimate signal, residual is noise
    from scipy.ndimage import median_filter

    # Use a window size appropriate for the signal
    window = min(len(data) // 10, 101)
    if window < 3:
        return 0.0

    if window % 2 == 0:
        window += 1

    smooth = median_filter(data, size=window)
    noise = data - smooth

    signal_power = np.var(smooth)
    noise_power = np.var(noise)

    if noise_power < 1e-20:
        return 60.0  # Very clean signal

    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)


def _detect_digital(data, sample_rate: float):
    """
    Detect if a signal is digital (binary on/off).

    Returns:
        (is_digital, duty_cycle, pulse_count)
    """
    import numpy as np

    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min

    if data_range < 0.1:
        return False, None, 0

    # Normalize to 0-1
    norm = (data - data_min) / data_range

    # Check if values cluster around 0 and 1
    near_low = np.sum(norm < 0.2) / len(norm)
    near_high = np.sum(norm > 0.8) / len(norm)
    in_middle = np.sum((norm >= 0.2) & (norm <= 0.8)) / len(norm)

    # Digital signal: most values near 0 or 1, very few in middle
    is_digital = (near_low + near_high > 0.85) and (in_middle < 0.15)

    if not is_digital:
        return False, None, 0

    # Compute duty cycle and pulse count
    threshold = 0.5
    above = norm > threshold
    duty_cycle = float(np.mean(above))

    # Count rising edges
    rising_edges = np.where(np.diff(above.astype(int)) == 1)[0]
    pulse_count = len(rising_edges)

    return True, duty_cycle, pulse_count


# --------------------------------------------------------------------------
# Classification logic
# --------------------------------------------------------------------------

def _classify_from_features(features: Dict, name: str, units: str):
    """
    Classify a channel based on computed features and metadata.

    Returns:
        (classification, confidence) where classification is one of:
        'pleth', 'stim', 'noise', 'empty', 'unknown'
    """
    # Check for empty/flat signal
    amp_range = features.get("amplitude_range", [0, 0])
    if amp_range[1] - amp_range[0] < 1e-8:
        return "empty", 0.99

    snr = features.get("snr_db", 0)
    periodicity = features.get("periodicity_score", 0)
    dominant_freq = features.get("dominant_freq_hz")
    is_digital = features.get("is_digital", False)

    # Name-based hints (high confidence)
    name_lower = name.lower() if name else ""
    stim_keywords = ["ttl", "stim", "laser", "digital", "trigger", "opto", "led", "pulse"]
    pleth_keywords = ["pleth", "flow", "airflow", "breathing", "resp"]

    name_is_stim = any(kw in name_lower for kw in stim_keywords)
    name_is_pleth = any(kw in name_lower for kw in pleth_keywords)

    # Digital signal → stim
    if is_digital:
        confidence = 0.95 if name_is_stim else 0.85
        return "stim", confidence

    # Name-based classification with feature confirmation
    if name_is_stim:
        if periodicity > 0.7:
            return "stim", 0.92
        return "stim", 0.75

    if name_is_pleth:
        if periodicity > 0.3 and snr > 5:
            return "pleth", 0.95
        return "pleth", 0.80

    # Feature-based classification
    # High periodicity + perfect frequency → stim
    if periodicity > 0.9 and dominant_freq and dominant_freq == round(dominant_freq):
        return "stim", 0.82

    # Moderate periodicity + biological frequency range → pleth
    if 0.3 < periodicity < 0.95 and dominant_freq and 1.0 <= dominant_freq <= 12.0:
        if snr > 8:
            return "pleth", 0.85
        elif snr > 3:
            return "pleth", 0.65

    # Low SNR, low periodicity → noise
    if snr < 3 and periodicity < 0.2:
        return "noise", 0.80

    # Low SNR → likely noise
    if snr < 5:
        return "noise", 0.60

    # Has signal but unclear type
    if snr > 10 and periodicity > 0.3:
        return "unknown", 0.50

    return "unknown", 0.40


# --------------------------------------------------------------------------
# Thumbnail generation
# --------------------------------------------------------------------------

def generate_thumbnail(
    file_path: Path,
    output_path: Optional[Path] = None,
    sample_seconds: float = 5.0,
    width_px: int = 600,
    height_per_channel: int = 120,
) -> Optional[str]:
    """
    Generate a multi-channel thumbnail PNG for a recording file.

    Args:
        file_path: Path to the recording file.
        output_path: Where to save the PNG. If None, auto-generates path.
        sample_seconds: How many seconds to show.
        width_px: Image width in pixels.
        height_per_channel: Height per channel subplot.

    Returns:
        Path to the generated PNG, or None on error.
    """
    import numpy as np

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # Determine output path
    if output_path is None:
        viz_dir = file_path.parent / "_internal" / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        output_path = viz_dir / f"channel_preview_{file_path.stem}.png"

    try:
        # Load data
        channels_data = _load_channels_for_thumbnail(file_path, sample_seconds)
        if not channels_data:
            return None

        # Generate plot
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        n_channels = len(channels_data)
        fig_height = max(2, n_channels * (height_per_channel / 100))
        fig, axes = plt.subplots(n_channels, 1, figsize=(width_px / 100, fig_height),
                                  squeeze=False)

        fig.suptitle(file_path.name, fontsize=10, fontweight="bold")

        for i, ch in enumerate(channels_data):
            ax = axes[i, 0]
            time = np.arange(len(ch["data"])) / ch["sample_rate"]
            ax.plot(time, ch["data"], linewidth=0.5, color="#4080ff")
            ax.set_ylabel(f"{ch['name']}\n({ch['units']})", fontsize=7, rotation=0,
                          labelpad=50, ha="left", va="center")
            ax.tick_params(labelsize=6)

            # Add classification label
            cls = ch.get("classification", "?")
            conf = ch.get("confidence", 0)
            color = {"pleth": "#2ecc71", "stim": "#e74c3c", "noise": "#95a5a6",
                      "empty": "#7f8c8d"}.get(cls, "#f39c12")
            ax.text(0.98, 0.92, f"{cls} ({conf:.0%})", transform=ax.transAxes,
                    fontsize=7, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.6))

            if i < n_channels - 1:
                ax.set_xticklabels([])

        axes[-1, 0].set_xlabel("Time (s)", fontsize=8)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return str(output_path)

    except Exception as e:
        print(f"[channel-classifier] Thumbnail error: {e}")
        return None


def _load_channels_for_thumbnail(file_path: Path, sample_seconds: float) -> List[Dict]:
    """Load channel data + classification for thumbnail generation."""
    import numpy as np

    result = classify_channels(file_path, sample_seconds=sample_seconds)
    if "error" in result:
        return []

    suffix = file_path.suffix.lower()
    channels_data = []

    if suffix == ".abf":
        import pyabf
        abf = pyabf.ABF(str(file_path))

        for ch_info in result["channels"]:
            ch = ch_info["index"]
            abf.setSweep(0, channel=ch)
            n_samples = min(len(abf.sweepY), int(sample_seconds * abf.sampleRate))
            channels_data.append({
                "name": ch_info["name"],
                "units": ch_info.get("units", ""),
                "data": abf.sweepY[:n_samples],
                "sample_rate": abf.sampleRate,
                "classification": ch_info["classification"],
                "confidence": ch_info["confidence"],
            })

    # For SMRX/EDF, similar logic would apply but we'll keep it simple
    # and return the classification result without re-loading data

    return channels_data
