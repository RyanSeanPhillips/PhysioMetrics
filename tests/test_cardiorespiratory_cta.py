"""
Cardiorespiratory CTA — breath-triggered analysis of HR coupling.

Triggers CTA on inspiration onset, plots:
  Panel 1: Raw pleth waveform (mean +/- SEM across breaths)
  Panel 2: Heart Rate (BPM)
  Panel 3: Respiratory Rate (Hz)
  Panel 4: RSA Amplitude (BPM)

Uses CTA service directly (no dialog needed).

Run:  python -m pytest tests/test_cardiorespiratory_cta.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import load_file_and_wait, MULTI_CHANNEL_ABF
from test_export_and_save import _setup_analysis

OUTPUT_DIR = ROOT / "tests" / "output"


class TestCardiorespiratoryCTA:

    def test_breath_triggered_cta(self, main_window, multi_channel_abf):
        """Generate breath-triggered CTAs for HR, respiratory rate, RSA, and pleth waveform."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from core.services.cta_service import CTAService
        from core.domain.cta.models import CTAConfig
        from core import metrics as _metrics

        # 1. Set up analysis (peaks + EKG)
        _setup_analysis(main_window, multi_channel_abf)

        st = main_window.state
        sr = st.sr_hz
        sweep_idx = 0
        t = st.t

        assert st.ecg_results_by_sweep, "No EKG results"
        ecg = st.ecg_results_by_sweep.get(sweep_idx)
        assert ecg is not None and hasattr(ecg, 'r_peaks'), "No R-peaks for sweep 0"

        # 2. Get breath onset times (from detected peaks)
        peaks_dict = st.peaks_by_sweep
        breath_dict = st.breath_by_sweep
        pks = peaks_dict.get(sweep_idx)
        breaths = breath_dict.get(sweep_idx, {})

        assert pks is not None, "No peaks for sweep 0"
        onsets = breaths.get('onsets')
        offsets = breaths.get('offsets')
        assert onsets is not None, "No breath onsets"

        # Use onset times as trigger events (limit to 100)
        onset_times = t[onsets[:100]].astype(float)
        n_events = len(onset_times)
        print(f"  Using {n_events} breath onsets as CTA triggers")

        # 3. Build signals for CTA
        # Get processed signal (with filters applied) — this returns the correct sweep
        y_proc = main_window._get_processed_for(st.analyze_chan, sweep_idx)
        assert y_proc is not None and len(y_proc) == len(t), \
            f"Processed signal length mismatch: {len(y_proc) if y_proc is not None else 'None'} vs {len(t)}"

        # Compute stepwise metrics
        exm = breaths.get('expmins')
        exo = breaths.get('expoffs')

        # Set ECG result for HR metrics
        _metrics.set_ecg_result(ecg)
        cur_pm = st.current_peak_metrics_by_sweep.get(sweep_idx) or st.peak_metrics_by_sweep.get(sweep_idx)
        _metrics.set_peak_metrics(cur_pm)

        signals = {}
        metric_labels = {}

        # Pleth waveform (raw processed signal)
        signals['pleth'] = y_proc
        metric_labels['pleth'] = 'Pleth Waveform'

        # Compute each metric
        metrics_to_compute = {
            'hr': 'Heart Rate (BPM)',
            'if': 'Respiratory Rate (Hz)',
            'rsa_amplitude': 'RSA Amplitude (BPM)',
        }

        for mkey, mlabel in metrics_to_compute.items():
            if mkey in _metrics.METRICS:
                try:
                    y2 = _metrics.METRICS[mkey](t, y_proc, sr, pks, onsets, offsets, exm, exo)
                    if y2 is not None and len(y2) == len(t):
                        signals[mkey] = y2
                        metric_labels[mkey] = mlabel
                        valid = ~np.isnan(y2)
                        print(f"  {mlabel}: {np.sum(valid)} valid points, "
                              f"range [{np.nanmin(y2[valid]):.1f}, {np.nanmax(y2[valid]):.1f}]"
                              if np.any(valid) else f"  {mlabel}: all NaN")
                except Exception as e:
                    print(f"  {mlabel}: FAILED — {e}")

        _metrics.set_ecg_result(None)
        _metrics.set_peak_metrics(None)

        assert 'hr' in signals, "HR metric not computed"
        assert 'if' in signals, "IF metric not computed"

        # 4. Run CTA service
        service = CTAService()
        config = CTAConfig(
            window_before=0.5,   # 500ms before breath onset
            window_after=1.0,    # 1s after (captures full breath cycle)
            n_points=200,        # 200 Hz resolution
            zscore_baseline=False,  # Raw values, not z-scored
        )

        results = {}
        for sig_key, sig_data in signals.items():
            result = service.calculate_cta(
                time_array=t,
                signal=sig_data,
                event_times=onset_times.tolist(),
                config=config,
                metric_key=sig_key,
                metric_label=metric_labels[sig_key],
                category='breath',
                label='onset',
                alignment='onset',
            )
            if result and result.mean is not None:
                results[sig_key] = result
                print(f"  CTA {sig_key}: {result.n_events} events, "
                      f"mean range [{result.mean.min():.2f}, {result.mean.max():.2f}]")

        assert len(results) >= 3, f"Expected 3+ CTA results, got {len(results)}: {list(results.keys())}"

        # 5. Plot
        n_panels = len(results)
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels),
                                 gridspec_kw={'hspace': 0.4})
        if n_panels == 1:
            axes = [axes]

        fig.suptitle(f'Cardiorespiratory CTA — Breath-Triggered (n={n_events} breaths)\n'
                     f'26402007.abf, Sweep 1, Pleth=IN 1, EKG=IN 7',
                     fontsize=13, fontweight='bold')

        plot_order = ['pleth', 'hr', 'if', 'rsa_amplitude']
        colors = {'pleth': '#4488ff', 'hr': '#ff4444', 'if': '#44cc44', 'rsa_amplitude': '#cc44cc'}

        for ax, key in zip(axes, [k for k in plot_order if k in results]):
            r = results[key]
            color = colors.get(key, '#888888')

            # Plot individual traces (faint)
            for trace in r.traces[:50]:  # limit to 50 for readability
                ax.plot(trace.time * 1000, trace.values, color=color, alpha=0.08, linewidth=0.5)

            # Plot mean +/- SEM
            t_ms = r.time_common * 1000  # convert to ms
            ax.plot(t_ms, r.mean, color=color, linewidth=2, label=f'Mean (n={r.n_events})')
            ax.fill_between(t_ms, r.mean - r.sem, r.mean + r.sem,
                           alpha=0.3, color=color)

            # Onset line
            ax.axvline(0, color='#ff5555', linestyle='--', linewidth=1.5, alpha=0.7, label='Breath onset')

            ax.set_ylabel(metric_labels[key], fontsize=10)
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='#cccccc', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#444444')
            ax.legend(loc='upper right', fontsize=8, facecolor='#2a2a2a',
                     edgecolor='#444444', labelcolor='#cccccc')
            ax.grid(True, alpha=0.15, color='#444444')

        axes[-1].set_xlabel('Time from breath onset (ms)', fontsize=10, color='#cccccc')

        # Save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "cardiorespiratory_cta.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)

        print(f"\n  Saved: {out_path}")
        print(f"  File size: {out_path.stat().st_size / 1024:.0f} KB")
        assert out_path.exists()
        assert out_path.stat().st_size > 5000
