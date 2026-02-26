"""
BatchAnalysisViewModel — orchestrates batch analysis from the Projects tab.

Runs analyze_file() in parallel using ThreadPoolExecutor, emitting
per-file progress signals so the UI can update rows in real time.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from core.domain.analysis.models import AnalysisConfig, AnalysisResult

# Cap workers to avoid memory pressure (each file loads full sweeps into RAM)
MAX_WORKERS = min(os.cpu_count() or 4, 4)


class BatchAnalysisViewModel(QObject):
    """ViewModel for batch analysis operations."""

    # Signals
    batch_started = pyqtSignal(int)                  # total file count
    file_started = pyqtSignal(int, str)              # index, file_name
    file_progress = pyqtSignal(int, str)             # index, detail_message (per-file stage)
    file_completed = pyqtSignal(int, dict, object)   # index, exp_dict, AnalysisResult
    batch_finished = pyqtSignal(list)                 # list of (exp_dict, AnalysisResult)
    batch_error = pyqtSignal(str)                     # fatal error message
    progress_message = pyqtSignal(str)                # status text updates

    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self._store = store
        self._worker = None
        self._cancel_event = threading.Event()

    def run_batch(
        self,
        experiments: List[Dict[str, Any]],
        config: AnalysisConfig,
        dry_run: bool = True,
    ):
        """Start batch analysis on a list of experiment dicts.

        Each experiment dict should have at least 'file_path' and optionally
        'channel' (e.g. "IN 0") for per-row channel override.

        Args:
            experiments: List of experiment row dicts from the DB/model.
            config: Analysis configuration to apply to all files.
            dry_run: If True, skip CSV writing.
        """
        if self._worker is not None and self._worker.isRunning():
            self.batch_error.emit("Batch analysis already running")
            return

        self._cancel_event.clear()

        from core.file_load_worker import FileLoadWorker
        self._worker = FileLoadWorker(
            self._run_parallel,
            experiments, config, dry_run,
            inject_progress=False,
        )
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self.batch_started.emit(len(experiments))
        self._worker.start()

    def cancel(self):
        """Request cancellation of the current batch."""
        self._cancel_event.set()

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _analyze_one(
        self, idx: int, exp: Dict[str, Any], config: AnalysisConfig, dry_run: bool
    ) -> tuple:
        """Analyze a single experiment. Called from thread pool workers."""
        from core.services.analysis_service import analyze_file

        file_path = Path(exp.get("file_path", ""))
        file_name = exp.get("file_name", file_path.name)
        channel = exp.get("channel") or None

        self.file_started.emit(idx, file_name)

        if not file_path.exists():
            result = AnalysisResult(file_path=file_path)
            result.error = f"File not found: {file_path}"
        else:
            def _progress(msg, _idx=idx, _name=file_name):
                self.progress_message.emit(f"[{_idx+1}] {_name}: {msg}")
                # Emit per-file stage for row-level progress
                self.file_progress.emit(_idx, msg)

            result = analyze_file(
                path=file_path,
                config=config,
                write_csv=not dry_run,
                analyze_channel=channel,
                progress_callback=_progress,
            )

        return idx, exp, result

    def _run_parallel(
        self,
        experiments: List[Dict[str, Any]],
        config: AnalysisConfig,
        dry_run: bool,
    ) -> List[tuple]:
        """Worker function — runs in FileLoadWorker thread, dispatches to pool."""
        n_workers = min(MAX_WORKERS, len(experiments))
        # For very small batches, just run sequentially (less overhead)
        if n_workers <= 1:
            return self._run_sequential(experiments, config, dry_run)

        self.progress_message.emit(
            f"Analyzing {len(experiments)} files with {n_workers} workers..."
        )

        # results keyed by idx for ordered output
        results_by_idx: Dict[int, tuple] = {}

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for idx, exp in enumerate(experiments):
                if self._cancel_event.is_set():
                    break
                future = pool.submit(self._analyze_one, idx, exp, config, dry_run)
                futures[future] = idx

            for future in as_completed(futures):
                if self._cancel_event.is_set():
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

                idx, exp, result = future.result()
                results_by_idx[idx] = (exp, result)
                self.file_completed.emit(idx, exp, result)

                # Update DB status
                if self._store and exp.get("id"):
                    status = "completed" if result.error is None else "error"
                    try:
                        self._store.update_experiment(
                            exp["id"], {"status": status}
                        )
                    except Exception:
                        pass

        # Return in original order
        return [results_by_idx[i] for i in sorted(results_by_idx)]

    def _run_sequential(
        self,
        experiments: List[Dict[str, Any]],
        config: AnalysisConfig,
        dry_run: bool,
    ) -> List[tuple]:
        """Fallback sequential execution."""
        results = []
        for idx, exp in enumerate(experiments):
            if self._cancel_event.is_set():
                break
            _, exp_out, result = self._analyze_one(idx, exp, config, dry_run)
            results.append((exp_out, result))

            if self._store and exp.get("id"):
                status = "completed" if result.error is None else "error"
                try:
                    self._store.update_experiment(
                        exp["id"], {"status": status}
                    )
                except Exception:
                    pass

        return results

    def _on_worker_finished(self, results):
        """Called when the worker thread completes."""
        self.batch_finished.emit(results if results else [])
        self._worker = None

    def _on_worker_error(self, error_msg):
        """Called on uncaught worker thread error."""
        self.batch_error.emit(error_msg)
        self._worker = None
