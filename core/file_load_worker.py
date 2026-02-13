"""
FileLoadWorker - Background QThread for all file loading operations.

Wraps any callable (loader function) and runs it in a background thread,
emitting progress/finished/error signals so the UI stays responsive.

Pattern follows MetadataThread in core/scan_manager.py.
"""

import inspect

from PyQt6.QtCore import QThread, pyqtSignal


class FileLoadWorker(QThread):
    """Background thread for all file loading operations."""

    progress = pyqtSignal(int, int, str)   # current, total, message
    finished = pyqtSignal(object)          # result data (tuple/dict depending on load type)
    error = pyqtSignal(str)                # error message string
    error_exc = pyqtSignal(object)         # actual exception object (for typed exception handling)

    def __init__(self, load_func, *args, inject_progress=True, **kwargs):
        """
        Args:
            load_func: The callable to run in the background thread
                       (e.g., abf_io.load_data_file, npz_io.load_state_from_npz, etc.)
            inject_progress: If True, automatically inject progress_callback kwarg
                            when the function signature accepts it. Set False for
                            functions that don't support progress callbacks.
            *args, **kwargs: Arguments passed to load_func
        """
        super().__init__()
        self._load_func = load_func
        self._args = args
        self._kwargs = kwargs
        self._inject_progress = inject_progress

    def run(self):
        try:
            # Only inject progress_callback if requested and function accepts it
            if self._inject_progress:
                try:
                    sig = inspect.signature(self._load_func)
                    if 'progress_callback' in sig.parameters:
                        self._kwargs['progress_callback'] = self._emit_progress
                except (ValueError, TypeError):
                    pass

            result = self._load_func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            # Emit both the exception object and the string message
            self.error_exc.emit(e)
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")

    def _emit_progress(self, current, total, message):
        self.progress.emit(current, total, message)
