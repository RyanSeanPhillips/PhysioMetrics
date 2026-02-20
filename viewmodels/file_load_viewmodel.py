"""
File loading view model.

QObject that orchestrates file loading operations, owns FileLoadWorker
instances, and emits signals for the view layer to update the UI.

No direct widget manipulation — all UI updates happen via signal connections.
"""

import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from core.domain.file_loading.models import (
    FileLoadResult,
    MultiFileLoadResult,
    NpzLoadResult,
    ChannelAutoDetection,
)
from core.services.file_load_service import FileLoadService


class FileLoadViewModel(QObject):
    """
    ViewModel for file loading operations.

    Owns worker threads and emits typed signals when loading completes.
    The view layer (MainWindow) connects to these signals for UI updates.
    """

    # --- Signals ---
    file_loaded = pyqtSignal(object)            # FileLoadResult
    multi_file_loaded = pyqtSignal(object)      # MultiFileLoadResult
    npz_loaded = pyqtSignal(object)             # NpzLoadResult
    npz_load_error = pyqtSignal(object)         # Exception (typed — e.g. OriginalFileNotFoundError)
    photometry_loaded = pyqtSignal(object)      # result_data dict
    photometry_load_error = pyqtSignal(str)     # error message
    load_error = pyqtSignal(str, str)           # title, message
    loading_started = pyqtSignal(str)           # description (for progress UI)
    loading_finished = pyqtSignal()
    progress_updated = pyqtSignal(int, str)     # percent (0-100), message

    def __init__(
        self,
        service: FileLoadService,
        state,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._service = service
        self._state = state

        # Worker lifecycle — prevent GC
        self._load_worker = None

        # Loading context
        self._loading_path: Optional[Path] = None
        self._loading_multi_paths: Optional[list[Path]] = None
        self._loading_npz_path: Optional[Path] = None
        self._loading_npz_metadata: Optional[dict] = None
        self._loading_t_start: float = 0.0

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: Path) -> None:
        """Load a single data file (ABF, SMRX, EDF) in a background thread."""
        from core import abf_io
        from core.file_load_worker import FileLoadWorker

        file_type = path.suffix.upper()[1:]
        self._loading_path = path
        self._loading_t_start = time.time()

        self.loading_started.emit(f"Opening {file_type} File")

        self._load_worker = FileLoadWorker(abf_io.load_data_file, path)
        self._load_worker.progress.connect(
            lambda c, t, m: self.progress_updated.emit(c, f"{m}\n{path.name}")
        )
        self._load_worker.finished.connect(self._on_single_file_loaded)
        self._load_worker.error.connect(
            lambda msg: (
                self.loading_finished.emit(),
                self.load_error.emit("Load error", msg.split('\n\n')[0]),
            )
        )
        self._load_worker.start()

    def load_multiple_files(self, file_paths: list[Path]) -> None:
        """Load and concatenate multiple data files in a background thread."""
        from core import abf_io
        from core.file_load_worker import FileLoadWorker

        self._loading_multi_paths = file_paths
        self._loading_t_start = time.time()

        self.loading_started.emit("Loading Multiple Files")

        self._load_worker = FileLoadWorker(abf_io.load_and_concatenate_abf_files, file_paths)
        self._load_worker.progress.connect(
            lambda c, t, m: self.progress_updated.emit(c, m)
        )
        self._load_worker.finished.connect(self._on_multi_file_loaded)
        self._load_worker.error.connect(
            lambda msg: (
                self.loading_finished.emit(),
                self.load_error.emit("Load error", msg.split('\n\n')[0]),
            )
        )
        self._load_worker.start()

    def load_npz_state(
        self,
        npz_path: Path,
        alternative_data_path: Optional[Path] = None,
        master_file_list: Optional[list[dict]] = None,
    ) -> None:
        """Load a .pleth.npz session file in a background thread."""
        from core.npz_io import load_state_from_npz, get_npz_metadata
        from core.file_load_worker import FileLoadWorker

        t_start = time.time()

        # Get metadata (fast — reads only header)
        metadata = get_npz_metadata(npz_path)
        if 'error' in metadata:
            self.load_error.emit("Load Error", f"Failed to read NPZ file:\n\n{metadata['error']}")
            return

        # Resolve alternative path if original doesn't exist
        if alternative_data_path is None:
            alternative_data_path = self._service.resolve_npz_data_path(
                metadata, master_file_list
            )

        self._loading_npz_path = npz_path
        self._loading_npz_metadata = metadata
        self._loading_t_start = t_start

        self.loading_started.emit("Loading PhysioMetrics Session")
        self.progress_updated.emit(30, f"Reading session file...\n{npz_path.name}")

        self._load_worker = FileLoadWorker(
            load_state_from_npz, npz_path,
            reload_raw_data=True, alternative_data_path=alternative_data_path,
        )
        self._load_worker.finished.connect(self._on_npz_loaded)
        self._load_worker.error_exc.connect(self._on_npz_load_error_internal)
        self._load_worker.start()

    def load_photometry_npz(self, npz_path: Path, exp_idx: int) -> None:
        """Load a photometry experiment from NPZ in a background thread."""
        from core import photometry
        from core.file_load_worker import FileLoadWorker

        self.loading_started.emit("Loading Photometry Data")

        self._load_worker = FileLoadWorker(
            photometry.load_experiment_from_npz, npz_path, exp_idx
        )
        self._load_worker.finished.connect(self._on_photometry_npz_loaded)
        self._load_worker.error.connect(
            lambda msg: (
                self.loading_finished.emit(),
                self.photometry_load_error.emit(
                    f"Failed to load photometry experiment:\n\n{msg.split(chr(10))[0]}"
                ),
            )
        )
        self._load_worker.start()

    # ------------------------------------------------------------------
    # Internal completion handlers
    # ------------------------------------------------------------------

    def _on_single_file_loaded(self, result) -> None:
        """Handle single file load completion."""
        sr, sweeps_by_ch, ch_names, t, file_metadata = result

        path = self._loading_path
        load_duration = time.time() - self._loading_t_start

        file_result = FileLoadResult(
            sr_hz=sr,
            sweeps_by_ch=sweeps_by_ch,
            channel_names=ch_names,
            time_array=t,
            file_metadata=file_metadata,
            source_path=path,
            load_duration_seconds=load_duration,
        )

        # Apply to state
        self._service.apply_single_file_data(self._state, file_result)
        self._service.reset_analysis_state(self._state)

        self.loading_finished.emit()
        self.file_loaded.emit(file_result)

    def _on_multi_file_loaded(self, result) -> None:
        """Handle multi-file load completion."""
        sr, sweeps_by_ch, ch_names, t, file_info = result

        file_paths = self._loading_multi_paths
        load_duration = time.time() - self._loading_t_start

        multi_result = MultiFileLoadResult(
            sr_hz=sr,
            sweeps_by_ch=sweeps_by_ch,
            channel_names=ch_names,
            time_array=t,
            file_info=file_info,
            source_paths=file_paths,
        )

        # Apply to state
        self._service.apply_multi_file_data(self._state, multi_result)
        self._service.reset_analysis_state(self._state)

        self.loading_finished.emit()
        self.multi_file_loaded.emit(multi_result)

    def _on_npz_loaded(self, result) -> None:
        """Handle NPZ load completion."""
        new_state, raw_data_loaded, gmm_cache, app_settings, event_markers = result

        npz_result = NpzLoadResult(
            new_state=new_state,
            raw_data_loaded=raw_data_loaded,
            gmm_cache=gmm_cache,
            app_settings=app_settings,
            event_markers=event_markers,
            npz_path=self._loading_npz_path,
            metadata=self._loading_npz_metadata,
        )

        self.loading_finished.emit()
        self.npz_loaded.emit(npz_result)

    def _on_npz_load_error_internal(self, exc) -> None:
        """Handle NPZ load error — forward typed exception to view."""
        self.loading_finished.emit()
        self.npz_load_error.emit(exc)

    def _on_photometry_npz_loaded(self, result_data) -> None:
        """Handle photometry NPZ load completion."""
        self.loading_finished.emit()
        if result_data:
            self.photometry_loaded.emit(result_data)
        else:
            self.photometry_load_error.emit("Failed to load experiment from NPZ file.")
