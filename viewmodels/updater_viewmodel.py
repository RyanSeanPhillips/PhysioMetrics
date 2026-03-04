"""
ViewModel for the auto-update system.

Manages download state, progress signals, and restart/rollback actions.
"""

import os
import subprocess

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class _DownloadWorker(QThread):
    """Background thread for downloading and preparing an update."""

    progress = pyqtSignal(int, int)  # bytes_downloaded, total_bytes
    finished = pyqtSignal(str)       # path to update script
    error = pyqtSignal(str)          # error message

    def __init__(self, asset_info: dict, parent=None):
        super().__init__(parent)
        self.asset_info = asset_info

    def run(self):
        try:
            from core import auto_updater

            url = self.asset_info['download_url']
            filename = self.asset_info['file_name']
            size = self.asset_info.get('size', 0)

            # Download
            zip_path = auto_updater.download_update(
                url, filename, size,
                progress_callback=lambda dl, total: self.progress.emit(dl, total),
            )

            # Verify + extract + generate script
            script_path = auto_updater.apply_update(zip_path)
            self.finished.emit(str(script_path))

        except Exception as e:
            self.error.emit(str(e))


class UpdaterViewModel(QObject):
    """
    ViewModel for auto-update UI.

    Signals:
        download_progress(int, int): bytes downloaded, total bytes
        download_complete(str): path to update script
        download_error(str): error message
        state_changed(str): idle/downloading/extracting/ready/error
    """

    download_progress = pyqtSignal(int, int)
    download_complete = pyqtSignal(str)
    download_error = pyqtSignal(str)
    state_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = 'idle'
        self._script_path = None
        self._worker = None

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, state: str):
        self._state = state
        self.state_changed.emit(state)

    def start_download(self, asset_info: dict):
        """Start downloading an update in background."""
        if self._state == 'downloading':
            return

        self._set_state('downloading')

        self._worker = _DownloadWorker(asset_info, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, downloaded: int, total: int):
        self.download_progress.emit(downloaded, total)

    def _on_finished(self, script_path: str):
        self._script_path = script_path
        self._set_state('ready')
        self.download_complete.emit(script_path)

    def _on_error(self, message: str):
        self._set_state('error')
        self.download_error.emit(message)

    def apply_and_restart(self):
        """Launch the update script and quit the app."""
        if not self._script_path:
            return

        # Launch batch script detached
        CREATE_NEW_CONSOLE = 0x00000010
        DETACHED_PROCESS = 0x00000008
        subprocess.Popen(
            ['cmd', '/c', self._script_path],
            creationflags=CREATE_NEW_CONSOLE | DETACHED_PROCESS,
            close_fds=True,
        )

        # Quit the application
        from PyQt6.QtWidgets import QApplication
        QApplication.quit()

    def apply_rollback(self):
        """Generate rollback script, launch it, and quit."""
        from core import auto_updater

        install_dir = auto_updater.get_install_dir()
        backup_dir = auto_updater._get_backup_dir()

        if install_dir is None or not auto_updater.has_backup():
            return

        script = auto_updater.generate_rollback_script(
            install_dir, backup_dir, os.getpid()
        )

        CREATE_NEW_CONSOLE = 0x00000010
        DETACHED_PROCESS = 0x00000008
        subprocess.Popen(
            ['cmd', '/c', str(script)],
            creationflags=CREATE_NEW_CONSOLE | DETACHED_PROCESS,
            close_fds=True,
        )

        from PyQt6.QtWidgets import QApplication
        QApplication.quit()
