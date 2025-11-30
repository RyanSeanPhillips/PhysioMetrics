"""
Save Metadata Dialog for PhysioMetrics.

This dialog provides a structured interface for collecting metadata about analyzed
breath data before saving, including experimental details like mouse strain, virus,
location, stimulation parameters, and animal information.
"""

import sys
import re
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox, QLabel,
    QCheckBox, QDialogButtonBox, QCompleter, QGridLayout, QWidget
)
from PyQt6.QtCore import Qt as QtCore_Qt


class SaveMetaDialog(QDialog):
    def __init__(self, abf_name: str, channel: str, parent=None, auto_stim: str = "", history: dict = None, last_values: dict = None):
        super().__init__(parent)
        self.setWindowTitle("Save analyzed data — name builder")

        self._abf_name = abf_name
        self._channel = channel
        self._history = history or {}
        self._last_values = last_values or {}

        lay = QFormLayout(self)

        # Mouse Strain with autocomplete
        self.le_strain = QLineEdit(self)
        self.le_strain.setPlaceholderText("e.g., VgatCre")
        if self._last_values.get('strain'):
            self.le_strain.setText(self._last_values['strain'])
        if self._history.get('strain'):
            completer = QCompleter(self._history['strain'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_strain.setCompleter(completer)
        lay.addRow("Mouse Strain:", self.le_strain)

        # Virus with autocomplete
        self.le_virus = QLineEdit(self)
        self.le_virus.setPlaceholderText("e.g., ConFoff-ChR2")
        if self._last_values.get('virus'):
            self.le_virus.setText(self._last_values['virus'])
        if self._history.get('virus'):
            completer = QCompleter(self._history['virus'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_virus.setCompleter(completer)
        lay.addRow("Virus:", self.le_virus)

        # Location with autocomplete
        self.le_location = QLineEdit(self)
        self.le_location.setPlaceholderText("e.g., preBotC or RTN")
        if self._last_values.get('location'):
            self.le_location.setText(self._last_values['location'])
        if self._history.get('location'):
            completer = QCompleter(self._history['location'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_location.setCompleter(completer)
        lay.addRow("Location:", self.le_location)

        # Stimulation type (can be auto-populated) with autocomplete
        self.le_stim = QLineEdit(self)
        self.le_stim.setPlaceholderText("e.g., 20Hz10s15ms or 15msPulse")
        # Prefer auto_stim if available, otherwise use last value
        if auto_stim:
            self.le_stim.setText(auto_stim)
        elif self._last_values.get('stim'):
            self.le_stim.setText(self._last_values['stim'])
        if self._history.get('stim'):
            completer = QCompleter(self._history['stim'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_stim.setCompleter(completer)
        lay.addRow("Stimulation type:", self.le_stim)

        # Laser power with autocomplete
        self.le_power = QLineEdit(self)
        self.le_power.setPlaceholderText("e.g., 8mW")
        if self._last_values.get('power'):
            self.le_power.setText(self._last_values['power'])
        if self._history.get('power'):
            completer = QCompleter(self._history['power'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_power.setCompleter(completer)
        lay.addRow("Laser power:", self.le_power)

        self.cb_sex = QComboBox(self)
        self.cb_sex.addItems(["", "M", "F", "Unknown"])
        # Set last sex value if available
        if self._last_values.get('sex'):
            idx = self.cb_sex.findText(self._last_values['sex'])
            if idx >= 0:
                self.cb_sex.setCurrentIndex(idx)
        lay.addRow("Sex:", self.cb_sex)

        # Animal ID with autocomplete
        self.le_animal = QLineEdit(self)
        self.le_animal.setPlaceholderText("e.g., 25121004")
        if self._last_values.get('animal'):
            self.le_animal.setText(self._last_values['animal'])
        if self._history.get('animal'):
            completer = QCompleter(self._history['animal'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_animal.setCompleter(completer)
        lay.addRow("Animal ID:", self.le_animal)

        # Read-only info
        self.lbl_abf = QLabel(abf_name, self)
        self.lbl_chn = QLabel(channel or "", self)
        lay.addRow("ABF file:", self.lbl_abf)
        lay.addRow("Channel:", self.lbl_chn)

        # Experiment type selection
        self.cb_experiment_type = QComboBox(self)
        self.cb_experiment_type.addItems([
            "30Hz Stimulus (default)",
            "Hargreaves Thermal Sensitivity",
            "Licking Behavior"
        ])
        self.cb_experiment_type.setToolTip(
            "Select experiment type to determine export format:\n"
            "• 30Hz Stimulus: Standard time series aligned to stim onset\n"
            "• Hargreaves: Metrics aligned to heat onset/withdrawal\n"
            "• Licking: Comparison of during vs outside licking bouts"
        )
        lay.addRow("Experiment Type:", self.cb_experiment_type)

        # Live preview of filename
        self.lbl_preview = QLabel("", self)
        self.lbl_preview.setStyleSheet("color:#b6bfda;")
        lay.addRow("File Name Preview:", self.lbl_preview)

        # File export options section (compact grid layout, spans both columns)
        lay.addRow("", QLabel(""))  # Spacer

        # Create container widget for grid layout (will span both columns)
        export_container = QWidget(self)
        export_container_layout = QGridLayout(export_container)
        export_container_layout.setContentsMargins(0, 0, 0, 0)
        export_container_layout.setSpacing(5)

        export_label = QLabel("<b>Files to Export:</b>")
        export_label.setStyleSheet("font-size: 10pt;")
        export_container_layout.addWidget(export_label, 0, 0, 1, 3)  # Spans 3 columns

        # Create inner widget for checkbox grid
        export_widget = QWidget(self)
        export_grid = QGridLayout(export_widget)
        export_grid.setContentsMargins(0, 0, 0, 0)
        export_grid.setSpacing(8)

        # Create checkboxes with smaller font
        small_font = "font-size: 9pt;"

        self.chk_save_npz = QCheckBox("NPZ Bundle*", self)
        self.chk_save_npz.setChecked(True)
        self.chk_save_npz.setEnabled(False)  # Always required
        self.chk_save_npz.setToolTip("Binary data bundle - always saved (fast, ~0.5s)")
        self.chk_save_npz.setStyleSheet(small_font)

        self.chk_save_timeseries = QCheckBox("Timeseries CSV", self)
        self.chk_save_timeseries.setChecked(True)
        self.chk_save_timeseries.setToolTip("Time-aligned metric traces (~9s)")
        self.chk_save_timeseries.setStyleSheet(small_font)

        self.chk_save_breaths = QCheckBox("Breaths CSV", self)
        self.chk_save_breaths.setChecked(True)
        self.chk_save_breaths.setToolTip("Per-breath metrics by region (~1-2s)")
        self.chk_save_breaths.setStyleSheet(small_font)

        self.chk_save_events = QCheckBox("Events CSV", self)
        self.chk_save_events.setChecked(True)
        self.chk_save_events.setToolTip("Apnea/eupnea/sniffing intervals (~0.5s)")
        self.chk_save_events.setStyleSheet(small_font)

        self.chk_save_pdf = QCheckBox("Summary PDF", self)
        self.chk_save_pdf.setChecked(True)
        self.chk_save_pdf.setToolTip("Visualization plots (~31s - can skip for quick exports)")
        self.chk_save_pdf.setStyleSheet(small_font)

        self.chk_save_session = QCheckBox("Session State", self)
        self.chk_save_session.setChecked(True)
        self.chk_save_session.setToolTip("Save analysis session (.pleth.npz) - allows resuming work later (~8 MB, <1s)")
        self.chk_save_session.setStyleSheet(small_font)

        self.chk_save_ml_training = QCheckBox("ML Training Data", self)
        self.chk_save_ml_training.setChecked(False)  # Off by default (optional feature)
        self.chk_save_ml_training.setToolTip("Export peak metrics + user edits for ML model training (saved to ML_Training_Data folder as .npz)")
        self.chk_save_ml_training.setStyleSheet(small_font)

        self.chk_ml_include_waveforms = QCheckBox("+ Waveform Cutouts", self)
        self.chk_ml_include_waveforms.setChecked(False)  # Off by default (increases file size)
        self.chk_ml_include_waveforms.setToolTip("Include raw waveform segments around each peak for neural network training (~10x larger files)")
        self.chk_ml_include_waveforms.setStyleSheet(small_font + " padding-left: 20px;")  # Indent to show it's a sub-option
        self.chk_ml_include_waveforms.setEnabled(False)  # Disabled until ML training is checked

        # Enable/disable waveform checkbox based on ML training checkbox
        self.chk_save_ml_training.toggled.connect(self.chk_ml_include_waveforms.setEnabled)

        # Arrange in 3 columns x 4 rows (added rows for ML data)
        # Row 0: NPZ, Timeseries, Breaths
        export_grid.addWidget(self.chk_save_npz, 0, 0)
        export_grid.addWidget(self.chk_save_timeseries, 0, 1)
        export_grid.addWidget(self.chk_save_breaths, 0, 2)
        # Row 1: Events, PDF, Session
        export_grid.addWidget(self.chk_save_events, 1, 0)
        export_grid.addWidget(self.chk_save_pdf, 1, 1)
        export_grid.addWidget(self.chk_save_session, 1, 2)
        # Row 2: ML Training Data
        export_grid.addWidget(self.chk_save_ml_training, 2, 0, 1, 2)  # Spans 2 columns
        # Row 3: Waveform cutouts (sub-option, indented)
        export_grid.addWidget(self.chk_ml_include_waveforms, 3, 0, 1, 3)  # Spans all columns

        # Add checkbox grid to container
        export_container_layout.addWidget(export_widget, 1, 0, 1, 3)  # Spans 3 columns

        # Add container to form layout (single argument = span both columns)
        lay.addRow(export_container)
        lay.addRow("", QLabel(""))  # Spacer

        # NEW: choose location checkbox
        self.cb_choose_dir = QCheckBox("Let me choose where to save", self)
        self.cb_choose_dir.setToolTip("If unchecked, files go to a 'Pleth_App_Analysis' folder automatically.")
        lay.addRow("", self.cb_choose_dir)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        lay.addRow(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        # Update preview on change
        self.le_strain.textChanged.connect(self._update_preview)
        self.le_virus.textChanged.connect(self._update_preview)
        self.le_location.textChanged.connect(self._update_preview)
        self.le_stim.textChanged.connect(self._update_preview)
        self.le_power.textChanged.connect(self._update_preview)
        self.cb_sex.currentTextChanged.connect(self._update_preview)
        self.le_animal.textChanged.connect(self._update_preview)

        self._update_preview()

        # Apply dark theme styling
        self._apply_dark_theme()

        # Enable dark title bar on Windows
        self._enable_dark_title_bar()

    def _apply_dark_theme(self):
        """Apply dark theme styling to the dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
                background-color: transparent;
            }
            QLineEdit {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QLineEdit:focus {
                border-color: #2a7fff;
            }
            QLineEdit::placeholder {
                color: #888888;
            }
            QComboBox {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 80px;
            }
            QComboBox:focus {
                border-color: #2a7fff;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #888888;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                selection-background-color: #2a7fff;
                selection-color: white;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2a2a2a;
            }
            QCheckBox::indicator:checked {
                background-color: #2a7fff;
                border-color: #2a7fff;
            }
            QCheckBox::indicator:hover {
                border-color: #2a7fff;
            }
            QCheckBox::indicator:disabled {
                background-color: #1a1a1a;
                border-color: #444;
            }
            QCheckBox:disabled {
                color: #666666;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #2a7fff;
            }
            QPushButton:pressed {
                background-color: #2a7fff;
            }
            QPushButton:default {
                border-color: #2a7fff;
            }
            QDialogButtonBox {
                background-color: transparent;
            }
        """)

        # Update the preview label color for dark theme
        self.lbl_preview.setStyleSheet("color: #88aaff; background-color: transparent;")

    def _enable_dark_title_bar(self):
        """Enable dark title bar on Windows 10/11."""
        if sys.platform == "win32":
            try:
                from ctypes import windll, byref, sizeof, c_int
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                hwnd = int(self.winId())
                value = c_int(1)
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, byref(value), sizeof(value)
                )
            except Exception:
                pass

    # --- Helpers: light canonicalization + sanitization ---
    def _norm_token(self, s: str) -> str:
        s0 = (s or "").strip()
        if not s0:
            return ""
        s1 = s0.replace(" ", "")
        s1 = re.sub(r"(?i)chr\s*2", "ChR2", s1)            # chr2 -> ChR2
        s1 = re.sub(r"(?i)gcamp\s*6f", "GCaMP6f", s1)      # gcamp6f -> GCaMP6f
        s1 = re.sub(r"(?i)([A-Za-z0-9_-]*?)cre$", lambda m: (m.group(1) or "") + "Cre", s1)  # ...cre -> ...Cre
        return s1

    def _san(self, s: str) -> str:
        s = (s or "").strip()
        s = s.replace(" ", "_")
        s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
        s = re.sub(r"_+", "_", s)
        s = re.sub(r"-+", "-", s)
        return s

    def _update_preview(self):
        # Read & normalize
        strain = self._norm_token(self.le_strain.text())
        virus  = self._norm_token(self.le_virus.text())
        location = self.le_location.text().strip()

        stim   = self.le_stim.text().strip()
        power  = self.le_power.text().strip()
        sex    = self.cb_sex.currentText().strip()
        animal = self.le_animal.text().strip()
        abf    = self._abf_name
        ch     = self._channel

        # Sanitize for filename
        strain_s = self._san(strain)
        virus_s  = self._san(virus)
        location_s = self._san(location)
        stim_s   = self._san(stim)
        power_s  = self._san(power)
        sex_s    = self._san(sex)
        animal_s = self._san(animal)
        abf_s    = self._san(abf)
        ch_s     = self._san(ch)

        # STANDARD ORDER:
        # Strain_Virus_Location_Sex_Animal_Stim_Power_ABF_Channel
        parts = [p for p in (strain_s, virus_s, location_s, sex_s, animal_s, stim_s, power_s, abf_s, ch_s) if p]
        preview = "_".join(parts) if parts else "analysis"
        self.lbl_preview.setText(preview)

    def values(self) -> dict:
        # Map friendly names to internal experiment type codes
        exp_type_map = {
            "30Hz Stimulus (default)": "30hz_stim",
            "Hargreaves Thermal Sensitivity": "hargreaves",
            "Licking Behavior": "licking"
        }
        experiment_type = exp_type_map.get(self.cb_experiment_type.currentText(), "30hz_stim")

        return {
            "strain": self.le_strain.text().strip(),
            "virus":  self.le_virus.text().strip(),
            "location": self.le_location.text().strip(),
            "stim":   self.le_stim.text().strip(),
            "power":  self.le_power.text().strip(),
            "sex":    self.cb_sex.currentText().strip(),
            "animal": self.le_animal.text().strip(),
            "abf":    self._abf_name,
            "chan":   self._channel,
            "preview": self.lbl_preview.text().strip(),
            "choose_dir": bool(self.cb_choose_dir.isChecked()),
            "experiment_type": experiment_type,
            # File export options
            "save_npz": True,  # Always true (checkbox disabled)
            "save_timeseries_csv": bool(self.chk_save_timeseries.isChecked()),
            "save_breaths_csv": bool(self.chk_save_breaths.isChecked()),
            "save_events_csv": bool(self.chk_save_events.isChecked()),
            "save_pdf": bool(self.chk_save_pdf.isChecked()),
            "save_session": bool(self.chk_save_session.isChecked()),
            "save_ml_training": bool(self.chk_save_ml_training.isChecked()),
            "ml_include_waveforms": bool(self.chk_ml_include_waveforms.isChecked()),
        }
