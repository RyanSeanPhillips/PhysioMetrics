"""
Save Metadata Dialog for PhysioMetrics.

This dialog provides a structured interface for collecting metadata about analyzed
breath data before saving, including experimental details like mouse strain, virus,
location, stimulation parameters, and animal information.

The dialog is non-modal, allowing users to click on the main window (e.g., Project
Builder tab) while the dialog is open to reference file information.

Features a collapsible notes preview panel that shows where the ABF file reference
was found in linked notes files (lazy-loaded for performance).
"""

import sys
import re
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox, QLabel,
    QCheckBox, QDialogButtonBox, QCompleter, QGridLayout, QWidget,
    QPushButton, QHBoxLayout, QFileDialog, QVBoxLayout, QGroupBox,
    QFrame, QSplitter, QSizePolicy, QTabWidget, QSpinBox
)
from PyQt6.QtCore import Qt as QtCore_Qt, QSettings, pyqtSignal, QTimer
from PyQt6.QtGui import QShortcut, QKeySequence


class SaveMetaDialog(QDialog):
    """Non-modal dialog for collecting save metadata.

    Emits accepted_with_values signal when user clicks OK, passing the values dict.
    This allows the dialog to be non-modal while still communicating results.
    """

    # Signal emitted when user accepts with values dict
    accepted_with_values = pyqtSignal(dict)

    # Fixed width for form panel (same as original dialog)
    FORM_PANEL_WIDTH = 420
    # Width for preview panel (approximately 2/3 of total when shown)
    PREVIEW_PANEL_WIDTH = 700

    def __init__(self, abf_name: str, channel: str, parent=None, auto_stim: str = "",
                 history: dict = None, last_values: dict = None, main_window=None,
                 file_info: dict = None):
        super().__init__(parent)
        self.setWindowTitle("Save analyzed data — name builder")

        # Store main window reference for accessing project data
        self._main_window = main_window

        self._abf_name = abf_name
        self._channel = channel
        self._history = history or {}
        self._last_values = last_values or {}
        self._file_info = file_info or {}

        # Track preview state
        self._preview_loaded = False
        self._preview_visible = False
        self._linked_notes = []
        self._highlight_stem = ""
        self._is_fuzzy_match = False

        # Search debounce timer (delays search until user stops typing)
        self._search_debounce_timer = QTimer(self)
        self._search_debounce_timer.setSingleShot(True)
        self._search_debounce_timer.timeout.connect(self._perform_search_reload)

        # Find linked notes (fast - just searches cached index, no file I/O)
        self._find_linked_notes()

        # Create main horizontal layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left panel: Form (fixed width)
        self._form_panel = QWidget()
        self._form_panel.setFixedWidth(self.FORM_PANEL_WIDTH)
        form_layout = QVBoxLayout(self._form_panel)
        form_layout.setContentsMargins(12, 12, 12, 12)

        # Create form content
        self._create_form_content(form_layout, auto_stim)

        main_layout.addWidget(self._form_panel)

        # Right panel: Preview (collapsible)
        self._preview_panel = QWidget()
        self._preview_panel.setVisible(False)
        preview_layout = QVBoxLayout(self._preview_panel)
        preview_layout.setContentsMargins(0, 12, 12, 12)

        self._create_preview_panel(preview_layout)

        main_layout.addWidget(self._preview_panel)

        # Apply dark theme styling
        self._apply_dark_theme()

        # Enable dark title bar on Windows
        self._enable_dark_title_bar()

        # Load preview visibility preference and set size accordingly
        settings = QSettings("PhysioMetrics", "PhysioMetrics")
        show_preview = settings.value("save_dialog/show_preview", True, type=bool)

        if show_preview and self._linked_notes:
            # Set full width with preview and show it immediately
            self.setFixedWidth(self.FORM_PANEL_WIDTH + self.PREVIEW_PANEL_WIDTH)
            self._preview_panel.setVisible(True)
            self._preview_panel.setFixedWidth(self.PREVIEW_PANEL_WIDTH)
            self._preview_visible = True
            if hasattr(self, 'btn_toggle_preview'):
                self.btn_toggle_preview.setText(f"Notes ({len(self._linked_notes)}) ◀")
            # Delay loading preview content until after dialog is shown
            QTimer.singleShot(50, self._load_preview_content)
        else:
            # Form only
            self.setFixedWidth(self.FORM_PANEL_WIDTH)

        # Center dialog on screen
        self._center_on_screen()

    def _center_on_screen(self):
        """Center the dialog on the screen."""
        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            dialog_geometry = self.frameGeometry()
            center_point = screen_geometry.center()
            dialog_geometry.moveCenter(center_point)
            self.move(dialog_geometry.topLeft())

    def _find_linked_notes(self):
        """Find notes files that reference the current ABF file.

        This only searches the cached index - no file I/O occurs here.
        """
        if not self._main_window:
            return

        abf_stem = Path(self._abf_name).stem if self._abf_name else ""
        if not abf_stem:
            return

        self._highlight_stem = abf_stem

        # Find exact matches first
        if hasattr(self._main_window, '_get_notes_for_abf'):
            self._linked_notes = self._main_window._get_notes_for_abf(abf_stem)

        # If no exact matches, try fuzzy matching
        if not self._linked_notes and hasattr(self._main_window, '_get_fuzzy_notes_for_abf'):
            self._linked_notes, fuzzy_stems = self._main_window._get_fuzzy_notes_for_abf(abf_stem)
            if self._linked_notes:
                self._is_fuzzy_match = True
                # Find closest matching stem for highlighting
                if fuzzy_stems:
                    orig_nums = re.findall(r'\d+', abf_stem)
                    if orig_nums:
                        orig_num = int(max(orig_nums, key=len))
                        min_diff = float('inf')
                        for stem in fuzzy_stems:
                            stem_nums = re.findall(r'\d+', stem)
                            if stem_nums:
                                stem_num = int(max(stem_nums, key=len))
                                diff = abs(stem_num - orig_num)
                                if diff < min_diff:
                                    min_diff = diff
                                    self._highlight_stem = stem

    def _create_form_content(self, layout: QVBoxLayout, auto_stim: str):
        """Create the left panel form content."""
        # Use a form layout for the fields
        form = QFormLayout()
        form.setSpacing(8)

        # --- File Info Section ---
        self._add_file_info_section(form)

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
        form.addRow("Mouse Strain:", self.le_strain)

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
        form.addRow("Virus:", self.le_virus)

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
        form.addRow("Location:", self.le_location)

        # Stimulation type with autocomplete
        self.le_stim = QLineEdit(self)
        self.le_stim.setPlaceholderText("e.g., 20Hz10s15ms or 15msPulse")
        if auto_stim:
            self.le_stim.setText(auto_stim)
        elif self._last_values.get('stim'):
            self.le_stim.setText(self._last_values['stim'])
        if self._history.get('stim'):
            completer = QCompleter(self._history['stim'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_stim.setCompleter(completer)
        form.addRow("Stimulation type:", self.le_stim)

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
        form.addRow("Laser power:", self.le_power)

        self.cb_sex = QComboBox(self)
        self.cb_sex.addItems(["", "M", "F", "Unknown"])
        if self._last_values.get('sex'):
            idx = self.cb_sex.findText(self._last_values['sex'])
            if idx >= 0:
                self.cb_sex.setCurrentIndex(idx)
        form.addRow("Sex:", self.cb_sex)

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
        form.addRow("Animal ID:", self.le_animal)

        # Read-only info
        self.lbl_abf = QLabel(self._abf_name, self)
        self.lbl_chn = QLabel(self._channel or "", self)
        form.addRow("ABF file:", self.lbl_abf)
        form.addRow("Channel:", self.lbl_chn)

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
        form.addRow("Experiment Type:", self.cb_experiment_type)

        # Live preview of filename
        self.lbl_preview = QLabel("", self)
        self.lbl_preview.setStyleSheet("color: #88aaff; background-color: transparent;")
        form.addRow("File Name Preview:", self.lbl_preview)

        layout.addLayout(form)

        # Spacer
        layout.addSpacing(8)

        # Export options section
        self._add_export_options_section(layout)

        # Spacer
        layout.addSpacing(8)

        # Choose location checkbox
        self.cb_choose_dir = QCheckBox("Let me choose where to save", self)
        self.cb_choose_dir.setToolTip("If unchecked, files go to a 'Pleth_App_Analysis' folder automatically.")
        layout.addWidget(self.cb_choose_dir)

        # Add stretch to push buttons to bottom
        layout.addStretch()

        # Bottom row: Notes toggle button + OK/Cancel
        bottom_layout = QHBoxLayout()

        # Notes preview toggle button (only show if notes are available)
        if self._linked_notes:
            match_type = "fuzzy" if self._is_fuzzy_match else "exact"
            btn_text = f"Notes ({len(self._linked_notes)}) ▶"
            self.btn_toggle_preview = QPushButton(btn_text, self)
            self.btn_toggle_preview.setToolTip(f"Show/hide notes preview ({match_type} match)")
            self.btn_toggle_preview.setFixedWidth(100)
            self.btn_toggle_preview.clicked.connect(self._toggle_preview)
            bottom_layout.addWidget(self.btn_toggle_preview)
        else:
            # Placeholder for alignment
            no_notes_label = QLabel('<span style="color: #666; font-size: 8pt;">No linked notes</span>')
            bottom_layout.addWidget(no_notes_label)

        bottom_layout.addStretch()

        # OK/Cancel buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        bottom_layout.addWidget(btns)

        layout.addLayout(bottom_layout)

        # Connect preview update signals
        self.le_strain.textChanged.connect(self._update_preview)
        self.le_virus.textChanged.connect(self._update_preview)
        self.le_location.textChanged.connect(self._update_preview)
        self.le_stim.textChanged.connect(self._update_preview)
        self.le_power.textChanged.connect(self._update_preview)
        self.cb_sex.currentTextChanged.connect(self._update_preview)
        self.le_animal.textChanged.connect(self._update_preview)

        self._update_preview()

    def _add_file_info_section(self, layout: QFormLayout):
        """Add file info section to the form."""
        info_group = QGroupBox("Current File Info")
        info_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #88aaff;
            }
        """)

        info_layout = QGridLayout(info_group)
        info_layout.setContentsMargins(8, 4, 8, 8)
        info_layout.setSpacing(4)

        label_style = "color: #888; font-size: 9pt;"
        value_style = "color: #e0e0e0; font-size: 9pt; font-weight: bold;"

        # File name
        lbl_file = QLabel("File:")
        lbl_file.setStyleSheet(label_style)
        self.lbl_file_value = QLabel(self._abf_name or "—")
        self.lbl_file_value.setStyleSheet(value_style)
        info_layout.addWidget(lbl_file, 0, 0)
        info_layout.addWidget(self.lbl_file_value, 0, 1)

        # Protocol
        lbl_protocol = QLabel("Protocol:")
        lbl_protocol.setStyleSheet(label_style)
        protocol_value = self._file_info.get('protocol', '—') or '—'
        self.lbl_protocol_value = QLabel(protocol_value)
        self.lbl_protocol_value.setStyleSheet(value_style)
        info_layout.addWidget(lbl_protocol, 0, 2)
        info_layout.addWidget(self.lbl_protocol_value, 0, 3)

        # Keywords
        lbl_keywords = QLabel("Keywords:")
        lbl_keywords.setStyleSheet(label_style)
        keywords_value = self._file_info.get('keywords', '—') or '—'
        self.lbl_keywords_value = QLabel(keywords_value)
        self.lbl_keywords_value.setStyleSheet(value_style)
        self.lbl_keywords_value.setWordWrap(True)
        info_layout.addWidget(lbl_keywords, 1, 0)
        info_layout.addWidget(self.lbl_keywords_value, 1, 1, 1, 3)

        info_layout.setColumnStretch(1, 1)
        info_layout.setColumnStretch(3, 1)

        layout.addRow(info_group)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #3a3a3a; max-height: 1px;")
        layout.addRow(separator)

    def _add_export_options_section(self, layout: QVBoxLayout):
        """Add export options checkboxes."""
        export_label = QLabel("<b>Files to Export:</b>")
        export_label.setStyleSheet("font-size: 10pt;")
        layout.addWidget(export_label)

        # Checkbox grid
        export_widget = QWidget()
        export_grid = QGridLayout(export_widget)
        export_grid.setContentsMargins(0, 4, 0, 0)
        export_grid.setSpacing(8)

        small_font = "font-size: 9pt;"

        # NPZ Bundle - always required
        self.chk_save_npz = QCheckBox("NPZ Bundle*", self)
        self.chk_save_npz.setChecked(True)
        self.chk_save_npz.setEnabled(False)
        self.chk_save_npz.setToolTip("Binary data bundle - always saved (fast, ~0.5s)")
        self.chk_save_npz.setStyleSheet(small_font)

        self.chk_save_timeseries = QCheckBox("Timeseries CSV", self)
        self.chk_save_timeseries.setToolTip("Time-aligned metric traces (~9s)")
        self.chk_save_timeseries.setStyleSheet(small_font)

        self.chk_save_breaths = QCheckBox("Breaths CSV", self)
        self.chk_save_breaths.setToolTip("Per-breath metrics by region (~1-2s)")
        self.chk_save_breaths.setStyleSheet(small_font)

        self.chk_save_events = QCheckBox("Events CSV", self)
        self.chk_save_events.setToolTip("Apnea/eupnea/sniffing intervals (~0.5s)")
        self.chk_save_events.setStyleSheet(small_font)

        self.chk_save_pdf = QCheckBox("Summary PDF", self)
        self.chk_save_pdf.setToolTip("Visualization plots (~31s - can skip for quick exports)")
        self.chk_save_pdf.setStyleSheet(small_font)

        self.chk_save_session = QCheckBox("Session State", self)
        self.chk_save_session.setToolTip("Save analysis session (.pleth.npz) - allows resuming work later")
        self.chk_save_session.setStyleSheet(small_font)

        self.chk_save_ml_training = QCheckBox("ML Training Data", self)
        self.chk_save_ml_training.setToolTip("Export peak metrics + user edits for ML model training")
        self.chk_save_ml_training.setStyleSheet(small_font)

        # Gear button for ML folder
        self.btn_ml_folder_settings = QPushButton("\u2699", self)
        self.btn_ml_folder_settings.setFixedSize(22, 22)
        self.btn_ml_folder_settings.setToolTip("Change ML training data save location")
        self.btn_ml_folder_settings.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 16px;
                color: #888;
                padding: 0px;
            }
            QPushButton:hover { color: #2a7fff; }
        """)
        self.btn_ml_folder_settings.clicked.connect(self._on_configure_ml_folder)

        self.chk_ml_include_waveforms = QCheckBox("+ Waveform Cutouts", self)
        self.chk_ml_include_waveforms.setChecked(False)
        self.chk_ml_include_waveforms.setToolTip("Include raw waveform segments (~10x larger files)")
        self.chk_ml_include_waveforms.setStyleSheet(small_font + " padding-left: 20px;")
        self.chk_ml_include_waveforms.setEnabled(False)

        # ML Labels section (State, Gas, Quality)
        self.ml_labels_group = QGroupBox("ML Labels")
        self.ml_labels_group.setStyleSheet("""
            QGroupBox {
                font-size: 9pt;
                font-weight: bold;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 6px;
                margin-left: 20px;
                padding: 8px;
                padding-top: 12px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #88aaff;
            }
            QGroupBox:disabled {
                color: #555;
                border-color: #2a2a2a;
            }
            QGroupBox:disabled::title {
                color: #555;
            }
        """)
        self.ml_labels_group.setEnabled(False)

        ml_labels_layout = QGridLayout(self.ml_labels_group)
        ml_labels_layout.setContentsMargins(8, 4, 8, 4)
        ml_labels_layout.setSpacing(6)

        # State dropdown
        state_label = QLabel("State:")
        state_label.setStyleSheet("font-weight: normal; font-size: 9pt;")
        self.cb_ml_state = QComboBox()
        self.cb_ml_state.addItems(["Awake", "Anesthetized", "Other"])
        self.cb_ml_state.setStyleSheet("font-size: 9pt;")
        self.cb_ml_state.setToolTip("Animal state during recording")

        # State "Other" text field (hidden by default)
        self.le_ml_state_other = QLineEdit()
        self.le_ml_state_other.setPlaceholderText("Specify...")
        self.le_ml_state_other.setStyleSheet("font-size: 9pt;")
        self.le_ml_state_other.setVisible(False)
        self.le_ml_state_other.setFixedWidth(80)
        self.cb_ml_state.currentTextChanged.connect(
            lambda t: self.le_ml_state_other.setVisible(t == "Other")
        )

        # Gas dropdown
        gas_label = QLabel("Gas:")
        gas_label.setStyleSheet("font-weight: normal; font-size: 9pt;")
        self.cb_ml_gas = QComboBox()
        self.cb_ml_gas.addItems(["Room Air", "O₂", "Hypoxia", "CO₂", "Hypercapnia", "Other"])
        self.cb_ml_gas.setStyleSheet("font-size: 9pt;")
        self.cb_ml_gas.setToolTip("Gas/breathing condition during recording")

        # Gas "Other" text field (hidden by default)
        self.le_ml_gas_other = QLineEdit()
        self.le_ml_gas_other.setPlaceholderText("Specify...")
        self.le_ml_gas_other.setStyleSheet("font-size: 9pt;")
        self.le_ml_gas_other.setVisible(False)
        self.le_ml_gas_other.setFixedWidth(80)
        self.cb_ml_gas.currentTextChanged.connect(
            lambda t: self.le_ml_gas_other.setVisible(t == "Other")
        )

        # Quality spinner
        quality_label = QLabel("Quality:")
        quality_label.setStyleSheet("font-weight: normal; font-size: 9pt;")
        self.spin_ml_quality = QSpinBox()
        self.spin_ml_quality.setRange(1, 10)
        self.spin_ml_quality.setValue(7)
        self.spin_ml_quality.setStyleSheet("font-size: 9pt;")
        self.spin_ml_quality.setToolTip("Data quality rating (1=poor, 10=excellent)")
        self.spin_ml_quality.setFixedWidth(50)
        quality_suffix = QLabel("/ 10")
        quality_suffix.setStyleSheet("font-weight: normal; font-size: 9pt; color: #888;")

        # Layout: State and Gas on first row, Quality on second
        ml_labels_layout.addWidget(state_label, 0, 0)
        ml_labels_layout.addWidget(self.cb_ml_state, 0, 1)
        ml_labels_layout.addWidget(self.le_ml_state_other, 0, 2)
        ml_labels_layout.addWidget(gas_label, 0, 3)
        ml_labels_layout.addWidget(self.cb_ml_gas, 0, 4)
        ml_labels_layout.addWidget(self.le_ml_gas_other, 0, 5)
        ml_labels_layout.addWidget(quality_label, 1, 0)
        quality_row = QHBoxLayout()
        quality_row.setSpacing(2)
        quality_row.addWidget(self.spin_ml_quality)
        quality_row.addWidget(quality_suffix)
        quality_row.addStretch()
        ml_labels_layout.addLayout(quality_row, 1, 1, 1, 2)

        # Connect ML checkbox to enable/disable both waveforms and labels
        self.chk_save_ml_training.toggled.connect(self._on_ml_training_toggled)

        # Load saved states
        self._load_export_checkbox_states()

        # Layout: 3 columns
        export_grid.addWidget(self.chk_save_npz, 0, 0)
        export_grid.addWidget(self.chk_save_timeseries, 0, 1)
        export_grid.addWidget(self.chk_save_breaths, 0, 2)
        export_grid.addWidget(self.chk_save_events, 1, 0)
        export_grid.addWidget(self.chk_save_pdf, 1, 1)
        export_grid.addWidget(self.chk_save_session, 1, 2)

        # ML row
        ml_row = QWidget()
        ml_layout = QHBoxLayout(ml_row)
        ml_layout.setContentsMargins(0, 0, 0, 0)
        ml_layout.setSpacing(4)
        ml_layout.addWidget(self.chk_save_ml_training)
        ml_layout.addWidget(self.btn_ml_folder_settings)
        ml_layout.addStretch()
        export_grid.addWidget(ml_row, 2, 0, 1, 3)
        export_grid.addWidget(self.chk_ml_include_waveforms, 3, 0, 1, 3)
        export_grid.addWidget(self.ml_labels_group, 4, 0, 1, 3)

        layout.addWidget(export_widget)

    def _create_preview_panel(self, layout: QVBoxLayout):
        """Create the right panel preview content."""
        # Header with title and hide button
        header = QHBoxLayout()

        # Title with match info
        if self._is_fuzzy_match:
            title_text = f'<span style="color: #FFA500;">Notes Preview (fuzzy match)</span>'
        else:
            title_text = '<span style="color: #88aaff;">Notes Preview</span>'
        title_label = QLabel(title_text)
        title_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        header.addWidget(title_label)

        header.addStretch()

        # Hide button
        btn_hide = QPushButton("◀ Hide")
        btn_hide.setFixedWidth(70)
        btn_hide.clicked.connect(self._toggle_preview)
        header.addWidget(btn_hide)

        layout.addLayout(header)

        # Search box row
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 4, 0, 4)

        search_label = QLabel("Search:")
        search_label.setStyleSheet("color: #888; font-size: 9pt;")
        search_layout.addWidget(search_label)

        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Type to highlight keywords... (Ctrl+F)")
        self._search_box.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 9pt;
            }
            QLineEdit:focus {
                border-color: #2a7fff;
            }
        """)
        self._search_box.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self._search_box, 1)

        # Clear search button
        btn_clear_search = QPushButton("✕")
        btn_clear_search.setFixedSize(24, 24)
        btn_clear_search.setToolTip("Clear search")
        btn_clear_search.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #888;
                font-size: 12px;
            }
            QPushButton:hover { color: #e0e0e0; }
        """)
        btn_clear_search.clicked.connect(lambda: self._search_box.clear())
        search_layout.addWidget(btn_clear_search)

        layout.addLayout(search_layout)

        # Ctrl+F shortcut to focus search
        shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        shortcut.activated.connect(self._focus_search)

        # Preview content container (lazy loaded) - will contain tabs if multiple files
        self._preview_container = QWidget()
        self._preview_container_layout = QVBoxLayout(self._preview_container)
        self._preview_container_layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder
        placeholder = QLabel('<span style="color: #888;">Loading preview...</span>')
        placeholder.setAlignment(QtCore_Qt.AlignmentFlag.AlignCenter)
        self._preview_container_layout.addWidget(placeholder)

        layout.addWidget(self._preview_container, 1)  # Stretch to fill

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_external = QPushButton("Open in External App")
        btn_external.clicked.connect(self._open_in_external_app)
        btn_layout.addWidget(btn_external)

        layout.addLayout(btn_layout)

        # Store reference to file tabs (created during load)
        self._file_tabs = None

    def _toggle_preview(self):
        """Toggle the preview panel visibility."""
        self._preview_visible = not self._preview_visible

        if self._preview_visible:
            # Show preview panel
            self._preview_panel.setVisible(True)
            self._preview_panel.setFixedWidth(self.PREVIEW_PANEL_WIDTH)
            self.setFixedWidth(self.FORM_PANEL_WIDTH + self.PREVIEW_PANEL_WIDTH)

            # Update toggle button text
            if hasattr(self, 'btn_toggle_preview'):
                self.btn_toggle_preview.setText(f"Notes ({len(self._linked_notes)}) ◀")

            # Load preview content if not already loaded
            if not self._preview_loaded:
                self._load_preview_content()
        else:
            # Hide preview panel
            self._preview_panel.setVisible(False)
            self.setFixedWidth(self.FORM_PANEL_WIDTH)

            # Update toggle button text
            if hasattr(self, 'btn_toggle_preview'):
                self.btn_toggle_preview.setText(f"Notes ({len(self._linked_notes)}) ▶")

        # Save preference
        settings = QSettings("PhysioMetrics", "PhysioMetrics")
        settings.setValue("save_dialog/show_preview", self._preview_visible)

    def _load_preview_content(self):
        """Load the preview content (lazy loading).

        Creates tabs for multiple notes files, each containing the full preview widget.
        """
        if self._preview_loaded or not self._linked_notes:
            return

        # Clear placeholder
        while self._preview_container_layout.count():
            item = self._preview_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Check if we can create previews
        if not self._main_window or not hasattr(self._main_window, '_create_note_preview_widget'):
            fallback = QLabel('<span style="color: #888;">Preview not available</span>')
            fallback.setAlignment(QtCore_Qt.AlignmentFlag.AlignCenter)
            self._preview_container_layout.addWidget(fallback)
            self._preview_loaded = True
            return

        # Single file - no tabs needed
        if len(self._linked_notes) == 1:
            preview_widget = self._main_window._create_note_preview_widget(
                self._linked_notes[0], self._highlight_stem
            )
            self._preview_container_layout.addWidget(preview_widget)
        else:
            # Multiple files - create tabs
            self._file_tabs = QTabWidget()
            self._file_tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: 1px solid #3e3e42;
                    background-color: #1e1e1e;
                }
                QTabBar::tab {
                    background-color: #2d2d30;
                    color: #d4d4d4;
                    padding: 6px 12px;
                    border: 1px solid #3e3e42;
                    border-bottom: none;
                    margin-right: 2px;
                    max-width: 200px;
                }
                QTabBar::tab:selected {
                    background-color: #1e1e1e;
                    border-bottom-color: #1e1e1e;
                }
                QTabBar::tab:hover:!selected {
                    background-color: #3e3e42;
                }
            """)

            # Add a tab for each notes file
            for i, note_info in enumerate(self._linked_notes):
                note_path = Path(note_info.get('path', ''))
                note_name = note_path.name if note_path else f"File {i+1}"

                # Truncate long names for tab
                tab_name = note_name[:25] + "..." if len(note_name) > 28 else note_name

                # Create preview widget for this file
                preview_widget = self._main_window._create_note_preview_widget(
                    note_info, self._highlight_stem
                )

                self._file_tabs.addTab(preview_widget, tab_name)
                self._file_tabs.setTabToolTip(i, str(note_path))

            self._preview_container_layout.addWidget(self._file_tabs)

        self._preview_loaded = True

    def _focus_search(self):
        """Focus the search box (called by Ctrl+F shortcut)."""
        if hasattr(self, '_search_box') and self._preview_visible:
            self._search_box.setFocus()
            self._search_box.selectAll()

    def _on_search_changed(self, text: str):
        """Handle search text changes - debounced to avoid lag on every keystroke."""
        if not self._preview_loaded:
            return

        # Store search term for highlighting
        self._search_term = text.strip()

        # Restart the debounce timer (300ms delay before reloading)
        # This prevents reloading on every keystroke
        self._search_debounce_timer.stop()
        self._search_debounce_timer.start(300)

    def _perform_search_reload(self):
        """Actually perform the search reload (called after debounce delay)."""
        # Reload the preview with both ABF stem and search term
        self._reload_preview_with_search()

    def _reload_preview_with_search(self):
        """Reload preview content with dual highlighting.

        ABF filename is always highlighted in green (primary).
        Search term is highlighted in orange/yellow (secondary).
        """
        if not self._main_window or not hasattr(self._main_window, '_create_note_preview_widget'):
            return

        # Get current search term
        search_term = getattr(self, '_search_term', '') or None

        # Clear current preview
        while self._preview_container_layout.count():
            item = self._preview_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Reload with both ABF stem (primary) and search term (secondary)
        if len(self._linked_notes) == 1:
            preview_widget = self._main_window._create_note_preview_widget(
                self._linked_notes[0], self._highlight_stem, search_term
            )
            self._preview_container_layout.addWidget(preview_widget)
        else:
            self._file_tabs = QTabWidget()
            self._file_tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: 1px solid #3e3e42;
                    background-color: #1e1e1e;
                }
                QTabBar::tab {
                    background-color: #2d2d30;
                    color: #d4d4d4;
                    padding: 6px 12px;
                    border: 1px solid #3e3e42;
                    border-bottom: none;
                    margin-right: 2px;
                    max-width: 200px;
                }
                QTabBar::tab:selected {
                    background-color: #1e1e1e;
                }
                QTabBar::tab:hover:!selected {
                    background-color: #3e3e42;
                }
            """)

            for i, note_info in enumerate(self._linked_notes):
                note_path = Path(note_info.get('path', ''))
                note_name = note_path.name if note_path else f"File {i+1}"
                tab_name = note_name[:25] + "..." if len(note_name) > 28 else note_name

                preview_widget = self._main_window._create_note_preview_widget(
                    note_info, self._highlight_stem, search_term
                )

                self._file_tabs.addTab(preview_widget, tab_name)
                self._file_tabs.setTabToolTip(i, str(note_path))

            self._preview_container_layout.addWidget(self._file_tabs)

    def _open_in_external_app(self):
        """Open the currently selected notes file in the default external application."""
        import subprocess
        import os

        if not self._linked_notes:
            return

        # Get the currently selected file index
        file_index = 0
        if self._file_tabs is not None and len(self._linked_notes) > 1:
            file_index = self._file_tabs.currentIndex()

        file_path = self._linked_notes[file_index].get('path', '')
        if file_path and Path(file_path).exists():
            if sys.platform == 'win32':
                os.startfile(file_path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', file_path])
            else:
                subprocess.run(['xdg-open', file_path])

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
            QCheckBox::indicator:disabled:checked {
                background-color: #2a7fff;
                border-color: #2a7fff;
            }
            QCheckBox:disabled {
                color: #888888;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 16px;
                min-width: 60px;
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

    # --- Helpers ---
    def _norm_token(self, s: str) -> str:
        s0 = (s or "").strip()
        if not s0:
            return ""
        s1 = s0.replace(" ", "")
        s1 = re.sub(r"(?i)chr\s*2", "ChR2", s1)
        s1 = re.sub(r"(?i)gcamp\s*6f", "GCaMP6f", s1)
        s1 = re.sub(r"(?i)([A-Za-z0-9_-]*?)cre$", lambda m: (m.group(1) or "") + "Cre", s1)
        return s1

    def _san(self, s: str) -> str:
        s = (s or "").strip()
        s = s.replace(" ", "_")
        s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
        s = re.sub(r"_+", "_", s)
        s = re.sub(r"-+", "-", s)
        return s

    def _update_preview(self):
        strain = self._norm_token(self.le_strain.text())
        virus = self._norm_token(self.le_virus.text())
        location = self.le_location.text().strip()
        stim = self.le_stim.text().strip()
        power = self.le_power.text().strip()
        sex = self.cb_sex.currentText().strip()
        animal = self.le_animal.text().strip()
        abf = self._abf_name
        ch = self._channel

        parts = [p for p in (
            self._san(strain), self._san(virus), self._san(location),
            self._san(sex), self._san(animal), self._san(stim),
            self._san(power), self._san(abf), self._san(ch)
        ) if p]
        preview = "_".join(parts) if parts else "analysis"
        self.lbl_preview.setText(preview)

    def values(self) -> dict:
        exp_type_map = {
            "30Hz Stimulus (default)": "30hz_stim",
            "Hargreaves Thermal Sensitivity": "hargreaves",
            "Licking Behavior": "licking"
        }
        experiment_type = exp_type_map.get(self.cb_experiment_type.currentText(), "30hz_stim")

        # Get ML state value (handle "Other" option)
        ml_state = self.cb_ml_state.currentText()
        if ml_state == "Other":
            ml_state = self.le_ml_state_other.text().strip() or "other"
        ml_state = ml_state.lower().replace(" ", "_")

        # Get ML gas value (handle "Other" option)
        ml_gas = self.cb_ml_gas.currentText()
        if ml_gas == "Other":
            ml_gas = self.le_ml_gas_other.text().strip() or "other"
        # Normalize gas values
        gas_map = {"Room Air": "room_air", "O₂": "oxygen", "Hypoxia": "hypoxia",
                   "CO₂": "co2", "Hypercapnia": "hypercapnia"}
        ml_gas = gas_map.get(ml_gas, ml_gas.lower().replace(" ", "_"))

        return {
            "strain": self.le_strain.text().strip(),
            "virus": self.le_virus.text().strip(),
            "location": self.le_location.text().strip(),
            "stim": self.le_stim.text().strip(),
            "power": self.le_power.text().strip(),
            "sex": self.cb_sex.currentText().strip(),
            "animal": self.le_animal.text().strip(),
            "abf": self._abf_name,
            "chan": self._channel,
            "preview": self.lbl_preview.text().strip(),
            "choose_dir": bool(self.cb_choose_dir.isChecked()),
            "experiment_type": experiment_type,
            "save_npz": True,
            "save_timeseries_csv": bool(self.chk_save_timeseries.isChecked()),
            "save_breaths_csv": bool(self.chk_save_breaths.isChecked()),
            "save_events_csv": bool(self.chk_save_events.isChecked()),
            "save_pdf": bool(self.chk_save_pdf.isChecked()),
            "save_session": bool(self.chk_save_session.isChecked()),
            "save_ml_training": bool(self.chk_save_ml_training.isChecked()),
            "ml_include_waveforms": bool(self.chk_ml_include_waveforms.isChecked()),
            # ML metadata
            "ml_animal_state": ml_state,
            "ml_gas": ml_gas,
            "ml_quality_score": self.spin_ml_quality.value(),
        }

    def _on_ml_training_toggled(self, checked: bool):
        """Enable/disable ML-related options when ML Training checkbox is toggled."""
        self.chk_ml_include_waveforms.setEnabled(checked)
        self.ml_labels_group.setEnabled(checked)

    def _on_configure_ml_folder(self):
        """Open folder dialog to configure ML training data save location."""
        from PyQt6.QtWidgets import QMessageBox

        settings = QSettings("PhysioMetrics", "PhysioMetrics")
        current_folder = settings.value("ml_training_folder", "")
        start_dir = current_folder if current_folder else str(Path.home())

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select ML Training Data Folder",
            start_dir,
            QFileDialog.Option.ShowDirsOnly
        )

        if folder:
            settings.setValue("ml_training_folder", folder)
            settings.sync()
            QMessageBox.information(
                self,
                "ML Training Folder Updated",
                f"ML training data will be saved to:\n\n{folder}"
            )

    def _load_export_checkbox_states(self):
        """Load export checkbox states from QSettings."""
        settings = QSettings("PhysioMetrics", "PhysioMetrics")
        self.chk_save_timeseries.setChecked(settings.value("export/save_timeseries", True, type=bool))
        self.chk_save_breaths.setChecked(settings.value("export/save_breaths", True, type=bool))
        self.chk_save_events.setChecked(settings.value("export/save_events", True, type=bool))
        self.chk_save_pdf.setChecked(settings.value("export/save_pdf", True, type=bool))
        self.chk_save_session.setChecked(settings.value("export/save_session", True, type=bool))
        self.chk_save_ml_training.setChecked(settings.value("export/save_ml_training", False, type=bool))
        self.chk_ml_include_waveforms.setChecked(settings.value("export/ml_include_waveforms", False, type=bool))

    def _save_export_checkbox_states(self):
        """Save export checkbox states to QSettings."""
        settings = QSettings("PhysioMetrics", "PhysioMetrics")
        settings.setValue("export/save_timeseries", self.chk_save_timeseries.isChecked())
        settings.setValue("export/save_breaths", self.chk_save_breaths.isChecked())
        settings.setValue("export/save_events", self.chk_save_events.isChecked())
        settings.setValue("export/save_pdf", self.chk_save_pdf.isChecked())
        settings.setValue("export/save_session", self.chk_save_session.isChecked())
        settings.setValue("export/save_ml_training", self.chk_save_ml_training.isChecked())
        settings.setValue("export/ml_include_waveforms", self.chk_ml_include_waveforms.isChecked())
        settings.sync()

    def accept(self):
        """Override accept to save checkbox states and emit signal before closing."""
        self._save_export_checkbox_states()
        self.accepted_with_values.emit(self.values())
        super().accept()
