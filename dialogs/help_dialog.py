"""
PhysioMetrics Help Dialog

Quick reference guide with workflow and exported data documentation.
"""

import sys
import os

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextBrowser, QLabel, QDialogButtonBox, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap


class HelpDialog(QDialog):
    """Quick reference help dialog with essential information."""

    def __init__(self, parent=None, update_info=None):
        super().__init__(parent)
        self.setWindowTitle("PhysioMetrics - Quick Reference")
        self.resize(850, 700)
        self.update_info = update_info  # Pre-checked update info from main window
        self.parent_window = parent

        # Load settings (position, size)
        self.settings = None
        if parent and hasattr(parent, 'settings'):
            self.settings = parent.settings
            # Restore window position (with validation to ensure on-screen)
            if self.settings.contains("help_dialog_pos"):
                from PyQt6.QtWidgets import QApplication
                pos = self.settings.value("help_dialog_pos")
                # Check if position is within any available screen
                is_visible = False
                for screen in QApplication.screens():
                    if screen.availableGeometry().contains(pos):
                        is_visible = True
                        break
                if is_visible:
                    self.move(pos)
                else:
                    # Center on primary screen if saved position is off-screen
                    screen_geometry = QApplication.primaryScreen().availableGeometry()
                    center = screen_geometry.center()
                    self.move(center.x() - self.width() // 2, center.y() - self.height() // 2)
            # Restore window size
            if self.settings.contains("help_dialog_size"):
                size = self.settings.value("help_dialog_size")
                self.resize(size)

        self._setup_ui()
        self._apply_dark_theme()
        self._enable_dark_title_bar()

    def closeEvent(self, event):
        """Save position and size when closing."""
        if self.settings:
            self.settings.setValue("help_dialog_pos", self.pos())
            self.settings.setValue("help_dialog_size", self.size())
        super().closeEvent(event)

    def _setup_ui(self):
        """Create the help dialog UI with two tabs."""
        layout = QVBoxLayout(self)

        # Header with app info
        header = QLabel("<h1>PhysioMetrics - Quick Reference Guide</h1>")
        header.setStyleSheet("color: #2a7fff; padding: 10px;")
        layout.addWidget(header)

        # Tab widget for workflow, data files, and about
        tabs = QTabWidget()
        tabs.addTab(self._create_quick_reference(), "Features & Usage")
        tabs.addTab(self._create_data_files_tab(), "Exported Data Files")
        tabs.addTab(self._create_about_tab(), "About")
        layout.addWidget(tabs)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)
        layout.addWidget(button_box)

    def _create_quick_reference(self):
        """Create single-page quick reference guide."""
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard |
            Qt.TextInteractionFlag.LinksAccessibleByMouse |
            Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        browser.setHtml("""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.5; padding: 10px;">

            <h2 style="color: #2a7fff;">Quick Start</h2>
            <ol style="margin-top: 5px;">
                <li><b>Open File:</b> Ctrl+O → Select .abf, .smrx, or .edf file <i>(already analyzed? you'll be prompted to resume)</i></li>
                <li><b>Select Channel:</b> Choose respiratory signal from dropdown</li>
                <li><b>Detect Peaks:</b> Click "Apply" (threshold auto-detected)</li>
                <li><b>Review/Edit:</b> Click "Toggle Breath" button, then left-click to toggle peaks on/off, right-click to toggle sigh labels.</li>
                <li><b>Preview:</b> Click "View Summary" to review results before export</li>
                <li><b>Export:</b> Click "Save Data" (Ctrl+S) → Enter animal info → Choose which files to save → Save</li>
            </ol>

            <h2 style="color: #2a7fff; margin-top: 20px;">Keyboard Shortcuts</h2>
            <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; margin-top: 5px;">
                <tr style="background-color: #2a2a2a;">
                    <th width="120">Mode/Button</th>
                    <th width="160">Shortcut</th>
                    <th>Action</th>
                </tr>
                <tr><td>General</td><td><b>Ctrl+O</b></td><td>Open file</td></tr>
                <tr><td>General</td><td><b>Ctrl+S</b></td><td>Save/export data</td></tr>
                <tr><td>General</td><td><b>F1</b></td><td>Show this help</td></tr>
                <tr><td><span style="color: #4CAF50;">Toggle Breath</span></td><td><b>Left-Click</b></td><td>Toggle peak on/off</td></tr>
                <tr><td><span style="color: #4CAF50;">Toggle Breath</span></td><td><b>Right-Click</b></td><td>Toggle sigh label on/off</td></tr>
                <tr><td><span style="color: #FFD700;">Move Point</span></td><td><b>Shift+Click & Drag</b></td><td>Snap to zero crossings</td></tr>
                <tr><td><span style="color: #9C27B0;">Mark Sniff</span></td><td><b>Shift+Click</b></td><td>Delete sniffing region</td></tr>
                <tr><td>Omit</td><td><b>Ctrl+Click</b></td><td>Delete omitted region</td></tr>
                <tr><td>Omit</td><td><b>Ctrl+Shift+Click</b></td><td>Toggle full sweep omission</td></tr>
                <tr><td>Omit</td><td><b>R</b></td><td>Snap regions to breath onsets</td></tr>
            </table>

            <h2 style="color: #2a7fff; margin-top: 20px;">Key Features</h2>
            <ul style="margin-top: 5px;">
                <li><b>Auto-Threshold:</b> Otsu's method finds optimal peak detection threshold automatically</li>
                <li><b>Manual Editing Modes:</b>
                    <ul style="margin-top: 3px; margin-bottom: 3px;">
                        <li><b><span style="color: #4CAF50;">Toggle Breath:</span></b> Left-click to toggle peaks on/off, right-click to toggle sigh labels</li>
                        <li><b><span style="color: #FFD700;">Move Point:</span></b> Click and drag peaks, onsets, or offsets. Hold Shift to snap to zero crossings</li>
                        <li><b><span style="color: #9C27B0;">Mark Sniff:</span></b> Click and drag to highlight sniffing regions. Shift+Click to delete a region</li>
                        <li><b>Omit:</b> Click and drag to mark artifacts/noise for exclusion from analysis
                            <ul style="margin-top: 3px; margin-bottom: 3px;">
                                <li>Omitted regions shown with gray overlay and "omitted" label</li>
                                <li>Ctrl+Click to delete an omitted region</li>
                                <li>Ctrl+Shift+Click to toggle full sweep omission</li>
                                <li>Press 'R' to snap regions to breath onset boundaries</li>
                                <li>Excluded from y-range calculation, metrics, and exports</li>
                            </ul>
                        </li>
                    </ul>
                </li>
                <li><b>Eupnea/Sniffing Detection (GMM Clustering):</b>
                    <ul style="margin-top: 3px; margin-bottom: 3px;">
                        <li><span style="color: green;">Green = Eupnea</span> (normal breathing), <span style="color: purple;">Purple = Sniffing</span></li>
                        <li>Click "GMM Clustering" to adjust cluster assignments</li>
                        <li>Fallback method available based on frequency alone (see Options tab)</li>
                    </ul>
                </li>
                <li><b>Apnea Detection:</b> Identifies breathing pauses (red line across bottom) based on interbreath interval threshold</li>
                <li><b>Outlier Marking:</b> <span style="color: #FFD700;">Yellow = Outlier breaths</span>. Adjust criteria in Options tab (amplitude, frequency, duration)</li>
                <li><b>Filters:</b> High-pass (remove drift), Low-pass (remove noise), Notch (60 Hz line noise)</li>
                <li><b>More Options:</b> Click to see histogram and adjust threshold interactively</li>
                <li><b>Red Dashed Line:</b> Shows current height threshold on plot</li>
                <li><b>Events Channel:</b> (<span style="color: #FFD700;">⚠ developing feature</span>) Dropdown for triggering on custom events</li>
                <li><b>Pulse Analysis:</b> (<span style="color: #FFD700;">⚠ in development</span>) 25ms stimulus analysis with phase response curves (PRC). Results appear in summary PDF but not yet saved to CSV files.</li>
                <li><b>Curation Tab:</b> Consolidate multiple files (best for identical experiments like 30Hz stims)</li>
                <li><b>Auto-Resume:</b> When opening a previously analyzed file/channel, you'll be prompted to resume or start over</li>
            </ul>

            <h2 style="color: #2a7fff; margin-top: 20px;">Common Workflows</h2>

            <h3 style="margin-bottom: 5px;">Basic Analysis</h3>
            <p style="margin-top: 0;">Load file → Select channel → Click Apply → Review → Export</p>

            <h3 style="margin-bottom: 5px;">Adjust Threshold</h3>
            <p style="margin-top: 0;">Three ways to adjust: (1) Drag the red dashed line directly on the main plot, (2) Use up/down arrows in the Prom/Thresh spinbox, or (3) Click "More Options" → Drag red line on histogram → OK. Then click Apply to re-detect peaks.</p>

            <h3 style="margin-bottom: 5px;">Save Session & Resume Later</h3>
            <p style="margin-top: 0;">Press Ctrl+S → Check "Session State" → Save → Resume later with Ctrl+O (auto-prompts if file was analyzed)</p>

            <h2 style="color: #2a7fff; margin-top: 20px;">Troubleshooting</h2>
            <ul style="margin-top: 5px;">
                <li><b>No peaks detected?</b> Try lowering threshold in "More Options" or check signal inversion</li>
                <li><b>Too many peaks?</b> Increase threshold or Min Peak Distance in "More Options"</li>
                <li><b>Apply button grayed?</b> Button re-enables when channel or filters change</li>
                <li><b>Respiratory signal upside down?</b> Enable "Invert Signal" checkbox</li>
                <li><b>Baseline drift?</b> Enable High-Pass filter (0.5 Hz recommended)</li>
                <li><b>Eupnea/sniffing wrong?</b> Click "GMM Clustering" or "Outlier Threshold" to adjust detection</li>
            </ul>

            <h2 style="color: #2a7fff; margin-top: 20px;">File Formats</h2>
            <ul style="margin-top: 5px;">
                <li><b>.abf</b> - Axon Binary Format (pCLAMP)</li>
                <li><b>.smrx</b> - Spike2 64-bit format (requires son64.dll)</li>
                <li><b>.edf</b> - European Data Format</li>
                <li><b>.pleth.npz</b> - PhysioMetrics session files (save/load complete analysis)</li>
            </ul>

        </body>
        </html>
        """)
        return browser

    def _create_data_files_tab(self):
        """Create tab documenting exported data file formats."""
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard |
            Qt.TextInteractionFlag.LinksAccessibleByMouse |
            Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        browser.setHtml("""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.5; padding: 10px;">

            <h2 style="color: #2a7fff;">Exported Data Files</h2>
            <p>When you press <b>Ctrl+S</b> to save/export, a dialog prompts for:</p>
            <ul style="margin-top: 5px;">
                <li><b>Animal Information:</b> Mouse strain, virus, location, stimulation parameters</li>
                <li><b>File Selection:</b> Choose which file types to export (checkboxes)</li>
                <li><b>Save Location:</b> Defaults to <code>Pleth_App_Analysis</code> folder in data file directory</li>
            </ul>
            <p style="margin-top: 10px;">Files are named: <code>[metadata]_[filename]_[channel]_[suffix]</code></p>

            <h3 style="color: #2a7fff; margin-top: 20px;">1. NPZ Bundle (always saved)</h3>
            <p><b>Filename:</b> <code>[name]_bundle.npz</code> <i>(~0.5s to save)</i></p>
            <p><b>Contains:</b> Compressed binary data bundle with downsampled processed trace, metric traces, peaks/breaths/sighs per sweep, stim spans, and metadata. Fast to save and load but not human-readable (binary format - use Python/NumPy to read).</p>

            <h3 style="color: #2a7fff; margin-top: 20px;">2. Session State (always saved)</h3>
            <p><b>Filename:</b> <code>[name]_session.npz</code> <i>(~8 MB, <1s to save)</i></p>
            <p><b>Contains:</b> Complete analysis session state including detected peaks, manual labels, filter settings, GMM clustering results, and all parameters. Enables resuming work later - when you open a file that was previously analyzed, PhysioMetrics prompts: "Resume analysis or start over?"</p>

            <h3 style="color: #2a7fff; margin-top: 20px;">3. Breaths CSV (optional)</h3>
            <p><b>Filename:</b> <code>[name]_breaths.csv</code> <i>(~1-2s to save)</i></p>
            <p><b>Contains:</b> One row per detected breath in wide layout format with columns grouped by region (ALL, BASELINE, STIM, POST) and normalization method (raw, per-sweep normalized, eupnea-normalized).</p>

            <table border="1" cellpadding="3" cellspacing="0" style="border-collapse: collapse; margin-top: 5px; font-size: 9px;">
                <tr style="background-color: #2a2a2a;">
                    <th>sweep</th>
                    <th>breath</th>
                    <th>t</th>
                    <th>region</th>
                    <th>is_sigh</th>
                    <th>is_sniffing</th>
                    <th>is_eupnea</th>
                    <th>is_apnea</th>
                    <th>if</th>
                    <th>amp_insp</th>
                    <th>ti</th>
                    <th>te</th>
                    <th>vent_proxy</th>
                    <th>...</th>
                </tr>
                <tr>
                    <td>1</td>
                    <td>1</td>
                    <td>-26.09</td>
                    <td>all</td>
                    <td>0</td>
                    <td>1</td>
                    <td>0</td>
                    <td>0</td>
                    <td>7.27</td>
                    <td>3.21</td>
                    <td>0.053</td>
                    <td>0.085</td>
                    <td>0.760</td>
                    <td>...</td>
                </tr>
                <tr>
                    <td>1</td>
                    <td>2</td>
                    <td>-25.97</td>
                    <td>all</td>
                    <td>0</td>
                    <td>1</td>
                    <td>0</td>
                    <td>0</td>
                    <td>10.47</td>
                    <td>2.42</td>
                    <td>0.047</td>
                    <td>0.049</td>
                    <td>0.734</td>
                    <td>...</td>
                </tr>
                <tr>
                    <td>1</td>
                    <td>3</td>
                    <td>-25.87</td>
                    <td>all</td>
                    <td>0</td>
                    <td>1</td>
                    <td>0</td>
                    <td>0</td>
                    <td>9.25</td>
                    <td>3.24</td>
                    <td>0.050</td>
                    <td>0.058</td>
                    <td>0.925</td>
                    <td>...</td>
                </tr>
                <tr>
                    <td>1</td>
                    <td>162</td>
                    <td>0.017</td>
                    <td>stim</td>
                    <td>0</td>
                    <td>0</td>
                    <td>1</td>
                    <td>0</td>
                    <td>3.94</td>
                    <td>2.64</td>
                    <td>0.081</td>
                    <td>0.087</td>
                    <td>0.498</td>
                    <td>...</td>
                </tr>
            </table>

            <p style="margin-top: 8px;"><b>Key Columns:</b> sweep, breath, t (time), region, is_sigh, is_sniffing, is_eupnea, is_apnea, if (instantaneous frequency), amp_insp, amp_exp, area_insp, area_exp, ti (inspiratory time), te (expiratory time), vent_proxy. Plus normalized versions for baseline/stim/post regions.</p>

            <h3 style="color: #2a7fff; margin-top: 20px;">4. Events CSV (optional)</h3>
            <p><b>Filename:</b> <code>[name]_events.csv</code> <i>(~0.5s to save)</i></p>
            <p><b>Contains:</b> Time intervals for stimulus events, apnea episodes, eupnea regions, and sniffing bouts.</p>

            <table border="1" cellpadding="4" cellspacing="0" style="border-collapse: collapse; margin-top: 5px; font-size: 10px;">
                <tr style="background-color: #2a2a2a;">
                    <th>sweep</th>
                    <th>event_type</th>
                    <th>start_time</th>
                    <th>end_time</th>
                    <th>duration</th>
                </tr>
                <tr>
                    <td>1</td>
                    <td>stimulus</td>
                    <td>0.000</td>
                    <td>0.010</td>
                    <td>0.010</td>
                </tr>
                <tr>
                    <td>1</td>
                    <td>apnea</td>
                    <td>3.811</td>
                    <td>4.428</td>
                    <td>0.617</td>
                </tr>
                <tr>
                    <td>1</td>
                    <td>eupnea</td>
                    <td>-0.110</td>
                    <td>14.918</td>
                    <td>15.028</td>
                </tr>
                <tr>
                    <td>1</td>
                    <td>sniffing</td>
                    <td>15.134</td>
                    <td>15.691</td>
                    <td>0.558</td>
                </tr>
            </table>

            <h3 style="color: #2a7fff; margin-top: 20px;">5. Timeseries CSV (optional)</h3>
            <p><b>Filename:</b> <code>[name]_timeseries.csv</code> <i>(~9s to save)</i></p>
            <p><b>Contains:</b> Time-aligned metric traces (one row per time point). For each metric, includes per-sweep columns (if_s1, if_s2, ..., if_s10), then mean and SEM. Three blocks: raw values, normalized (_norm), and eupnea-normalized (_norm_eupnea).</p>

            <table border="1" cellpadding="3" cellspacing="0" style="border-collapse: collapse; margin-top: 5px; font-size: 9px;">
                <tr style="background-color: #2a2a2a;">
                    <th>t</th>
                    <th>if_s1</th>
                    <th>if_s2</th>
                    <th>...</th>
                    <th>if_s10</th>
                    <th>if_mean</th>
                    <th>if_sem</th>
                    <th>amp_insp_s1</th>
                    <th>...</th>
                    <th>ti_mean</th>
                    <th>...</th>
                    <th>if_norm_s1</th>
                    <th>...</th>
                </tr>
                <tr>
                    <td>-26.25</td>
                    <td></td>
                    <td></td>
                    <td>...</td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td>...</td>
                    <td></td>
                    <td>...</td>
                    <td></td>
                    <td>...</td>
                </tr>
                <tr>
                    <td>-26.09</td>
                    <td>7.27</td>
                    <td>10.47</td>
                    <td>...</td>
                    <td>6.79</td>
                    <td>7.45</td>
                    <td>0.52</td>
                    <td>3.21</td>
                    <td>...</td>
                    <td>0.051</td>
                    <td>...</td>
                    <td>1.19</td>
                    <td>...</td>
                </tr>
            </table>

            <p style="margin-top: 8px;"><b>Structure:</b> t column, then for each metric (if, amp_insp, amp_exp, area_insp, area_exp, ti, te, vent_proxy, max_dinsp, max_dexp, regularity, sniff_conf, eupnea_conf): sweep1...sweep10, mean, sem. This pattern repeats for _norm and _norm_eupnea blocks.</p>

            <h3 style="color: #2a7fff; margin-top: 20px;">6. Summary PDF (optional)</h3>
            <p><b>Filename:</b> <code>[name]_summary.pdf</code> <i>(~31s to generate - can skip for quick exports)</i></p>
            <p><b>Contains:</b> Visualization plots including trace overview, breathing metrics over time, and statistical summaries. This is the same content shown when you click "View Summary" before exporting.</p>

            <h2 style="color: #2a7fff; margin-top: 25px;">Export Tips</h2>
            <ul style="margin-top: 5px;">
                <li><b>Quick Export:</b> Uncheck "Summary PDF" to save ~30 seconds</li>
                <li><b>Session Files:</b> Enable "Session State" to resume analysis later (auto-prompts on file open)</li>
                <li><b>File Location:</b> Check "Let me choose where to save" to select custom save location</li>
                <li><b>All files saved to:</b> <code>Pleth_App_Analysis</code> folder (created automatically)</li>
            </ul>

        </body>
        </html>
        """)
        return browser

    def _create_about_tab(self):
        """Create About tab with splash screen and version info."""
        from version_info import VERSION_STRING

        # Create widget to hold content
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Create three-column layout
        from PyQt6.QtWidgets import QHBoxLayout
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(15)

        # LEFT COLUMN
        left_column = QVBoxLayout()
        left_column.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        # Add splash screen image
        image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'images', 'plethapp_splash_dark-01.png')
        if os.path.exists(image_path):
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            # Scale image to smaller size for three-column layout
            scaled_pixmap = pixmap.scaled(130, 130, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            left_column.addWidget(image_label)

        # Add version info
        version_label = QLabel(f"<h2 style='color: #2a7fff;'>PhysioMetrics</h2><p style='font-size: 11pt;'>Version {VERSION_STRING}</p>")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_column.addWidget(version_label)

        # Check for updates button
        from PyQt6.QtWidgets import QPushButton
        check_update_btn = QPushButton("Check for Updates")
        check_update_btn.setMaximumWidth(160)
        check_update_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #4a4a4a;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)
        check_update_btn.clicked.connect(self._on_check_updates_clicked)
        left_column.addWidget(check_update_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        left_column.addSpacing(8)

        # Update status label
        self.update_label = QLabel()
        self.update_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_label.setOpenExternalLinks(True)
        self.update_label.setTextFormat(Qt.TextFormat.RichText)
        self.update_label.setWordWrap(True)
        self.update_label.setMaximumWidth(220)
        left_column.addWidget(self.update_label)

        # Perform initial update check asynchronously to avoid blocking UI
        self._check_for_updates_async()

        left_column.addStretch()

        # MIDDLE COLUMN
        middle_column = QVBoxLayout()
        middle_column.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Add author and citation information
        info_html = """
            <div style='text-align: left; padding: 5px;'>
                <p style='font-size: 9pt; margin-bottom: 8px;'>
                    <b>Advanced respiratory signal analysis tool for breath pattern detection<br>
                    and eupnea/apnea identification.</b>
                </p>

                <p style='font-size: 9pt; margin-top: 8px; margin-bottom: 3px;'>
                    <b style='color: #2a7fff;'>Developer</b><br>
                    Ryan Sean Phillips<br>
                    <span style='font-size: 8pt;'>
                    Seattle Children's Research Institute, Norcliffe Foundation Center<br>
                    <a href='mailto:ryan.phillips@seattlechildrens.org' style='color: #2a7fff;'>ryan.phillips@seattlechildrens.org</a> |
                    ORCID: <a href='https://orcid.org/0000-0002-8570-2348' style='color: #2a7fff;'>0000-0002-8570-2348</a>
                    </span>
                </p>

                <p style='font-size: 9pt; margin-top: 10px; margin-bottom: 3px;'>
                    <b style='color: #2a7fff;'>Funding</b><br>
                    <span style='font-size: 8pt;'>
                    NIDA K01 Award K01DA058543
                    </span>
                </p>

                <p style='font-size: 9pt; margin-top: 10px; margin-bottom: 3px;'>
                    <b style='color: #2a7fff;'>Citation</b><br>
                    <span style='font-size: 8pt; font-family: monospace;'>
                    Phillips, R.S. (2025). PhysioMetrics v""" + VERSION_STRING + """.<br>
                    DOI: <a href='https://doi.org/10.5281/zenodo.17575911' style='color: #2a7fff;'>10.5281/zenodo.17575911</a>
                    </span>
                </p>

                <p style='font-size: 8pt; margin-top: 10px;'>
                    <a href='https://github.com/RyanSeanPhillips/PhysioMetrics' style='color: #2a7fff;'>
                    GitHub</a> |
                    <a href='https://github.com/RyanSeanPhillips/PhysioMetrics/releases' style='color: #2a7fff;'>
                    Releases</a> |
                    <a href='https://github.com/RyanSeanPhillips/PhysioMetrics/issues' style='color: #2a7fff;'>
                    Issues</a>
                </p>
            </div>
        """

        info_label = QLabel(info_html)
        info_label.setWordWrap(True)
        info_label.setOpenExternalLinks(True)
        info_label.setTextFormat(Qt.TextFormat.RichText)
        info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        middle_column.addWidget(info_label)

        middle_column.addStretch()

        # RIGHT COLUMN
        right_column = QVBoxLayout()
        right_column.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Telemetry settings group
        from core import config as app_config
        from PyQt6.QtWidgets import QGroupBox, QCheckBox

        telemetry_group = QGroupBox("Privacy & Usage Statistics")
        telemetry_layout = QVBoxLayout()

        # Telemetry checkbox
        self.telemetry_checkbox = QCheckBox("Share anonymous usage statistics")
        self.telemetry_checkbox.setChecked(app_config.is_telemetry_enabled())
        self.telemetry_checkbox.setStyleSheet("font-size: 9pt;")
        self.telemetry_checkbox.toggled.connect(self._on_telemetry_toggled)
        telemetry_layout.addWidget(self.telemetry_checkbox)

        # Crash reports checkbox
        self.crash_reports_checkbox = QCheckBox("Send crash reports")
        self.crash_reports_checkbox.setChecked(app_config.is_crash_reports_enabled())
        self.crash_reports_checkbox.setStyleSheet("font-size: 9pt;")
        self.crash_reports_checkbox.toggled.connect(self._on_crash_reports_toggled)
        telemetry_layout.addWidget(self.crash_reports_checkbox)

        # Learn more link
        learn_more_label = QLabel('<a href="#details" style="color: #2a7fff; font-size: 8pt;">What data is collected?</a>')
        learn_more_label.setOpenExternalLinks(False)
        learn_more_label.linkActivated.connect(self._show_telemetry_details)
        telemetry_layout.addWidget(learn_more_label)

        telemetry_group.setLayout(telemetry_layout)
        telemetry_group.setMaximumWidth(280)
        right_column.addWidget(telemetry_group)

        right_column.addStretch()

        # Add columns to layout
        columns_layout.addLayout(left_column)
        columns_layout.addLayout(middle_column)
        columns_layout.addLayout(right_column)

        main_layout.addLayout(columns_layout)
        main_layout.addStretch()

        return widget

    def _on_telemetry_toggled(self, checked):
        """Handle telemetry checkbox toggle."""
        from core import config as app_config
        app_config.set_telemetry_enabled(checked)

    def _on_crash_reports_toggled(self, checked):
        """Handle crash reports checkbox toggle."""
        from core import config as app_config
        app_config.set_crash_reports_enabled(checked)

    def _show_telemetry_details(self):
        """Show detailed information about what data is collected."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton

        details_dialog = QDialog(self)
        details_dialog.setWindowTitle("Telemetry Details")
        details_dialog.resize(500, 400)

        layout = QVBoxLayout(details_dialog)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard |
            Qt.TextInteractionFlag.LinksAccessibleByMouse |
            Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        browser.setHtml("""
            <h3 style="color: #2a7fff;">What Data is Collected?</h3>

            <h4>Usage Data</h4>
            <p>When you use PhysioMetrics, we collect anonymous usage statistics:</p>
            <ul>
                <li><b>File types:</b> Whether you load ABF, SMRX, or EDF files (not file names)</li>
                <li><b>Features used:</b> Which tools you use (GMM, manual editing, spectral analysis)</li>
                <li><b>Export types:</b> What you export (PDF, CSV, NPZ)</li>
                <li><b>Session duration:</b> How long you use the app</li>
                <li><b>System info:</b> OS, Python version, PhysioMetrics version</li>
            </ul>

            <h4>Crash Reports</h4>
            <p>If the app crashes, we collect:</p>
            <ul>
                <li><b>Error messages:</b> What went wrong</li>
                <li><b>Stack traces:</b> Where the error occurred in the code</li>
                <li><b>No data:</b> Your files and data are NEVER included</li>
            </ul>

            <h4>Anonymous User ID</h4>
            <p>A random UUID (like "a3f2e8c9-4b7d-...") is generated once and stored on your computer.
            This lets us count unique users without knowing who you are.</p>

            <p style="color: #FFD700; font-weight: bold;">What we NEVER collect:</p>
            <ul>
                <li>File names, paths, or directory structure</li>
                <li>Animal metadata (strain, virus, injection site)</li>
                <li>Actual breathing data (frequencies, amplitudes, metrics)</li>
                <li>Your name, email, or institution</li>
            </ul>

            <p><b>You have complete control.</b> You can disable telemetry at any time using the checkboxes above.</p>
        """)
        layout.addWidget(browser)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(details_dialog.close)
        layout.addWidget(close_btn)

        details_dialog.setStyleSheet(self.styleSheet())
        details_dialog.exec()

    def _on_check_updates_clicked(self):
        """Handle manual check for updates button click."""
        # Show "checking..." message
        self.update_label.setText('<p style="color: #888;">Checking for updates...</p>')

        # Force a fresh check (ignore cached update_info)
        self._check_for_updates_async(force_check=True)

    def _check_for_updates_async(self, force_check=False):
        """Check for updates in background thread and update label.

        Args:
            force_check: If True, always check GitHub even if update_info was cached
        """
        from core import update_checker

        # If update info already provided from main window and not forcing, use it immediately
        if self.update_info is not None and not force_check:
            self.update_label.setText(update_checker.get_update_message(self.update_info))
            return

        # Otherwise, check in background
        from PyQt6.QtCore import QThread, pyqtSignal

        class UpdateChecker(QThread):
            """Background thread for checking updates."""
            update_checked = pyqtSignal(object)  # Emits update_info or None

            def run(self):
                """Run update check in background."""
                update_info = update_checker.check_for_updates()
                self.update_checked.emit(update_info)

        def on_update_checked(update_info):
            """Handle update check result."""
            if update_info:
                # Update available
                self.update_label.setText(update_checker.get_update_message(update_info))
                # Update cached info
                self.update_info = update_info
            else:
                # No update or check failed
                if force_check:
                    # User explicitly clicked button - show "up to date" message
                    self.update_label.setText(update_checker.get_no_update_message())
                else:
                    # Silent automatic check - don't show anything
                    pass

        # Create and start background thread
        self.update_thread = UpdateChecker()
        self.update_thread.update_checked.connect(on_update_checked)
        self.update_thread.start()

    def _apply_dark_theme(self):
        """Apply dark theme styling to the dialog."""
        self.setStyleSheet("""
            QDialog, QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QTextBrowser {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                padding: 10px;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #4a4a4a;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #e0e0e0;
                padding: 8px 16px;
                border: 1px solid #3a3a3a;
            }
            QTabBar::tab:selected {
                background-color: #3a3a3a;
                color: #e0e0e0;
            }
            QTabBar::tab:hover {
                background-color: #000000;
                color: #ffffff;
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
