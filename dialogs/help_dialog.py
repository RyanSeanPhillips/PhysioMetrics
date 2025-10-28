"""
PlethApp Help Dialog

Quick reference guide with workflow and exported data documentation.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextBrowser, QLabel, QDialogButtonBox, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import os


class HelpDialog(QDialog):
    """Quick reference help dialog with essential information."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PlethApp - Quick Reference")
        self.resize(850, 700)

        self._setup_ui()
        self._apply_dark_theme()

    def _setup_ui(self):
        """Create the help dialog UI with two tabs."""
        layout = QVBoxLayout(self)

        # Header with app info
        header = QLabel("<h1>PlethApp - Quick Reference Guide</h1>")
        header.setStyleSheet("color: #2a7fff; padding: 10px;")
        layout.addWidget(header)

        # Tab widget for workflow, data files, and about
        tabs = QTabWidget()
        tabs.addTab(self._create_quick_reference(), "Usage & Workflow")
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
        browser.setHtml("""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.5; padding: 10px;">

            <h2 style="color: #2a7fff;">Quick Start</h2>
            <ol style="margin-top: 5px;">
                <li><b>Open File:</b> Ctrl+O → Select .abf, .smrx, or .edf file <i>(already analyzed? you'll be prompted to resume)</i></li>
                <li><b>Select Channel:</b> Choose respiratory signal from dropdown</li>
                <li><b>Detect Peaks:</b> Click "Apply" (threshold auto-detected)</li>
                <li><b>Review/Edit:</b> Click editing mode button (Add/Delete Peak), then click plot to edit. Label sighs if needed.</li>
                <li><b>Preview:</b> Click "View Summary" to review results before export</li>
                <li><b>Export:</b> Click "Save Data" (Ctrl+S) → Enter animal info → Choose which files to save → Save</li>
            </ol>

            <h2 style="color: #2a7fff; margin-top: 20px;">Keyboard Shortcuts</h2>
            <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; margin-top: 5px;">
                <tr style="background-color: #2a2a2a;">
                    <th width="170">Shortcut</th>
                    <th>Action</th>
                </tr>
                <tr><td><b>Ctrl+O</b></td><td>Open file</td></tr>
                <tr><td><b>Ctrl+S</b></td><td>Save/export data</td></tr>
                <tr><td><b>F1</b></td><td>Show this help</td></tr>
                <tr><td><b>Shift+Click</b></td><td>(in add/delete mode) Toggle between Add/Delete peak</td></tr>
                <tr><td><b>Ctrl+Click</b></td><td>(in add/delete mode) Switch to Add Sigh mode</td></tr>
                <tr><td><b>Shift+Click & Drag</b></td><td>(in move point mode) Snap to zero crossings</td></tr>
                <tr><td><b>Shift+Click</b></td><td>(in mark sniff mode) Delete sniffing region</td></tr>
            </table>

            <h2 style="color: #2a7fff; margin-top: 20px;">Key Features</h2>
            <ul style="margin-top: 5px;">
                <li><b>Auto-Threshold:</b> Otsu's method finds optimal peak detection threshold automatically</li>
                <li><b>Manual Editing Modes:</b>
                    <ul style="margin-top: 3px; margin-bottom: 3px;">
                        <li><b>Add/Delete Peak:</b> Click editing mode button first, then Shift+Click toggles modes, Ctrl+Click for sigh mode</li>
                        <li><b>Move Point:</b> Click and drag peaks, onsets, or offsets. Hold Shift to snap to zero crossings</li>
                        <li><b>Mark Sniff:</b> Click and drag to highlight sniffing regions. Shift+Click to delete a region</li>
                        <li><b>Omit Sweep:</b> Exclude specific sweeps from analysis</li>
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
                <li><b>Curation Tab:</b> Consolidate multiple files (best for identical experiments like 30Hz stims)</li>
                <li><b>Auto-Resume:</b> When opening a previously analyzed file/channel, you'll be prompted to resume or start over</li>
            </ul>

            <h2 style="color: #2a7fff; margin-top: 20px;">Common Workflows</h2>

            <h3 style="margin-bottom: 5px;">Basic Analysis</h3>
            <p style="margin-top: 0;">Load file → Select channel → Click Apply → Review → Export</p>

            <h3 style="margin-bottom: 5px;">Adjust Threshold</h3>
            <p style="margin-top: 0;">Click "More Options" → Drag red line → OK → Click Apply</p>

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
                <li><b>.pleth.npz</b> - PlethApp session files (save/load complete analysis)</li>
            </ul>

        </body>
        </html>
        """)
        return browser

    def _create_data_files_tab(self):
        """Create tab documenting exported data file formats."""
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
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
            <p><b>Contains:</b> Complete analysis session state including detected peaks, manual labels, filter settings, GMM clustering results, and all parameters. Enables resuming work later - when you open a file that was previously analyzed, PlethApp prompts: "Resume analysis or start over?"</p>

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
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        # Add splash screen image
        image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'images', 'plethapp_splash_dark-01.png')
        if os.path.exists(image_path):
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            # Scale image to half size
            scaled_pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(image_label)

        # Add version info
        version_label = QLabel(f"<h2 style='color: #2a7fff;'>PlethApp</h2><p style='font-size: 14pt;'>Version {VERSION_STRING}</p>")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

        # Add description
        description = QLabel(
            "<p style='text-align: center; max-width: 500px;'>"
            "Advanced respiratory signal analysis tool for breath pattern detection, "
            "eupnea/apnea identification, and breathing regularity assessment."
            "</p>"
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)

        layout.addStretch()

        return widget

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
