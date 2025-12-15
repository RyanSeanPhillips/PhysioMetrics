"""
AI Settings Dialog - Configure and test AI integration.

Allows users to:
- Use Ollama for FREE local AI (no API key needed!)
- Enter API keys for Claude, OpenAI, or Gemini
- Test connection
- Try simple AI features (file grouping suggestions, trace analysis)
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QTextEdit, QGroupBox, QMessageBox,
    QTabWidget, QWidget, QFileDialog, QCheckBox
)
from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from pathlib import Path
from typing import Optional, Dict, List
import json


class AIWorker(QThread):
    """Background worker for AI API calls."""
    finished = pyqtSignal(object)  # AIResponse or Exception
    progress = pyqtSignal(str)  # Status message

    def __init__(self, client, method: str, **kwargs):
        super().__init__()
        self.client = client
        self.method = method
        self.kwargs = kwargs

    def run(self):
        try:
            self.progress.emit(f"Calling {self.method}...")
            if self.method == "complete":
                result = self.client.complete(**self.kwargs)
            elif self.method == "analyze_trace":
                result = self.client.analyze_respiratory_trace(**self.kwargs)
            elif self.method == "suggest_groupings":
                result = self.client.suggest_file_groupings(**self.kwargs)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(e)


class AISettingsDialog(QDialog):
    """Dialog for configuring AI integration settings."""

    def __init__(self, parent=None, files_metadata: List[Dict] = None):
        super().__init__(parent)
        self.setWindowTitle("AI Integration Settings")
        self.setMinimumSize(600, 500)

        self.files_metadata = files_metadata or []
        self.ai_client = None
        self.worker = None

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Create the dialog UI."""
        layout = QVBoxLayout(self)

        # Create tabs
        tabs = QTabWidget()

        # Tab 1: Settings
        settings_tab = self._create_settings_tab()
        tabs.addTab(settings_tab, "Settings")

        # Tab 2: Test / Demo
        test_tab = self._create_test_tab()
        tabs.addTab(test_tab, "Test AI")

        # Tab 3: File Grouping (if files available)
        if self.files_metadata:
            grouping_tab = self._create_grouping_tab()
            tabs.addTab(grouping_tab, "File Grouping")

        layout.addWidget(tabs)

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        button_layout.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _create_settings_tab(self) -> QWidget:
        """Create the settings configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Provider selection
        provider_group = QGroupBox("AI Provider")
        provider_layout = QVBoxLayout(provider_group)

        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            "🆓 Ollama (Local - FREE!)",
            "Claude (Anthropic)",
            "GPT (OpenAI)",
            "Gemini (Google)"
        ])
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        provider_row.addWidget(self.provider_combo)
        provider_row.addStretch()
        provider_layout.addLayout(provider_row)

        # Model selection (for Ollama)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumWidth(200)
        model_row.addWidget(self.model_combo)

        self.refresh_models_btn = QPushButton("↻")
        self.refresh_models_btn.setMaximumWidth(30)
        self.refresh_models_btn.setToolTip("Refresh available models from Ollama")
        self.refresh_models_btn.clicked.connect(self._refresh_ollama_models)
        model_row.addWidget(self.refresh_models_btn)
        model_row.addStretch()
        provider_layout.addLayout(model_row)

        # API Key input
        key_row = QHBoxLayout()
        self.api_key_label = QLabel("API Key:")
        key_row.addWidget(self.api_key_label)
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Enter your API key...")
        key_row.addWidget(self.api_key_input)

        self.show_key_checkbox = QCheckBox("Show")
        self.show_key_checkbox.toggled.connect(self._toggle_key_visibility)
        key_row.addWidget(self.show_key_checkbox)

        provider_layout.addLayout(key_row)

        # Status/help text
        self.provider_status = QLabel("")
        self.provider_status.setWordWrap(True)
        self.provider_status.setStyleSheet("color: gray; font-size: 11px;")
        self.provider_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        provider_layout.addWidget(self.provider_status)

        layout.addWidget(provider_group)

        # Test connection button
        test_row = QHBoxLayout()
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self._test_connection)
        test_row.addWidget(self.test_btn)

        self.connection_status = QLabel("")
        self.connection_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        test_row.addWidget(self.connection_status)
        test_row.addStretch()

        layout.addLayout(test_row)

        # Info about providers
        info_group = QGroupBox("About AI Providers")
        info_layout = QVBoxLayout(info_group)
        info_text = QLabel(
            "<b>🆓 FREE Option - Ollama (Recommended to start!):</b><br>"
            "• Runs entirely on your computer - no API key needed!<br>"
            "• Download: <a href='https://ollama.com/download'>https://ollama.com/download</a><br>"
            "• After installing, run: <code>ollama pull llama3.2</code><br><br>"
            "<b>Paid Cloud Options (require API keys):</b><br>"
            "• Claude: <a href='https://console.anthropic.com/'>https://console.anthropic.com/</a><br>"
            "• GPT: <a href='https://platform.openai.com/api-keys'>https://platform.openai.com/api-keys</a><br>"
            "• Gemini: <a href='https://aistudio.google.com/app/apikey'>https://aistudio.google.com/app/apikey</a><br><br>"
            "<i>API keys are stored locally on your computer and never shared.</i>"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("font-size: 11px;")
        info_text.setOpenExternalLinks(True)  # Make links clickable
        info_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        info_layout.addWidget(info_text)
        layout.addWidget(info_group)

        layout.addStretch()
        return widget

    def _create_test_tab(self) -> QWidget:
        """Create the AI testing tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Simple prompt test
        prompt_group = QGroupBox("Test Prompt")
        prompt_layout = QVBoxLayout(prompt_group)

        self.test_prompt = QTextEdit()
        self.test_prompt.setPlaceholderText("Enter a test prompt...")
        self.test_prompt.setMaximumHeight(100)
        self.test_prompt.setPlainText("What is 2 + 2? Explain briefly.")
        prompt_layout.addWidget(self.test_prompt)

        send_btn = QPushButton("Send to AI")
        send_btn.clicked.connect(self._send_test_prompt)
        prompt_layout.addWidget(send_btn)

        layout.addWidget(prompt_group)

        # Response area
        response_group = QGroupBox("AI Response")
        response_layout = QVBoxLayout(response_group)

        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setPlaceholderText("Response will appear here...")
        response_layout.addWidget(self.response_text)

        # Token usage
        self.usage_label = QLabel("")
        self.usage_label.setStyleSheet("color: gray; font-size: 11px;")
        self.usage_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        response_layout.addWidget(self.usage_label)

        layout.addWidget(response_group)

        return widget

    def _create_grouping_tab(self) -> QWidget:
        """Create the file grouping suggestions tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Info
        info = QLabel(f"Ask AI to suggest how to group {len(self.files_metadata)} files into experiments.")
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(info)

        # Preview of files
        files_group = QGroupBox(f"Files to Analyze ({len(self.files_metadata)})")
        files_layout = QVBoxLayout(files_group)

        files_preview = QTextEdit()
        files_preview.setReadOnly(True)
        files_preview.setMaximumHeight(150)

        # Show first 10 files
        preview_text = ""
        for i, f in enumerate(self.files_metadata[:10]):
            preview_text += f"{f.get('file_name', 'Unknown')} - {f.get('protocol', '')}\n"
        if len(self.files_metadata) > 10:
            preview_text += f"... and {len(self.files_metadata) - 10} more files"
        files_preview.setPlainText(preview_text)
        files_layout.addWidget(files_preview)

        layout.addWidget(files_group)

        # Suggest button
        suggest_btn = QPushButton("Get AI Grouping Suggestions")
        suggest_btn.clicked.connect(self._get_grouping_suggestions)
        layout.addWidget(suggest_btn)

        # Suggestions output
        suggestions_group = QGroupBox("AI Suggestions")
        suggestions_layout = QVBoxLayout(suggestions_group)

        self.suggestions_text = QTextEdit()
        self.suggestions_text.setReadOnly(True)
        self.suggestions_text.setPlaceholderText("Grouping suggestions will appear here...")
        suggestions_layout.addWidget(self.suggestions_text)

        layout.addWidget(suggestions_group)

        return widget

    def _on_provider_changed(self, index: int):
        """Handle provider selection change."""
        providers = ["ollama", "claude", "openai", "gemini"]
        provider = providers[index]

        is_ollama = (provider == "ollama")

        # Show/hide API key input based on provider
        self.api_key_label.setVisible(not is_ollama)
        self.api_key_input.setVisible(not is_ollama)
        self.show_key_checkbox.setVisible(not is_ollama)

        # Show/hide model selection and refresh button
        self.model_combo.setVisible(is_ollama)
        self.refresh_models_btn.setVisible(is_ollama)

        # Check if SDK/server is available
        try:
            from core.ai_client import check_provider_available, get_ollama_models
            status = check_provider_available(provider)

            if provider == "ollama":
                if not status["sdk_installed"]:
                    self.provider_status.setText(
                        "⚠️ OpenAI SDK needed. Run: pip install openai"
                    )
                    self.provider_status.setStyleSheet("color: orange; font-size: 11px;")
                elif not status["server_running"]:
                    self.provider_status.setText(
                        "⚠️ Ollama not running. Download from https://ollama.com/download\n"
                        "Then run: ollama pull llama3.2"
                    )
                    self.provider_status.setStyleSheet("color: orange; font-size: 11px;")
                else:
                    self.provider_status.setText("✓ Ollama is running (FREE!)")
                    self.provider_status.setStyleSheet("color: green; font-size: 11px;")
                    # Load available models
                    self._refresh_ollama_models()
            else:
                if not status["sdk_installed"]:
                    sdk_names = {
                        "claude": "anthropic",
                        "openai": "openai",
                        "gemini": "google-generativeai"
                    }
                    self.provider_status.setText(
                        f"⚠️ SDK not installed. Run: pip install {sdk_names[provider]}"
                    )
                    self.provider_status.setStyleSheet("color: orange; font-size: 11px;")
                else:
                    self.provider_status.setText("✓ SDK installed")
                    self.provider_status.setStyleSheet("color: green; font-size: 11px;")

                # Load saved API key for this provider
                settings = QSettings("PhysioMetrics", "BreathAnalysis")
                key = settings.value(f"ai/{provider}_api_key", "")
                self.api_key_input.setText(key if key else "")

        except ImportError:
            self.provider_status.setText("AI client module not found")
            self.provider_status.setStyleSheet("color: red; font-size: 11px;")

    def _refresh_ollama_models(self):
        """Refresh the list of available Ollama models."""
        try:
            from core.ai_client import get_ollama_models
            models = get_ollama_models()

            self.model_combo.clear()
            if models:
                self.model_combo.addItems(models)
            else:
                # Add default suggestions if no models found
                self.model_combo.addItems(["llama3.2", "llama3.2:1b", "mistral", "codellama"])
                self.provider_status.setText(
                    "⚠️ No models found. Run: ollama pull llama3.2"
                )
                self.provider_status.setStyleSheet("color: orange; font-size: 11px;")
        except Exception as e:
            self.model_combo.clear()
            self.model_combo.addItems(["llama3.2", "llama3.2:1b", "mistral"])
            print(f"[ai-settings] Error fetching Ollama models: {e}")

    def _toggle_key_visibility(self, show: bool):
        """Toggle API key visibility."""
        if show:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)

    def _get_current_provider(self) -> str:
        """Get currently selected provider name."""
        providers = ["ollama", "claude", "openai", "gemini"]
        return providers[self.provider_combo.currentIndex()]

    def _get_current_model(self) -> Optional[str]:
        """Get currently selected model (for Ollama)."""
        if self._get_current_provider() == "ollama":
            return self.model_combo.currentText().strip() or None
        return None

    def _init_client(self) -> bool:
        """Initialize AI client with current settings. Returns True if successful."""
        provider = self._get_current_provider()

        if provider == "ollama":
            # Ollama doesn't need API key
            model = self._get_current_model()
            try:
                from core.ai_client import AIClient, check_ollama_running
                if not check_ollama_running():
                    QMessageBox.warning(self, "Ollama Not Running",
                        "Ollama server is not running.\n\n"
                        "1. Download Ollama (FREE): https://ollama.com/download\n"
                        "2. Install and run it\n"
                        "3. Run: ollama pull llama3.2\n"
                        "4. Try again")
                    return False
                self.ai_client = AIClient(provider="ollama", model=model)
                return True
            except ImportError as e:
                QMessageBox.critical(self, "Import Error",
                                   f"Could not import AI client:\n{e}\n\n"
                                   "Run: pip install openai")
                return False
            except Exception as e:
                QMessageBox.critical(self, "Connection Error", f"Failed to connect to Ollama:\n{e}")
                return False
        else:
            # Cloud providers need API key
            api_key = self.api_key_input.text().strip()

            if not api_key:
                QMessageBox.warning(self, "Missing API Key", "Please enter an API key.")
                return False

            try:
                from core.ai_client import AIClient
                self.ai_client = AIClient(provider=provider, api_key=api_key)
                return True
            except ImportError as e:
                QMessageBox.critical(self, "Import Error",
                                   f"Could not import AI client:\n{e}\n\n"
                                   f"Make sure the required SDK is installed.")
                return False
            except Exception as e:
                QMessageBox.critical(self, "Connection Error", f"Failed to initialize AI client:\n{e}")
                return False

    def _test_connection(self):
        """Test the AI connection with a simple prompt."""
        if not self._init_client():
            return

        self.test_btn.setEnabled(False)
        self.connection_status.setText("Testing...")
        self.connection_status.setStyleSheet("color: gray;")

        # Create worker for background API call
        self.worker = AIWorker(
            self.ai_client,
            "complete",
            prompt="Say 'Connection successful!' and nothing else.",
            max_tokens=20
        )
        self.worker.finished.connect(self._on_test_finished)
        self.worker.start()

    def _on_test_finished(self, result):
        """Handle test connection result."""
        self.test_btn.setEnabled(True)

        if isinstance(result, Exception):
            error_msg = str(result)
            # Show short version in label, full in tooltip
            short_msg = error_msg[:80] + "..." if len(error_msg) > 80 else error_msg
            self.connection_status.setText(f"❌ Failed (click for details)")
            self.connection_status.setStyleSheet("color: red; text-decoration: underline; cursor: pointer;")
            self.connection_status.setToolTip(error_msg)
            # Store full error for click handler
            self._last_error = error_msg
            self.connection_status.mousePressEvent = lambda e: self._show_error_details()
        else:
            self.connection_status.setText(f"✓ Connected to {result.model}")
            self.connection_status.setStyleSheet("color: green;")
            self.connection_status.setToolTip("")
            self._last_error = None

    def _show_error_details(self):
        """Show full error message in a dialog."""
        if hasattr(self, '_last_error') and self._last_error:
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton

            dialog = QDialog(self)
            dialog.setWindowTitle("Connection Error Details")
            dialog.setMinimumSize(500, 300)

            layout = QVBoxLayout(dialog)

            error_text = QTextEdit()
            error_text.setReadOnly(True)
            error_text.setPlainText(self._last_error)
            layout.addWidget(error_text)

            # Add explanation for common errors
            explanation = ""
            if "401" in self._last_error:
                explanation = "\n\n💡 Error 401 = Invalid API key. Check that you copied the full key correctly."
            elif "429" in self._last_error:
                explanation = "\n\n💡 Error 429 = Rate limit or quota exceeded. Your account may need credits loaded or payment setup."
            elif "403" in self._last_error:
                explanation = "\n\n💡 Error 403 = Access denied. Your API key may not have permission for this model."

            if explanation:
                error_text.append(explanation)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)

            dialog.exec()

    def _send_test_prompt(self):
        """Send the test prompt to AI."""
        if not self._init_client():
            return

        prompt = self.test_prompt.toPlainText().strip()
        if not prompt:
            return

        self.response_text.setPlainText("Waiting for response...")
        self.usage_label.setText("")

        self.worker = AIWorker(
            self.ai_client,
            "complete",
            prompt=prompt,
            max_tokens=1000
        )
        self.worker.finished.connect(self._on_prompt_finished)
        self.worker.start()

    def _on_prompt_finished(self, result):
        """Handle prompt response."""
        if isinstance(result, Exception):
            self.response_text.setPlainText(f"Error: {result}")
        else:
            self.response_text.setPlainText(result.content)
            self.usage_label.setText(
                f"Model: {result.model} | "
                f"Tokens: {result.usage.get('input_tokens', 0)} in, "
                f"{result.usage.get('output_tokens', 0)} out"
            )

    def _get_grouping_suggestions(self):
        """Get AI suggestions for file grouping."""
        if not self._init_client():
            return

        if not self.files_metadata:
            QMessageBox.warning(self, "No Files", "No files available to analyze.")
            return

        self.suggestions_text.setPlainText("Analyzing files...")

        self.worker = AIWorker(
            self.ai_client,
            "suggest_groupings",
            files_metadata=self.files_metadata
        )
        self.worker.finished.connect(self._on_grouping_finished)
        self.worker.start()

    def _on_grouping_finished(self, result):
        """Handle grouping suggestions result."""
        if isinstance(result, Exception):
            self.suggestions_text.setPlainText(f"Error: {result}")
        else:
            # Try to pretty-print if it's JSON
            content = result.content
            try:
                # Find JSON in response
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    parsed = json.loads(json_str)
                    content = json.dumps(parsed, indent=2)
            except:
                pass  # Keep original content
            self.suggestions_text.setPlainText(content)

    def _save_settings(self):
        """Save current settings to QSettings."""
        settings = QSettings("PhysioMetrics", "BreathAnalysis")

        provider = self._get_current_provider()

        # Save provider preference
        settings.setValue("ai/provider", provider)

        if provider == "ollama":
            # Save selected model for Ollama
            model = self.model_combo.currentText().strip()
            if model:
                settings.setValue("ai/ollama_model", model)
        else:
            # Save API key for cloud providers
            api_key = self.api_key_input.text().strip()
            if api_key:
                settings.setValue(f"ai/{provider}_api_key", api_key)

        self.connection_status.setText("Settings saved")
        self.connection_status.setStyleSheet("color: green;")

    def _load_settings(self):
        """Load settings from QSettings."""
        settings = QSettings("PhysioMetrics", "BreathAnalysis")

        # Load provider preference (default to ollama since it's free)
        provider = settings.value("ai/provider", "ollama")
        provider_index = {"ollama": 0, "claude": 1, "openai": 2, "gemini": 3}.get(provider, 0)
        self.provider_combo.setCurrentIndex(provider_index)

        # Load saved Ollama model
        if provider == "ollama":
            saved_model = settings.value("ai/ollama_model", "llama3.2")
            idx = self.model_combo.findText(saved_model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            else:
                self.model_combo.setCurrentText(saved_model)

        # Trigger provider change to load API key / check status
        self._on_provider_changed(provider_index)


# Test dialog standalone
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Sample file metadata for testing
    test_files = [
        {"file_name": "25121004.abf", "protocol": "30Hz_opto_10s", "file_path": "/data/mouse1/25121004.abf"},
        {"file_name": "25121005.abf", "protocol": "30Hz_opto_10s", "file_path": "/data/mouse1/25121005.abf"},
        {"file_name": "25121006.abf", "protocol": "baseline", "file_path": "/data/mouse1/25121006.abf"},
        {"file_name": "25121010.abf", "protocol": "6Hz_opto_30s", "file_path": "/data/mouse2/25121010.abf"},
    ]

    dialog = AISettingsDialog(files_metadata=test_files)
    dialog.exec()
