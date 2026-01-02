"""
AI Notebook Manager - Manages AI chat notebook functionality.

This module was extracted from main.py to reduce complexity.
It handles all AI chat interactions, tool calls, and AI commands.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal, QSettings

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow


class AIWorker(QThread):
    """Background worker for AI API calls with native tool support."""
    # response, is_error, input_tokens, output_tokens, tool_calls (JSON string)
    finished = pyqtSignal(str, bool, int, int, str)

    def __init__(self, provider: str, api_key: str, messages: list,
                 context: str, model: str, tools: list, system_prompt: str):
        super().__init__()
        self.provider = provider
        self.api_key = api_key
        self.messages = messages
        self.context = context
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt

    def run(self):
        import time
        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                from core.ai_client import AIClient

                client = AIClient(provider=self.provider, api_key=self.api_key, model=self.model)

                # Use native tool use
                response = client.chat_with_tools(
                    messages=self.messages,
                    tools=self.tools,
                    system_prompt=self.system_prompt,
                    max_tokens=4096,
                    temperature=0.7
                )

                # Extract token usage
                input_tokens = response.usage.get('input_tokens', 0) if response.usage else 0
                output_tokens = response.usage.get('output_tokens', 0) if response.usage else 0

                # Convert tool_calls to JSON string for signal
                tool_calls_json = json.dumps(response.tool_calls) if response.tool_calls else ""

                self.finished.emit(response.content, False, input_tokens, output_tokens, tool_calls_json)
                return  # Success, exit retry loop

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = '429' in str(e) or 'rate_limit' in error_str or 'rate limit' in error_str

                if is_rate_limit and attempt < max_retries - 1:
                    # Rate limit - wait and retry with exponential backoff
                    delay = base_delay * (2 ** attempt)  # 2, 4, 8 seconds
                    time.sleep(delay)
                    continue  # Retry
                else:
                    # Final attempt failed or non-rate-limit error
                    self.finished.emit(str(e), True, 0, 0, "")
                    return


class FollowupWorker(QThread):
    """Background worker for follow-up AI requests after tool execution."""
    finished = pyqtSignal(str, bool, int, int, str)

    def __init__(self, provider: str, api_key: str, messages: list, model: str, tools: list):
        super().__init__()
        self.provider = provider
        self.api_key = api_key
        self.messages = messages
        self.model = model
        self.tools = tools

    def run(self):
        try:
            from core.ai_client import AIClient
            client = AIClient(provider=self.provider, api_key=self.api_key, model=self.model)

            system_prompt = """You received tool results. Now provide a helpful response to the user.
If they asked for code, generate Python code in ```python ``` blocks.
Use the file paths and data from the tool results.
Be concise and helpful."""

            response = client.chat_with_tools(
                messages=self.messages,
                tools=self.tools,
                system_prompt=system_prompt,
                max_tokens=4096,
                temperature=0.7
            )
            input_tokens = response.usage.get('input_tokens', 0) if response.usage else 0
            output_tokens = response.usage.get('output_tokens', 0) if response.usage else 0
            tool_calls_json = json.dumps(response.tool_calls) if response.tool_calls else ""
            self.finished.emit(response.content, False, input_tokens, output_tokens, tool_calls_json)
        except Exception as e:
            self.finished.emit(str(e), True, 0, 0, "")


class AINotebookManager:
    """
    Manages AI chat notebook functionality.

    This class handles:
    - AI chat message sending/receiving
    - Model selection and configuration
    - AI tool definitions and execution
    - AI commands (load project, update metadata, etc.)
    """

    # Soft token warning threshold
    TOKEN_WARNING_THRESHOLD = 500000  # 500k tokens - just a warning

    def __init__(self, main_window: 'QMainWindow'):
        """
        Initialize the AI Notebook Manager.

        Args:
            main_window: Reference to MainWindow for widget access
        """
        self.mw = main_window

        # State variables
        self._chat_conversation_history = []
        self._chat_worker = None
        self._ai_models = []
        self._total_tokens_used = 0
        self._token_warning_shown = False

    # -------------------------------------------------------------------------
    # Model Selection
    # -------------------------------------------------------------------------

    def init_model_selector(self):
        """Initialize the AI model selection dropdown with only AVAILABLE models."""
        settings = QSettings("PhysioMetrics", "BreathAnalysis")

        # Only show models that are actually configured/available
        # Format: (display_name, provider, model_id)
        self._ai_models = []

        # 1. Add Ollama models if running and installed
        ollama_available = False
        try:
            from core.ai_client import get_ollama_models, check_ollama_running
            if check_ollama_running():
                installed_models = get_ollama_models()
                if installed_models:
                    ollama_available = True
                    for model in installed_models[:8]:  # Limit to 8 models
                        self._ai_models.append((f"🆓 Ollama: {model}", "ollama", model))
        except Exception as e:
            print(f"[ai] Could not fetch Ollama models: {e}")

        # 2. Add Claude models only if API key is configured
        claude_key = settings.value("ai/claude_api_key", "")
        if claude_key:
            self._ai_models.append(("Claude: Haiku (fast)", "claude", "claude-3-5-haiku-latest"))
            self._ai_models.append(("Claude: Sonnet (smart)", "claude", "claude-sonnet-4-20250514"))

        # 3. Add OpenAI models only if API key is configured
        openai_key = settings.value("ai/openai_api_key", "")
        if openai_key:
            self._ai_models.append(("GPT-4o", "openai", "gpt-4o"))
            self._ai_models.append(("GPT-4o mini (fast)", "openai", "gpt-4o-mini"))

        # 4. If nothing available, show setup hint
        if not self._ai_models:
            self._ai_models.append(("⚙️ Click gear to set up AI", "setup_hint", ""))

        # Populate combo
        self.mw.modelSelectCombo.clear()
        for display_name, _, _ in self._ai_models:
            self.mw.modelSelectCombo.addItem(display_name)

        # Load saved selection
        saved_provider = settings.value("ai/provider", "ollama")
        saved_model = settings.value("ai/selected_model", "llama3.2")

        # Find and select the saved model
        for i, (_, provider, model_id) in enumerate(self._ai_models):
            if provider == saved_provider and model_id == saved_model:
                self.mw.modelSelectCombo.setCurrentIndex(i)
                break

        # Connect change signal to save selection
        self.mw.modelSelectCombo.currentIndexChanged.connect(self.on_model_changed)

    def on_model_changed(self, index: int):
        """Save selected model to settings."""
        if 0 <= index < len(self._ai_models):
            _, provider, model_id = self._ai_models[index]
            settings = QSettings("PhysioMetrics", "BreathAnalysis")
            settings.setValue("ai/provider", provider)
            settings.setValue("ai/selected_model", model_id)

    def get_selected_model(self) -> tuple:
        """Get the currently selected (provider, model_id) tuple."""
        if hasattr(self.mw, 'modelSelectCombo'):
            index = self.mw.modelSelectCombo.currentIndex()
            if 0 <= index < len(self._ai_models):
                _, provider, model_id = self._ai_models[index]
                return (provider, model_id)
        # Default to Ollama
        return ("ollama", "llama3.2")

    def get_model_display_name(self) -> str:
        """Get a short display name for the current model."""
        if hasattr(self.mw, 'modelSelectCombo'):
            index = self.mw.modelSelectCombo.currentIndex()
            if 0 <= index < len(self._ai_models):
                display_name, provider, model_id = self._ai_models[index]
                # Extract short name
                if provider == "ollama":
                    return model_id.split(":")[0].capitalize()
                elif 'haiku' in model_id.lower():
                    return "Haiku"
                elif 'sonnet' in model_id.lower():
                    return "Sonnet"
                elif 'gpt-4o-mini' in model_id.lower():
                    return "GPT-4o mini"
                elif 'gpt-4o' in model_id.lower():
                    return "GPT-4o"
                else:
                    return model_id.split("-")[0].capitalize()
        return "AI"

    # -------------------------------------------------------------------------
    # Chat Operations
    # -------------------------------------------------------------------------

    def on_chat_send(self):
        """Handle sending a message in the chatbot panel."""
        if not hasattr(self.mw, 'chatInputEdit') or not hasattr(self.mw, 'chatHistoryText'):
            return

        message = self.mw.chatInputEdit.text().strip()
        if not message:
            return

        # Add user message to chat history
        self.mw.chatHistoryText.append(f"<b style='color: #569cd6;'>You:</b> {message}")
        self.mw.chatInputEdit.clear()

        # Get selected provider and model from dropdown
        provider, model = self.get_selected_model()

        # Check if provider is available
        settings = QSettings("PhysioMetrics", "BreathAnalysis")

        if provider == "setup_hint":
            # User selected the "Click gear to set up AI" hint
            self.mw.chatHistoryText.append(
                f"<b style='color: #f48771;'>AI Setup Required:</b><br><br>"
                f"<b>Option 1 - Ollama (FREE, runs locally):</b><br>"
                f"• Download from: <a href='https://ollama.com/download'>https://ollama.com/download</a><br>"
                f"• Run: <code>ollama pull llama3.2</code><br><br>"
                f"<b>Option 2 - Cloud API (requires account):</b><br>"
                f"• Click ⚙️ gear icon to add API key<br>"
                f"• Supports Claude and GPT models"
            )
            self._scroll_chat_to_bottom()
            return

        if provider == "ollama":
            # Ollama doesn't need API key - check if server is running
            try:
                from core.ai_client import check_ollama_running
                if not check_ollama_running():
                    self.mw.chatHistoryText.append(
                        f"<b style='color: #f44747;'>Error:</b> Ollama is not running.<br>"
                        f"<i>Download FREE from: <a href='https://ollama.com/download'>https://ollama.com/download</a><br>"
                        f"Then run: ollama pull {model}</i>"
                    )
                    self._scroll_chat_to_bottom()
                    return
                # Use Ollama (no API key needed)
                self._send_to_ai_api(message, provider, api_key=None, model=model)
            except ImportError as e:
                self.mw.chatHistoryText.append(
                    f"<b style='color: #f44747;'>Error:</b> {e}"
                )
                self._scroll_chat_to_bottom()
        else:
            # Cloud provider - needs API key
            api_key = settings.value(f"ai/{provider}_api_key", "")

            if api_key:
                # Use real AI API
                self._send_to_ai_api(message, provider, api_key, model=model)
            else:
                # No API key configured
                self.mw.chatHistoryText.append(
                    f"<b style='color: #f44747;'>Error:</b> No API key configured for {provider}.<br>"
                    f"<i>Click the ⚙️ gear icon to configure, or select an Ollama model (FREE!).</i>"
                )
                self._scroll_chat_to_bottom()

    def _send_to_ai_api(self, message: str, provider: str, api_key: str, model: str = None):
        """Send message to actual AI API in background thread."""
        from PyQt6.QtWidgets import QMessageBox

        # Check token usage and warn (but allow continuing)
        if self._total_tokens_used >= self.TOKEN_WARNING_THRESHOLD and not self._token_warning_shown:
            msg = QMessageBox(self.mw)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("High Token Usage")
            msg.setText(f"You have used {self._total_tokens_used:,} tokens in this session.")
            msg.setInformativeText(
                "This is just a heads up about your usage.\n\n"
                "Click 'Continue' to keep chatting, or 'Clear Chat' to start fresh."
            )
            msg.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
            clear_btn = msg.addButton("Clear Chat", QMessageBox.ButtonRole.DestructiveRole)
            msg.exec()

            if msg.clickedButton() == clear_btn:
                self.clear_chat_history()
                return

            self._token_warning_shown = True

        # Add user message to conversation history
        self._chat_conversation_history.append({
            "role": "user",
            "content": message
        })

        # Disable send button, enable stop button
        if hasattr(self.mw, 'chatSendButton'):
            self.mw.chatSendButton.setEnabled(False)
        if hasattr(self.mw, 'chatStopButton'):
            self.mw.chatStopButton.setEnabled(True)

        # Show thinking indicator
        self.mw.chatHistoryText.append("<i style='color: #888;'>Thinking...</i>")
        self._scroll_chat_to_bottom()

        # Build context
        context = self._build_ai_context()

        # Prepare messages for API (copy to avoid mutation)
        messages_for_api = list(self._chat_conversation_history)

        # Use the model passed as parameter
        selected_model = model

        # Define native tools for Claude (Ollama may not support tools)
        tools = self._get_ai_tool_definitions() if provider != "ollama" else []

        # Build system prompt
        system_prompt = self._build_system_prompt(context)

        # Create worker thread
        self._chat_worker = AIWorker(
            provider, api_key, messages_for_api, context,
            selected_model, tools, system_prompt
        )
        self._chat_worker.finished.connect(self._on_ai_response)
        self._chat_worker.start()

    def _build_system_prompt(self, context: str) -> str:
        """Build the comprehensive system prompt for AI."""
        return f"""You are an AI assistant for PhysioMetrics, a respiratory signal analysis application.
You help users analyze plethysmography data and generate Python code for plotting and analysis.

=== DEMO REQUEST ===

If the user says "demo", "example", "show me", "sample plot", or similar:

First, use list_available_files() to find a file with exports, 10 sweeps, and "30hz" in name/protocol.

Then: "Create a demo plot showing breathing frequency response to optogenetic stimulation. Use a single panel with dark theme (plt.style.use('dark_background')), plot individual sweeps in light gray, mean frequency in cyan with SEM shading, and highlight stimulation periods with steelblue rectangles. Make it publication-ready with white text and clean styling."

Helper functions to use:
- load_means_csv(filename) - DataFrame with frequency_sweep0, frequency_sweep1, ..., frequency_mean, frequency_sem
- load_bundle_npz(filename) - bundle with stim timing
- get_stim_spans(bundle) - returns [(start_time, end_time), ...]
- add_stim_shading(ax, stim_spans, color='steelblue', alpha=0.4) - adds stim rectangles

=== SEARCH STRATEGY ===
ALWAYS call get_searchable_values() FIRST before searching! This shows you:
- Exact protocol names, status values, animal IDs, etc. that exist in the data
- Which files have exports

IMPORTANT: If the user asks for something that doesn't exist in the searchable values:
- DON'T search for it (you'll get 0 results)
- TELL the user what's actually available and suggest alternatives

=== QUICK CONTEXT ===
{context}

=== EXPORTED DATA FILE STRUCTURE ===

PhysioMetrics exports several file types. Use the export_path from file context above.

**1. _means_by_time.csv** - Time-series metrics (best for plotting over time)
Columns:
- `time`: Time in seconds (relative to stim start if stim present)
- `sweep`: Sweep index (0, 1, 2, ...)
- For each metric: `<metric>_sweep0`, `<metric>_sweep1`, etc., `<metric>_mean`, `<metric>_sem`
- Normalized versions: `<metric>_norm_mean`, `<metric>_norm_sem` (baseline-normalized)

Key metrics available:
- `frequency` or `if` - Instantaneous frequency (Hz, breaths/min)
- `amp_insp` - Inspiratory amplitude
- `amp_exp` - Expiratory amplitude
- `ti` - Inspiratory time (seconds)
- `te` - Expiratory time (seconds)
- `ttot` - Total breath cycle time (seconds)
- `area_insp` - Inspiratory area under curve
- `area_exp` - Expiratory area under curve
- `duty_cycle` - Ti/Ttot ratio (0-1)
- `ve` - Minute ventilation (frequency × amplitude)

**2. _breaths.csv** - Per-breath data (one row per breath)
Columns:
- `sweep_idx`, `breath_idx` - Identifiers
- `time_onset`, `time_peak` - Timing (seconds)
- All metrics above, plus:
- `is_in_stim` - Boolean, True if breath occurred during stimulation
- `is_eupnea` - Boolean, True if classified as eupnea
- `is_sniffing` - Boolean, True if classified as sniffing

**3. _bundle.npz** - Complete data bundle (numpy format)
Load with: `data = np.load('file_bundle.npz', allow_pickle=True)`
Contains:
- `t_downsampled` - Time array (downsampled)
- `trace_downsampled` - Signal trace per sweep
- `stim_spans_json` - JSON string with stim timing
- Metric arrays for each sweep

**4. Stimulation Timing**
From _bundle.npz, stim_spans are stored as JSON:
```python
import json
stim_spans = json.loads(str(data['stim_spans_json']))
# stim_spans is dict: {{sweep_idx: [(start_time, end_time), ...]}}
```
From _breaths.csv, use `is_in_stim` column.

=== HELPER FUNCTIONS (available in Code Notebook) ===

The Code Notebook has these helper functions pre-loaded:

```python
# List all files with exports
print(list_available_files())

# Load data by filename (partial match)
df = load_means_csv('xxx.abf')      # Returns DataFrame from _means_by_time.csv
df = load_breaths_csv('xxx.abf')    # Returns DataFrame from _breaths.csv
data = load_bundle_npz('xxx.abf')   # Returns numpy NpzFile

# Get stim timing from bundle
stim_spans = get_stim_spans(data)   # Returns {{sweep_idx: [(start, end), ...]}}

# Add stim shading to plot
add_stim_shading(ax, stim_spans, sweep_idx=0, color='blue', alpha=0.2)

# Get export paths
paths = get_export_paths('xxx')     # Returns {{filename: export_path}}
```

=== CODE TEMPLATES ===

**Plot frequency vs time with mean±SEM and stim shading (RECOMMENDED):**
```python
# Load data using helper function
df = load_means_csv('xxx.abf')  # Replace xxx.abf with actual filename
bundle = load_bundle_npz('xxx.abf')
stim_spans = get_stim_spans(bundle)

fig, ax = plt.subplots(figsize=(12, 6))

# Overlay all sweeps in gray
sweep_cols = [c for c in df.columns if c.startswith('frequency_sweep')]
for col in sweep_cols:
    ax.plot(df['time'], df[col], alpha=0.3, color='gray', linewidth=0.5)

# Plot mean±SEM
ax.fill_between(df['time'],
                df['frequency_mean'] - df['frequency_sem'],
                df['frequency_mean'] + df['frequency_sem'],
                alpha=0.3, color='blue', label='Mean ± SEM')
ax.plot(df['time'], df['frequency_mean'], 'b-', linewidth=2, label='Mean')

# Add blue stim shading
add_stim_shading(ax, stim_spans, color='blue', alpha=0.2, label='Laser ON')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.legend()
ax.set_title('Breathing Frequency Over Time')
```

**Compare eupnea vs sniffing:**
```python
df = load_breaths_csv('xxx.abf')
eupnea = df[df['is_eupnea'] == True]
sniffing = df[df['is_sniffing'] == True]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(eupnea['frequency'], bins=30, alpha=0.7, label='Eupnea')
ax[0].hist(sniffing['frequency'], bins=30, alpha=0.7, label='Sniffing')
ax[0].legend()
ax[0].set_xlabel('Frequency (Hz)')
ax[1].boxplot([eupnea['ti'].dropna(), sniffing['ti'].dropna()], labels=['Eupnea', 'Sniffing'])
ax[1].set_ylabel('Ti (s)')
```

**Multi-metric comparison:**
```python
df = load_means_csv('xxx.abf')
metrics = ['frequency', 'amp_insp', 'ti', 'duty_cycle']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, metric in zip(axes.flat, metrics):
    mean_col = f'{{metric}}_mean'
    sem_col = f'{{metric}}_sem'
    if mean_col in df.columns:
        ax.fill_between(df['time'], df[mean_col]-df[sem_col], df[mean_col]+df[sem_col], alpha=0.3)
        ax.plot(df['time'], df[mean_col])
        ax.set_ylabel(metric)
        ax.set_xlabel('Time (s)')
plt.tight_layout()
```

=== OTHER COMMANDS (include in your response text) ===
- [LOAD_PROJECT: project_name] - Load a saved project
- [UPDATE_META: filename | field=value] - Update file metadata

WHEN GENERATING CODE:
- ALWAYS use actual filenames from the context or search results (NOT placeholder 'xxx.abf')
- Helper functions accept: full paths, filenames, or partial name matches
- For raw Windows paths, use raw strings: r'C:\\path\\to\\file.csv'
- Import: pandas as pd, matplotlib.pyplot as plt, numpy as np
- Code runs in a Code Notebook - user clicks "Run" to execute
- Plots will render inline in the output area

Be concise. Use markdown. Put code in ```python ``` blocks."""

    def _on_ai_response(self, response: str, is_error: bool, input_tokens: int = 0,
                        output_tokens: int = 0, tool_calls_json: str = ""):
        """Handle response from AI API with native tool support."""
        # Remove "Thinking..." message
        cursor = self.mw.chatHistoryText.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)

        # Find and remove the last "Thinking..." message
        text = self.mw.chatHistoryText.toPlainText()
        if "Thinking..." in text:
            html = self.mw.chatHistoryText.toHtml()
            # Remove the thinking indicator line
            html = html.replace("<i style='color: #888;'>Thinking...</i>", "")
            html = html.replace("<i style=\" color:#888;\">Thinking...</i>", "")
            self.mw.chatHistoryText.setHtml(html)

        # Track token usage
        self._total_tokens_used += input_tokens + output_tokens

        if is_error:
            self.mw.chatHistoryText.append(f"<b style='color: #f48771;'>Error:</b> {response}")
        else:
            # Check for tool calls
            if tool_calls_json:
                try:
                    tool_calls = json.loads(tool_calls_json)
                    if tool_calls:
                        # Execute tool calls
                        tool_results = []
                        for tc in tool_calls:
                            tool_name = tc.get('name', '')
                            tool_input = tc.get('input', {})
                            tool_id = tc.get('id', '')

                            # Show what tool is being called
                            self.mw.chatHistoryText.append(
                                f"<i style='color: #dcdcaa;'>🔧 Calling {tool_name}...</i>"
                            )
                            self._scroll_chat_to_bottom()

                            # Execute the tool
                            result = self._execute_tool(tool_name, tool_input)
                            tool_results.append({
                                'id': tool_id,
                                'name': tool_name,
                                'result': result
                            })

                            # Show result summary
                            if 'error' in result:
                                self.mw.chatHistoryText.append(
                                    f"<i style='color: #f48771;'>✗ {tool_name}: {result.get('error', 'Unknown error')}</i>"
                                )
                            else:
                                # Show brief success message
                                if 'files' in result:
                                    count = len(result['files'])
                                    self.mw.chatHistoryText.append(
                                        f"<i style='color: #89d185;'>✓ Found {count} file(s)</i>"
                                    )
                                elif 'protocols' in result:
                                    self.mw.chatHistoryText.append(
                                        f"<i style='color: #89d185;'>✓ Found {len(result['protocols'])} protocols</i>"
                                    )
                                else:
                                    self.mw.chatHistoryText.append(
                                        f"<i style='color: #89d185;'>✓ {tool_name} completed</i>"
                                    )

                        # Send follow-up with tool results
                        self._send_native_tool_followup(response, tool_calls, tool_results)
                        return  # Don't display the partial response yet

                except json.JSONDecodeError:
                    pass  # No valid tool calls, continue with normal response

            # Add assistant response to conversation history
            self._chat_conversation_history.append({
                "role": "assistant",
                "content": response
            })

            # Display the response
            model_name = self.get_model_display_name()

            # Debug: print raw response
            if len(response) > 500:
                print(f"[AI Response Debug] First 500 chars: {response[:500]}")
            formatted = self._format_ai_response(response, strip_code=False)
            self.mw.chatHistoryText.append(f"<b style='color: #4ec9b0;'>{model_name}:</b><br>{formatted}")

            # Execute any AI commands in the response
            command_results = self._execute_ai_commands(response)
            if command_results:
                for result in command_results:
                    if result['success']:
                        self.mw.chatHistoryText.append(
                            f"<i style='color: #89d185;'>✓ {result['command']}: {result['message']}</i>"
                        )
                    else:
                        self.mw.chatHistoryText.append(
                            f"<i style='color: #f48771;'>✗ {result['command']}: {result['message']}</i>"
                        )

        self._scroll_chat_to_bottom()

        # Re-enable send button, disable stop button
        if hasattr(self.mw, 'chatSendButton'):
            self.mw.chatSendButton.setEnabled(True)
        if hasattr(self.mw, 'chatStopButton'):
            self.mw.chatStopButton.setEnabled(False)

    def _send_native_tool_followup(self, original_response: str, tool_calls: list, tool_results: list):
        """Send a follow-up request with native tool results so AI can generate final response."""
        # Build the assistant message with tool_use blocks
        assistant_content = []
        if original_response:
            assistant_content.append({"type": "text", "text": original_response})
        for tc in tool_calls:
            assistant_content.append({
                "type": "tool_use",
                "id": tc.get('id', ''),
                "name": tc.get('name', ''),
                "input": tc.get('input', {})
            })

        # Build the user message with tool_result blocks
        user_content = []
        for tr in tool_results:
            result_data = tr.get('result', {})
            is_error = 'error' in result_data
            user_content.append({
                "type": "tool_result",
                "tool_use_id": tr.get('id', ''),
                "content": json.dumps(result_data, indent=2, default=str),
                "is_error": is_error
            })

        # Add to conversation history with proper structure
        self._chat_conversation_history.append({
            "role": "assistant",
            "content": assistant_content
        })
        self._chat_conversation_history.append({
            "role": "user",
            "content": user_content
        })

        # Get API settings from the model selector
        provider, selected_model = self.get_selected_model()
        settings = QSettings("PhysioMetrics", "BreathAnalysis")
        api_key = settings.value(f"ai/{provider}_api_key", "") if provider != "ollama" else None
        tools = self._get_ai_tool_definitions() if provider != "ollama" else []

        if provider != "ollama" and not api_key:
            self.mw.chatHistoryText.append("<b style='color: #f48771;'>Error:</b> No API key configured")
            return

        # Prepare messages
        messages_for_api = list(self._chat_conversation_history)

        self._chat_worker = FollowupWorker(provider, api_key, messages_for_api, selected_model, tools)
        self._chat_worker.finished.connect(self._on_ai_response)
        self._chat_worker.start()

    def on_chat_stop(self):
        """Stop the current AI request."""
        if self._chat_worker and self._chat_worker.isRunning():
            # Terminate the worker thread
            self._chat_worker.terminate()
            self._chat_worker.wait(1000)

            # Remove "Thinking..." message
            html = self.mw.chatHistoryText.toHtml()
            html = html.replace("<i style='color: #888;'>Thinking...</i>", "")
            html = html.replace("<i style=\" color:#888;\">Thinking...</i>", "")
            self.mw.chatHistoryText.setHtml(html)

            self.mw.chatHistoryText.append("<i style='color: #888;'>Request cancelled.</i>")
            self._scroll_chat_to_bottom()

        # Re-enable send button, disable stop button
        if hasattr(self.mw, 'chatSendButton'):
            self.mw.chatSendButton.setEnabled(True)
        if hasattr(self.mw, 'chatStopButton'):
            self.mw.chatStopButton.setEnabled(False)

    def clear_chat_history(self):
        """Clear the chat history and reset token tracking."""
        if hasattr(self.mw, 'chatHistoryText'):
            self.mw.chatHistoryText.clear()
            self.mw.chatHistoryText.append(
                "<i style='color: #888;'>Chat cleared. Token usage reset.</i>"
            )

        # Reset conversation history
        self._chat_conversation_history = []
        self._total_tokens_used = 0
        self._token_warning_shown = False

        self.mw._log_status_message("Chat history cleared", 1500)

    def _scroll_chat_to_bottom(self):
        """Scroll the chat history to the bottom."""
        if hasattr(self.mw, 'chatHistoryText'):
            scrollbar = self.mw.chatHistoryText.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    # -------------------------------------------------------------------------
    # Response Formatting
    # -------------------------------------------------------------------------

    def _format_ai_response(self, response: str, strip_code: bool = False) -> str:
        """Format AI response with markdown-style formatting."""
        import re

        # Handle None response
        if not response:
            return ""

        # Convert markdown code blocks to HTML with syntax highlighting
        def format_code_block(match):
            lang = match.group(1) or ''
            code = match.group(2)

            if strip_code:
                return ""  # Strip code blocks for commands display

            # Escape HTML in code
            code = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            # Add to code notebook if it's Python code
            if lang.lower() == 'python' and hasattr(self.mw, 'codeInputEdit'):
                # Store the code for potential auto-insertion
                if not hasattr(self, '_last_generated_code'):
                    self._last_generated_code = code
                else:
                    self._last_generated_code = code

                # Auto-insert into code notebook
                self.mw.codeInputEdit.setPlainText(code)
                self.mw._log_status_message("Code inserted into notebook - click Run to execute", 3000)

            return f"<pre style='background-color: #1e1e1e; padding: 10px; border-radius: 4px; overflow-x: auto;'><code>{code}</code></pre>"

        # Replace code blocks
        formatted = re.sub(r'```(\w*)\n(.*?)```', format_code_block, response, flags=re.DOTALL)

        # Convert inline code
        formatted = re.sub(r'`([^`]+)`', r"<code style='background-color: #2d2d2d; padding: 2px 4px; border-radius: 2px;'>\1</code>", formatted)

        # Convert bold
        formatted = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', formatted)

        # Convert italic
        formatted = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', formatted)

        # Convert bullet points
        formatted = re.sub(r'^- (.+)$', r'• \1', formatted, flags=re.MULTILINE)

        # Convert newlines to <br>
        formatted = formatted.replace('\n', '<br>')

        return formatted

    # -------------------------------------------------------------------------
    # Context Building
    # -------------------------------------------------------------------------

    def _build_ai_context(self) -> str:
        """Build context string for AI about current project state."""
        context_parts = []

        # Project info
        if hasattr(self.mw, '_project_directory') and self.mw._project_directory:
            context_parts.append(f"Project directory: {self.mw._project_directory}")

        # File count
        if hasattr(self.mw, '_master_file_list') and self.mw._master_file_list:
            num_files = len(self.mw._master_file_list)
            context_parts.append(f"Data files loaded: {num_files}")

            # Sample of files with exports
            files_with_exports = [
                f for f in self.mw._master_file_list
                if f.get('export_path') or f.get('status') == 'Exported'
            ]
            if files_with_exports:
                context_parts.append(f"Files with exports: {len(files_with_exports)}")
                # Show first few
                for f in files_with_exports[:3]:
                    name = f.get('file_name', 'Unknown')
                    export = f.get('export_path', '')
                    context_parts.append(f"  - {name}: {export}")
                if len(files_with_exports) > 3:
                    context_parts.append(f"  ... and {len(files_with_exports) - 3} more")

        # Notes files
        if hasattr(self.mw, '_discovered_notes_files') and self.mw._discovered_notes_files:
            num_notes = len(self.mw._discovered_notes_files)
            context_parts.append(f"Notes files: {num_notes}")

        return '\n'.join(context_parts) if context_parts else "No project loaded"

    # -------------------------------------------------------------------------
    # AI Tool Definitions
    # -------------------------------------------------------------------------

    def _get_ai_tool_definitions(self) -> list:
        """Get the list of tool definitions for AI API."""
        return [
            {
                "name": "search_files",
                "description": "Search for files in the project by various criteria. Returns matching files with their metadata and export paths.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search term to match against filename, protocol, animal_id, or keywords"
                        },
                        "protocol": {
                            "type": "string",
                            "description": "Filter by exact protocol name"
                        },
                        "has_export": {
                            "type": "boolean",
                            "description": "If true, only return files that have been exported"
                        },
                        "status": {
                            "type": "string",
                            "description": "Filter by status (e.g., 'Exported', 'Pending', 'Analyzed')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 20)"
                        }
                    }
                }
            },
            {
                "name": "get_csv_columns",
                "description": "Get column names from an exported CSV file to understand available data.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the exported CSV file, or filename to search for"
                        },
                        "export_path": {
                            "type": "string",
                            "description": "Direct path to export directory"
                        }
                    }
                }
            },
            {
                "name": "list_protocols",
                "description": "List all unique protocol names in the current project.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "list_animals",
                "description": "List all unique animal IDs in the current project.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "project_summary",
                "description": "Get a summary of the current project including file counts, protocols, and export status.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "list_available_files",
                "description": "List all files with their export status and paths. Use this to find files for plotting.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "include_exports": {
                            "type": "boolean",
                            "description": "If true (default), include export path info"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of files to list (default: 50)"
                        }
                    }
                }
            },
            {
                "name": "get_searchable_values",
                "description": "Get all unique values for searchable fields (protocols, statuses, animal_ids, etc). CALL THIS FIRST before searching to know what values exist!",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def _execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute an AI tool and return the result."""
        try:
            if tool_name == "search_files":
                return self._ai_tool_search_files(**tool_input)
            elif tool_name == "get_csv_columns":
                return self._ai_tool_get_csv_columns(**tool_input)
            elif tool_name == "list_protocols":
                return self._ai_tool_list_protocols()
            elif tool_name == "list_animals":
                return self._ai_tool_list_animals()
            elif tool_name == "project_summary":
                return self._ai_tool_project_summary()
            elif tool_name == "list_available_files":
                return self._ai_tool_list_all_files(**tool_input)
            elif tool_name == "get_searchable_values":
                return self._ai_tool_get_searchable_values()
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e)}

    # -------------------------------------------------------------------------
    # AI Tool Implementations
    # -------------------------------------------------------------------------

    def _ai_tool_search_files(self, query: str = None, protocol: str = None,
                              has_export: bool = None, status: str = None,
                              limit: int = 20, **kwargs) -> dict:
        """Search for files matching criteria."""
        if not hasattr(self.mw, '_master_file_list') or not self.mw._master_file_list:
            return {"error": "No files loaded", "files": []}

        results = []
        for f in self.mw._master_file_list:
            # Apply filters
            if query:
                query_lower = query.lower()
                searchable = ' '.join([
                    str(f.get('file_name', '')),
                    str(f.get('protocol', '')),
                    str(f.get('animal_id', '')),
                    str(f.get('keywords_display', '')),
                    str(f.get('strain', '')),
                ]).lower()
                if query_lower not in searchable:
                    continue

            if protocol and f.get('protocol', '').lower() != protocol.lower():
                continue

            if has_export is True:
                if not f.get('export_path') and f.get('status') != 'Exported':
                    continue

            if status and f.get('status', '').lower() != status.lower():
                continue

            # Build result
            result = {
                'file_name': f.get('file_name', ''),
                'file_path': str(f.get('file_path', '')),
                'protocol': f.get('protocol', ''),
                'animal_id': f.get('animal_id', ''),
                'strain': f.get('strain', ''),
                'status': f.get('status', ''),
                'export_path': str(f.get('export_path', '')) if f.get('export_path') else None,
                'sweeps': f.get('sweeps', ''),
            }
            results.append(result)

            if len(results) >= limit:
                break

        return {
            "total_matches": len(results),
            "files": results
        }

    def _ai_tool_get_csv_columns(self, file_path: str = None, export_path: str = None) -> dict:
        """Get column names from an exported CSV file."""
        import pandas as pd

        # Find the file
        target_path = None

        if export_path:
            target_path = Path(export_path)
        elif file_path:
            # Search for matching file
            if hasattr(self.mw, '_master_file_list'):
                for f in self.mw._master_file_list:
                    if file_path.lower() in str(f.get('file_name', '')).lower():
                        if f.get('export_path'):
                            target_path = Path(f.get('export_path'))
                            break

        if not target_path:
            return {"error": "Could not find export path for file"}

        # Look for CSV files
        csv_files = {}
        if target_path.is_dir():
            for csv in target_path.glob("*.csv"):
                try:
                    df = pd.read_csv(csv, nrows=0)
                    csv_files[csv.name] = list(df.columns)
                except Exception as e:
                    csv_files[csv.name] = f"Error reading: {e}"
        elif target_path.suffix == '.csv':
            try:
                df = pd.read_csv(target_path, nrows=0)
                csv_files[target_path.name] = list(df.columns)
            except Exception as e:
                return {"error": f"Error reading CSV: {e}"}

        return {"csv_files": csv_files}

    def _ai_tool_list_protocols(self) -> dict:
        """List all unique protocols in the project."""
        if not hasattr(self.mw, '_master_file_list') or not self.mw._master_file_list:
            return {"protocols": [], "error": "No files loaded"}

        protocols = set()
        for f in self.mw._master_file_list:
            protocol = f.get('protocol', '')
            if protocol:
                protocols.add(protocol)

        return {"protocols": sorted(list(protocols))}

    def _ai_tool_list_animals(self) -> dict:
        """List all unique animal IDs in the project."""
        if not hasattr(self.mw, '_master_file_list') or not self.mw._master_file_list:
            return {"animals": [], "error": "No files loaded"}

        animals = set()
        for f in self.mw._master_file_list:
            animal = f.get('animal_id', '')
            if animal:
                animals.add(animal)

        return {"animals": sorted(list(animals))}

    def _ai_tool_project_summary(self) -> dict:
        """Get a summary of the current project."""
        if not hasattr(self.mw, '_master_file_list') or not self.mw._master_file_list:
            return {"error": "No project loaded"}

        files = self.mw._master_file_list
        total = len(files)

        # Count by status
        status_counts = {}
        for f in files:
            status = f.get('status', 'Unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

        # Count with exports
        with_exports = sum(1 for f in files if f.get('export_path') or f.get('status') == 'Exported')

        # Protocols
        protocols = set(f.get('protocol', '') for f in files if f.get('protocol'))

        # File types
        type_counts = {}
        for f in files:
            path = str(f.get('file_path', ''))
            ext = Path(path).suffix.lower() if path else 'unknown'
            type_counts[ext] = type_counts.get(ext, 0) + 1

        return {
            "total_files": total,
            "files_with_exports": with_exports,
            "status_breakdown": status_counts,
            "protocols": sorted(list(protocols)),
            "file_types": type_counts,
            "project_directory": str(self.mw._project_directory) if hasattr(self.mw, '_project_directory') else None
        }

    def _ai_tool_list_all_files(self, include_exports: bool = True, limit: int = 50) -> dict:
        """List all files with their export status."""
        if not hasattr(self.mw, '_master_file_list') or not self.mw._master_file_list:
            return {"files": [], "error": "No files loaded"}

        results = []
        for f in self.mw._master_file_list[:limit]:
            entry = {
                'file_name': f.get('file_name', ''),
                'protocol': f.get('protocol', ''),
                'status': f.get('status', ''),
                'sweeps': f.get('sweeps', ''),
            }
            if include_exports and f.get('export_path'):
                entry['export_path'] = str(f.get('export_path'))
            results.append(entry)

        return {
            "total_files": len(self.mw._master_file_list),
            "showing": len(results),
            "files": results
        }

    def _ai_tool_get_searchable_values(self) -> dict:
        """Get all unique searchable values from the project."""
        if not hasattr(self.mw, '_master_file_list') or not self.mw._master_file_list:
            return {"error": "No files loaded"}

        files = self.mw._master_file_list

        # Collect unique values
        protocols = set()
        statuses = set()
        animals = set()
        strains = set()
        experiments = set()

        for f in files:
            if f.get('protocol'):
                protocols.add(f['protocol'])
            if f.get('status'):
                statuses.add(f['status'])
            if f.get('animal_id'):
                animals.add(f['animal_id'])
            if f.get('strain'):
                strains.add(f['strain'])
            if f.get('experiment'):
                experiments.add(f['experiment'])

        # Count files with exports
        with_exports = sum(1 for f in files if f.get('export_path') or f.get('status') == 'Exported')

        return {
            "protocols": sorted(list(protocols)),
            "statuses": sorted(list(statuses)),
            "animal_ids": sorted(list(animals)),
            "strains": sorted(list(strains)),
            "experiments": sorted(list(experiments)),
            "total_files": len(files),
            "files_with_exports": with_exports
        }

    # -------------------------------------------------------------------------
    # AI Commands
    # -------------------------------------------------------------------------

    def _execute_ai_commands(self, response: str) -> list:
        """Execute AI commands embedded in the response."""
        import re

        results = []

        # Pattern: [COMMAND: args]
        command_pattern = r'\[(\w+):\s*([^\]]+)\]'
        matches = re.findall(command_pattern, response)

        for command, args in matches:
            command = command.upper()
            args = args.strip()

            try:
                if command == 'LOAD_PROJECT':
                    result = self._ai_load_project(args)
                elif command == 'UPDATE_META':
                    # Parse: filename | field=value
                    parts = args.split('|')
                    if len(parts) >= 2:
                        filename = parts[0].strip()
                        fields = parts[1].strip()
                        result = self._ai_update_metadata(filename, fields)
                    else:
                        result = {'command': command, 'success': False, 'message': 'Invalid format'}
                elif command == 'NEW_PROJECT':
                    result = self._ai_new_project(args)
                elif command == 'SAVE_PROJECT':
                    result = self._ai_save_project()
                elif command == 'RENAME_PROJECT':
                    result = self._ai_rename_project(args)
                elif command == 'SCAN_DIRECTORY':
                    result = self._ai_scan_directory(args)
                elif command == 'SCAN_SAVED_DATA':
                    result = self._ai_scan_saved_data()
                elif command == 'SET_FILE_TYPES':
                    result = self._ai_set_file_types(args)
                elif command == 'FILTER_ROWS':
                    result = self._ai_filter_rows(args)
                elif command == 'CLEAR_FILTER':
                    result = self._ai_clear_filter()
                elif command == 'SET_FILTER_COLUMN':
                    result = self._ai_set_filter_column(args)
                else:
                    result = {'command': command, 'success': False, 'message': f'Unknown command: {command}'}

                results.append(result)

            except Exception as e:
                results.append({
                    'command': command,
                    'success': False,
                    'message': str(e)
                })

        return results

    def _ai_load_project(self, project_name: str) -> dict:
        """Load a project by name."""
        try:
            # Find project file
            projects_dir = Path.home() / '.physiometrics' / 'projects'
            if not projects_dir.exists():
                return {'command': 'LOAD_PROJECT', 'success': False, 'message': 'No projects directory found'}

            # Search for matching project
            for proj_file in projects_dir.glob('*.physiometrics'):
                if project_name.lower() in proj_file.stem.lower():
                    # Load the project
                    if hasattr(self.mw, 'project_manager'):
                        project_data = self.mw.project_manager.load_project(proj_file)
                        self.mw._populate_ui_from_project(project_data)
                        return {
                            'command': 'LOAD_PROJECT',
                            'success': True,
                            'message': f"Loaded project: {proj_file.stem}"
                        }

            return {
                'command': 'LOAD_PROJECT',
                'success': False,
                'message': f"Project '{project_name}' not found"
            }

        except Exception as e:
            return {'command': 'LOAD_PROJECT', 'success': False, 'message': str(e)}

    def _ai_update_metadata(self, filename: str, fields_str: str) -> dict:
        """Update metadata for a file."""
        try:
            # Find the file
            target_idx = None
            for i, f in enumerate(self.mw._master_file_list):
                if filename.lower() in str(f.get('file_name', '')).lower():
                    target_idx = i
                    break

            if target_idx is None:
                return {
                    'command': 'UPDATE_META',
                    'success': False,
                    'message': f"File '{filename}' not found"
                }

            # Parse fields (field=value, field2=value2)
            updates = {}
            for part in fields_str.split(','):
                if '=' in part:
                    key, value = part.split('=', 1)
                    updates[key.strip()] = value.strip()

            # Update the file
            for key, value in updates.items():
                self.mw._master_file_list[target_idx][key] = value

            # Refresh table if model exists
            if hasattr(self.mw, '_file_table_model'):
                self.mw._file_table_model.layoutChanged.emit()

            return {
                'command': 'UPDATE_META',
                'success': True,
                'message': f"Updated {len(updates)} field(s) for {filename}"
            }

        except Exception as e:
            return {'command': 'UPDATE_META', 'success': False, 'message': str(e)}

    def _ai_new_project(self, project_name: str) -> dict:
        """Create a new project."""
        try:
            if hasattr(self.mw, 'on_project_new'):
                self.mw.on_project_new(project_name=project_name)
                return {
                    'command': 'NEW_PROJECT',
                    'success': True,
                    'message': f"Created project: {project_name}"
                }
            return {'command': 'NEW_PROJECT', 'success': False, 'message': 'Method not available'}
        except Exception as e:
            return {'command': 'NEW_PROJECT', 'success': False, 'message': str(e)}

    def _ai_save_project(self) -> dict:
        """Save the current project."""
        try:
            if hasattr(self.mw, 'on_project_save'):
                self.mw.on_project_save()
                return {
                    'command': 'SAVE_PROJECT',
                    'success': True,
                    'message': 'Project saved'
                }
            return {'command': 'SAVE_PROJECT', 'success': False, 'message': 'Method not available'}
        except Exception as e:
            return {'command': 'SAVE_PROJECT', 'success': False, 'message': str(e)}

    def _ai_rename_project(self, new_name: str) -> dict:
        """Rename the current project."""
        try:
            if hasattr(self.mw, 'on_project_rename'):
                self.mw.on_project_rename(new_name=new_name)
                return {
                    'command': 'RENAME_PROJECT',
                    'success': True,
                    'message': f"Renamed to: {new_name}"
                }
            return {'command': 'RENAME_PROJECT', 'success': False, 'message': 'Method not available'}
        except Exception as e:
            return {'command': 'RENAME_PROJECT', 'success': False, 'message': str(e)}

    def _ai_scan_directory(self, directory_path: str) -> dict:
        """Scan a directory for data files."""
        try:
            path = Path(directory_path)
            if not path.exists():
                return {
                    'command': 'SCAN_DIRECTORY',
                    'success': False,
                    'message': f"Directory not found: {directory_path}"
                }

            if hasattr(self.mw, 'on_project_scan_new_files'):
                self.mw.on_project_scan_new_files(scan_path=path)
                return {
                    'command': 'SCAN_DIRECTORY',
                    'success': True,
                    'message': f"Scanning: {directory_path}"
                }
            return {'command': 'SCAN_DIRECTORY', 'success': False, 'message': 'Method not available'}
        except Exception as e:
            return {'command': 'SCAN_DIRECTORY', 'success': False, 'message': str(e)}

    def _ai_scan_saved_data(self) -> dict:
        """Scan for saved/exported data files."""
        try:
            if hasattr(self.mw, 'on_project_scan_saved_data'):
                self.mw.on_project_scan_saved_data()
                return {
                    'command': 'SCAN_SAVED_DATA',
                    'success': True,
                    'message': 'Scanning for saved data...'
                }
            return {'command': 'SCAN_SAVED_DATA', 'success': False, 'message': 'Method not available'}
        except Exception as e:
            return {'command': 'SCAN_SAVED_DATA', 'success': False, 'message': str(e)}

    def _ai_set_file_types(self, types_str: str) -> dict:
        """Set which file types to scan for."""
        try:
            types = [t.strip().lower() for t in types_str.split(',')]
            valid_types = {'abf', 'smrx', 'edf', 'photometry', 'notes'}
            invalid = [t for t in types if t not in valid_types]

            if invalid:
                return {
                    'command': 'SET_FILE_TYPES',
                    'success': False,
                    'message': f"Invalid types: {invalid}. Valid: {valid_types}"
                }

            # Update checkboxes
            if hasattr(self.mw, 'scanAbfCheckbox'):
                self.mw.scanAbfCheckbox.setChecked('abf' in types)
            if hasattr(self.mw, 'scanSmrxCheckbox'):
                self.mw.scanSmrxCheckbox.setChecked('smrx' in types)
            if hasattr(self.mw, 'scanEdfCheckbox'):
                self.mw.scanEdfCheckbox.setChecked('edf' in types)
            if hasattr(self.mw, 'scanPhotometryCheckbox'):
                self.mw.scanPhotometryCheckbox.setChecked('photometry' in types)
            if hasattr(self.mw, 'scanNotesCheckbox'):
                self.mw.scanNotesCheckbox.setChecked('notes' in types)

            return {
                'command': 'SET_FILE_TYPES',
                'success': True,
                'message': f"Set file types: {types}"
            }
        except Exception as e:
            return {'command': 'SET_FILE_TYPES', 'success': False, 'message': str(e)}

    def _ai_filter_rows(self, search_text: str) -> dict:
        """Filter table rows by search text."""
        try:
            if hasattr(self.mw, 'tableFilterEdit'):
                self.mw.tableFilterEdit.setText(search_text)
                return {
                    'command': 'FILTER_ROWS',
                    'success': True,
                    'message': f"Filtering by: {search_text}"
                }
            return {'command': 'FILTER_ROWS', 'success': False, 'message': 'Filter not available'}
        except Exception as e:
            return {'command': 'FILTER_ROWS', 'success': False, 'message': str(e)}

    def _ai_clear_filter(self) -> dict:
        """Clear the table filter."""
        try:
            if hasattr(self.mw, 'tableFilterEdit'):
                self.mw.tableFilterEdit.clear()
                return {
                    'command': 'CLEAR_FILTER',
                    'success': True,
                    'message': 'Filter cleared'
                }
            return {'command': 'CLEAR_FILTER', 'success': False, 'message': 'Filter not available'}
        except Exception as e:
            return {'command': 'CLEAR_FILTER', 'success': False, 'message': str(e)}

    def _ai_set_filter_column(self, column_name: str) -> dict:
        """Set which column to filter by."""
        try:
            if hasattr(self.mw, 'filterColumnCombo'):
                # Find the column index
                combo = self.mw.filterColumnCombo
                for i in range(combo.count()):
                    if column_name.lower() in combo.itemText(i).lower():
                        combo.setCurrentIndex(i)
                        return {
                            'command': 'SET_FILTER_COLUMN',
                            'success': True,
                            'message': f"Filter column: {combo.itemText(i)}"
                        }

                # Column not found
                available = [combo.itemText(i) for i in range(combo.count())]
                return {
                    'command': 'SET_FILTER_COLUMN',
                    'success': False,
                    'message': f"Unknown column '{column_name}'. Available: {available}"
                }

        except Exception as e:
            return {
                'command': 'SET_FILTER_COLUMN',
                'success': False,
                'message': f"Error setting filter column: {str(e)}"
            }

    def _process_chat_message_local(self, message: str) -> str:
        """Process chat message locally (no API) based on keyword matching."""
        msg_lower = message.lower()

        # Get current data state
        num_data_files = len(self.mw._master_file_list) if hasattr(self.mw, '_master_file_list') and self.mw._master_file_list else 0
        notes_files = getattr(self.mw, '_discovered_notes_files', [])
        num_notes = len(notes_files) if notes_files else 0

        # Check for file-related queries
        if any(word in msg_lower for word in ['file', 'files', 'data', 'see', 'list', 'show', 'what']):
            if 'note' in msg_lower:
                # Query about notes files
                if num_notes == 0:
                    return ("I don't see any notes files yet. Please scan a directory first using "
                            "'Scan for New Files' with the 'Notes' checkbox enabled.")
                else:
                    notes_list = "<br>".join([f"• {f.name}" for f in notes_files[:10]])
                    extra = f"<br>... and {num_notes - 10} more" if num_notes > 10 else ""
                    return (f"I can see <b>{num_notes} notes file(s)</b>:<br>{notes_list}{extra}<br><br>"
                            "Once AI integration is configured (click ⚙), I'll be able to read and "
                            "extract metadata from these files!")

            elif any(word in msg_lower for word in ['data', 'abf', 'smrx', 'edf', 'file']):
                # Query about data files
                if num_data_files == 0:
                    return ("I don't see any data files yet. Please scan a directory first using "
                            "'Scan for New Files'.")
                else:
                    # Summarize by file type
                    abf_count = sum(1 for f in self.mw._master_file_list if str(f.get('file_path', '')).lower().endswith('.abf'))
                    smrx_count = sum(1 for f in self.mw._master_file_list if str(f.get('file_path', '')).lower().endswith('.smrx'))
                    edf_count = sum(1 for f in self.mw._master_file_list if str(f.get('file_path', '')).lower().endswith('.edf'))

                    # Get unique protocols
                    protocols = set(f.get('protocol', 'Unknown') for f in self.mw._master_file_list)
                    protocols_str = ", ".join(sorted(protocols)[:5])
                    if len(protocols) > 5:
                        protocols_str += f" (+{len(protocols)-5} more)"

                    # Sample file names
                    sample_files = [f.get('file_name', 'Unknown') for f in self.mw._master_file_list[:5]]
                    samples_str = "<br>".join([f"• {name}" for name in sample_files])
                    extra = f"<br>... and {num_data_files - 5} more" if num_data_files > 5 else ""

                    return (f"I can see <b>{num_data_files} data file(s)</b>:<br>"
                            f"• ABF: {abf_count}<br>• SMRX: {smrx_count}<br>• EDF: {edf_count}<br><br>"
                            f"<b>Protocols found:</b> {protocols_str}<br><br>"
                            f"<b>Sample files:</b><br>{samples_str}{extra}")

        # Check for plot requests
        if any(word in msg_lower for word in ['plot', 'graph', 'chart', 'visualize', 'draw']):
            if num_data_files == 0:
                return "I'd love to help you plot data, but no files are loaded yet. Please scan a directory first."

            # Generate sample code in the notebook
            sample_code = '''# Example: Load and plot data from first file
from pathlib import Path
import pyabf
import matplotlib.pyplot as plt

# Get the first file path
file_path = master_file_list[0]['file_path']
print(f"Loading: {file_path}")

# Load and plot (for ABF files)
if str(file_path).lower().endswith('.abf'):
    abf = pyabf.ABF(str(file_path))
    abf.setSweep(0)
    plt.figure(figsize=(10, 4))
    plt.plot(abf.sweepX, abf.sweepY)
    plt.xlabel('Time (s)')
    plt.ylabel(abf.sweepLabelY)
    plt.title(f'{Path(file_path).name}')
    plt.tight_layout()
    plt.show()
'''
            if hasattr(self.mw, 'codeInputEdit'):
                self.mw.codeInputEdit.setPlainText(sample_code)

            return ("I've generated sample plotting code in the Code Notebook below! "
                    "Click <b>▶ Run</b> to execute it.<br><br>"
                    "The code will load and plot the first file in your list. "
                    "You can modify it to plot different files or customize the visualization.")

        # Check for help queries
        if any(word in msg_lower for word in ['help', 'can you', 'what can', 'how']):
            return ("I can help you with:<br><br>"
                    "• <b>File queries</b>: 'What files do I have?', 'List notes files'<br>"
                    "• <b>Plotting</b>: 'Plot the data', 'Show me a graph'<br>"
                    "• <b>Analysis</b>: 'Analyze breathing patterns', 'Find sniffing episodes'<br><br>"
                    "Once you configure an AI API (click ⚙️), I can provide more advanced "
                    "assistance with code generation and data interpretation!")

        # Default response
        return ("I understood your message, but I'm not sure how to help with that yet.<br><br>"
                "Try asking about:<br>"
                "• Your data files<br>"
                "• Plotting options<br>"
                "• Analysis capabilities<br><br>"
                "Or click ⚙️ to set up an AI API for full assistance!")

    # -------------------------------------------------------------------------
    # AI Settings
    # -------------------------------------------------------------------------

    def open_ai_settings(self):
        """Open the AI Settings dialog for configuring AI integration."""
        try:
            from dialogs.ai_settings_dialog import AISettingsDialog

            # Get files metadata for the dialog
            files_metadata = []
            if hasattr(self.mw, '_master_file_list'):
                for task in self.mw._master_file_list:
                    metadata = {
                        'file_name': task.get('file_name', ''),
                        'protocol': task.get('protocol', ''),
                        'keywords_display': task.get('keywords_display', ''),
                        'experiment': task.get('experiment', ''),
                        'file_path': str(task.get('file_path', '')),
                    }
                    files_metadata.append(metadata)

            dialog = AISettingsDialog(self.mw, files_metadata=files_metadata)
            dialog.exec()

        except ImportError as e:
            self.mw._show_warning("AI Module Not Found",
                             f"Could not load AI settings dialog:\n{e}\n\n"
                             "Make sure the dialogs/ai_settings_dialog.py file exists.")
        except Exception as e:
            self.mw._show_error("Error", f"Failed to open AI settings:\n{e}")
