"""
Claude Code Launcher Service — launches Claude CLI in an external terminal.

No Qt dependencies. Pure Python service.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class ClaudeLauncherService:
    """Launches Claude Code CLI in an external terminal window."""

    def __init__(self, project_root: Optional[str] = None):
        self._project_root = project_root or str(
            Path(__file__).parent.parent.parent
        )

    def is_available(self) -> bool:
        """Check if the 'claude' CLI is on PATH."""
        return shutil.which("claude") is not None

    def launch(self, project_dir: Optional[str] = None) -> bool:
        """Launch Claude Code in a new terminal window.

        Args:
            project_dir: Working directory for Claude. Defaults to project root.

        Returns:
            True if launched successfully, False on error.
        """
        cwd = project_dir or self._project_root
        self._ensure_mcp_config(cwd)

        try:
            # Clean environment: remove CLAUDECODE so the new session
            # doesn't think it's nested inside another Claude Code session
            env = os.environ.copy()
            env.pop("CLAUDECODE", None)

            if os.name == "nt":
                # Windows: open new cmd window with claude
                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k", "claude"],
                    cwd=cwd,
                    env=env,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            else:
                # Linux/Mac: try common terminal emulators
                for term_cmd in [
                    ["gnome-terminal", "--", "claude"],
                    ["xterm", "-e", "claude"],
                    ["open", "-a", "Terminal", "claude"],
                ]:
                    try:
                        subprocess.Popen(term_cmd, cwd=cwd, env=env)
                        break
                    except FileNotFoundError:
                        continue
                else:
                    return False
            return True
        except Exception as e:
            print(f"[claude-launcher] Error launching: {e}")
            return False

    def launch_assistant(self, workspace_dir: Optional[str] = None,
                         root_data_dir: Optional[str] = None) -> bool:
        """Launch the Metadata Extraction Assistant as a separate process.

        Args:
            workspace_dir: Working directory for Claude in the assistant.
            root_data_dir: Root data directory to show in the file tree.

        Returns:
            True if launched successfully, False on error.
        """
        import sys
        python = sys.executable

        # Ensure MCP config exists in the workspace
        ws = workspace_dir
        if not ws:
            appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
            ws = os.path.join(appdata, "PhysioMetrics", "ai_workspace")
        os.makedirs(ws, exist_ok=True)
        self._ensure_mcp_config(ws)

        try:
            env = os.environ.copy()
            env.pop("CLAUDECODE", None)

            cmd = [python, "-m", "_internal.scripts.metadata_assistant"]
            cmd.extend(["--workspace", ws])
            if root_data_dir:
                cmd.extend(["--root", root_data_dir])

            subprocess.Popen(
                cmd,
                cwd=self._project_root,
                env=env,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
            )
            return True
        except Exception as e:
            print(f"[claude-launcher] Error launching assistant: {e}")
            return False

    def _ensure_mcp_config(self, project_dir: str):
        """Write/update .mcp.json with app + project MCP server configs."""
        mcp_path = Path(project_dir) / ".mcp.json"
        python_path = self._get_python_path()

        config = {}
        if mcp_path.exists():
            try:
                with open(mcp_path, "r") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, OSError):
                config = {}

        servers = config.setdefault("mcpServers", {})

        # Ensure app bridge MCP is configured
        servers["app"] = {
            "command": python_path,
            "args": ["-u", "tools/app_mcp.py"],
            "cwd": self._project_root,
        }

        # Ensure project MCP is configured
        servers["project"] = {
            "command": python_path,
            "args": ["-u", "tools/project_mcp.py"],
            "cwd": self._project_root,
        }

        # Ensure file-index MCP is configured
        servers["file-index"] = {
            "command": python_path,
            "args": ["-u", "tools/file_index_mcp.py"],
            "cwd": self._project_root,
        }

        # Ensure assistant MCP is configured (for metadata extraction GUI)
        servers["assistant"] = {
            "command": python_path,
            "args": ["-u", "_internal/scripts/metadata_assistant/mcp/assistant_mcp.py"],
            "cwd": self._project_root,
        }

        try:
            with open(mcp_path, "w") as f:
                json.dump(config, f, indent=4)
        except OSError as e:
            print(f"[claude-launcher] Error writing .mcp.json: {e}")

    def _get_python_path(self) -> str:
        """Get the Python interpreter path for MCP servers."""
        # Use the same interpreter that's running this code
        import sys
        return sys.executable.replace("\\", "/")
