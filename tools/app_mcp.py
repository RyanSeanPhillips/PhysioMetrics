#!/usr/bin/env python
"""
App MCP Server — bridges Claude Code to the running PhysioMetrics app.

Requires the app to be running with its bridge service active.
Communicates via TCP on localhost to get app state, navigate files,
and trigger actions.

Configure in .mcp.json:
{
    "mcpServers": {
        "app": {
            "command": "C:/Users/rphil2/AppData/Local/miniforge3/envs/plethapp/python.exe",
            "args": ["-u", "tools/app_mcp.py"],
            "cwd": "C:/Users/rphil2/Dropbox/python scripts/breath_analysis/pyqt6"
        }
    }
}
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.mcp_framework import MCPServer
from core.services.app_bridge_service import bridge_call

server = MCPServer("app", "1.0.0")


def _call(command: str, args: dict = None) -> dict:
    """Call the app bridge, raising on error."""
    result = bridge_call(command, args or {})
    if result.get("status") == "error":
        raise RuntimeError(result.get("error", "Unknown bridge error"))
    return result.get("result", result)


# ============================================================
# APP TOOLS
# ============================================================


@server.tool(
    name="app_get_state",
    description="Get current app state: loaded file, channels, sweep position, sample rate. Returns 'loaded: false' if no file is open.",
)
def app_get_state(args):
    return _call("get_state")


@server.tool(
    name="app_get_selection",
    description="Get currently selected files in the project table.",
)
def app_get_selection(args):
    return _call("get_selection")


@server.tool(
    name="app_navigate",
    description="Navigate to a specific file, sweep, or time position in the app.",
    params={
        "file_path": {"type": "string", "description": "File to load (optional — omit to stay on current file)"},
        "sweep": {"type": "integer", "description": "Sweep index to navigate to"},
        "time": {"type": "number", "description": "Time position in seconds to scroll to"},
    },
)
def app_navigate(args):
    return _call("navigate", args)


@server.tool(
    name="app_ping",
    description="Check if the PhysioMetrics app is running and the bridge is active.",
)
def app_ping(args):
    return _call("ping")


@server.tool(
    name="app_list_commands",
    description="List all available bridge commands. Use this to discover what the app bridge supports.",
)
def app_list_commands(args):
    return _call("list_commands")


@server.tool(
    name="app_refresh_project",
    description="Force the app to reload the experiment table from the SQLite DB. Use after MCP tools modify experiment metadata so the table reflects latest data without a restart.",
)
def app_refresh_project(args):
    return _call("refresh_project")


@server.tool(
    name="app_reload",
    description="Trigger hot reload (Ctrl+R equivalent). Reloads dialog modules, plotting, and core processing code without restarting the app. Use after modifying non-main.py files.",
)
def app_reload(args):
    return _call("reload")


@server.tool(
    name="app_switch_tab",
    description="Switch to a named tab in the app. Use 'data_files' or 'consolidate' for sub-tabs, 'project'/'analysis'/'curation' for top-level pages.",
    params={
        "tab": {"type": "string", "description": "Tab name: 'project', 'analysis', 'curation', 'data_files', 'consolidate'"},
    },
    required=["tab"],
)
def app_switch_tab(args):
    return _call("switch_tab", args)


@server.tool(
    name="app_click_button",
    description="Click a named button in the app UI. Use the button's Qt object name (e.g. 'scanFilesButton', 'saveProjectButton').",
    params={
        "button": {"type": "string", "description": "Button object name"},
    },
    required=["button"],
)
def app_click_button(args):
    return _call("click_button", args)


@server.tool(
    name="app_open_dialog",
    description="Open a dialog by name. Available: 'peak_detection', 'analysis_options', 'help'.",
    params={
        "dialog": {"type": "string", "description": "Dialog name"},
    },
    required=["dialog"],
)
def app_open_dialog(args):
    return _call("open_dialog", args)


@server.tool(
    name="app_close_dialog",
    description="Close the active modal dialog.",
)
def app_close_dialog(args):
    return _call("close_dialog")


@server.tool(
    name="app_get_ui_state",
    description="Get current UI state: active tab, open dialogs, selected table rows.",
)
def app_get_ui_state(args):
    return _call("get_ui_state")


@server.tool(
    name="app_screenshot",
    description="Capture a screenshot of the app window. Returns a PNG path that Claude can view with the Read tool. Use this to see the current UI state for layout adjustments.",
    params={
        "target": {
            "type": "string",
            "description": "What to capture: 'main' (full window, default), 'project' (project builder tab), 'plot' (plot area)",
        },
    },
)
def app_screenshot(args):
    result = _call("screenshot", {
        "target": args.get("target", "main"),
        "output_path": args.get("output_path", ""),
    })
    return result


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    server.run()
