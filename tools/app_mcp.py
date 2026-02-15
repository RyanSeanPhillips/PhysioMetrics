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
    description="Force the app to reload the project file from disk. Useful after MCP tools modify the .physiometrics file on a network drive.",
)
def app_refresh_project(args):
    return _call("refresh_project")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    server.run()
