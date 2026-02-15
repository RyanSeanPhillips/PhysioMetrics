"""
Lightweight MCP server framework with decorator-based tool registration.

Usage:
    from tools.mcp_framework import MCPServer, tool

    server = MCPServer("my-server", "1.0.0")

    @server.tool(
        name="my_tool",
        description="Does something useful",
        params={"arg1": {"type": "string", "description": "First argument"}},
        required=["arg1"],
    )
    def handle_my_tool(args: dict) -> dict:
        return {"result": args["arg1"]}

    # Run the server
    server.run()

Adding a new tool is just writing a decorated function — no need to touch
the TOOLS list, handle_tool_call, or the protocol code.
"""

import sys
import json
from typing import Callable, Dict, Any, Optional, List


class MCPServer:
    """Decorator-based MCP server. Register tools with @server.tool(...)."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self._tools: Dict[str, dict] = {}  # name -> {schema, handler}

    def tool(
        self,
        name: str,
        description: str,
        params: Optional[Dict[str, Any]] = None,
        required: Optional[List[str]] = None,
    ):
        """
        Decorator to register an MCP tool.

        Args:
            name: Tool name (e.g., "project_open").
            description: Human-readable description.
            params: Dict of param_name -> JSON Schema property definition.
            required: List of required parameter names.

        The decorated function receives (args: dict) and should return
        a result dict or raise an exception.
        """
        def decorator(fn: Callable):
            schema = {
                "type": "object",
                "properties": params or {},
            }
            if required:
                schema["required"] = required

            self._tools[name] = {
                "name": name,
                "description": description,
                "inputSchema": schema,
                "handler": fn,
            }
            return fn
        return decorator

    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        params: Optional[Dict[str, Any]] = None,
        required: Optional[List[str]] = None,
    ):
        """Imperative alternative to the @tool decorator."""
        schema = {
            "type": "object",
            "properties": params or {},
        }
        if required:
            schema["required"] = required

        self._tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": schema,
            "handler": handler,
        }

    def get_tool_list(self) -> List[dict]:
        """Return tool definitions for tools/list response."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            }
            for t in self._tools.values()
        ]

    def handle_call(self, name: str, arguments: dict) -> dict:
        """Dispatch a tool call to its handler."""
        tool = self._tools.get(name)
        if tool is None:
            return _error_result(f"Unknown tool: {name}")

        try:
            result = tool["handler"](arguments)
            return _text_result(result)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return _error_result(str(e))

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def run(self):
        """Main MCP server loop — reads/writes newline-delimited JSON-RPC on stdio."""
        while True:
            msg_id = None
            try:
                msg = _read_message()
                if msg is None:
                    break

                method = msg.get("method")
                msg_id = msg.get("id")
                params = msg.get("params", {})

                if method == "initialize":
                    _write_message({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {
                                "name": self.name,
                                "version": self.version,
                            },
                        },
                    })

                elif method == "notifications/initialized":
                    pass

                elif method == "tools/list":
                    _write_message({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {"tools": self.get_tool_list()},
                    })

                elif method == "tools/call":
                    tool_name = params.get("name")
                    tool_args = params.get("arguments", {})
                    result = self.handle_call(tool_name, tool_args)
                    _write_message({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": result,
                    })

                elif msg_id is not None:
                    _write_message({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32601, "message": f"Unknown method: {method}"},
                    })

            except Exception as e:
                sys.stderr.write(f"MCP Error: {e}\n")
                sys.stderr.flush()
                if msg_id is not None:
                    _write_message({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32603, "message": str(e)},
                    })


# --- Protocol helpers ---

def _read_message():
    """Read a newline-delimited JSON-RPC message from stdin."""
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def _write_message(msg):
    """Write a newline-delimited JSON-RPC message to stdout."""
    print(json.dumps(msg), flush=True)


def _text_result(data) -> dict:
    """Create a successful text result."""
    return {"content": [{"type": "text", "text": json.dumps(data, indent=2, default=str)}]}


def _error_result(msg: str) -> dict:
    """Create an error result."""
    return {"content": [{"type": "text", "text": f"Error: {msg}"}], "isError": True}
