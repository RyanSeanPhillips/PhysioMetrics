#!/usr/bin/env python
"""
Code Index MCP Server — exposes the code index as tools for Claude Code.

Runs as a stdio MCP server. Configure in .claude/settings.local.json:
{
    "mcpServers": {
        "code-index": {
            "command": "python",
            "args": ["tools/code_index_mcp.py"]
        }
    }
}
"""

import sys
import os
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from core.adapters.code_index_sqlite import CodeIndexSQLite
from core.services.code_index_service import CodeIndexService

DB_PATH = Path(PROJECT_ROOT) / "_internal" / "code_index.db"

# Lazy service singleton
_service = None


def get_service() -> CodeIndexService:
    global _service
    if _service is None:
        db = CodeIndexSQLite(DB_PATH)
        _service = CodeIndexService(db, Path(PROJECT_ROOT))
    return _service


# === MCP Protocol ===

TOOLS = [
    {
        "name": "index_rebuild",
        "description": "Full or incremental rebuild of the code index. Use 'full' for complete rebuild, 'incremental' to only re-index changed files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["full", "incremental"], "default": "incremental",
                         "description": "Rebuild mode: 'full' or 'incremental'"},
            },
        },
    },
    {
        "name": "find_function",
        "description": "Find functions/methods by name, class, or file pattern. Returns file path, line numbers, params, docstring.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Function name (substring match)"},
                "class_name": {"type": "string", "description": "Filter by class name"},
                "file_pattern": {"type": "string", "description": "Filter by file path pattern"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "find_class",
        "description": "Find class definitions by name or base class.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Class name (substring match)"},
                "base_class": {"type": "string", "description": "Filter by base class name"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "get_callers",
        "description": "Find all call sites that reference a function by name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "function_name": {"type": "string", "description": "Function name to search for"},
                "limit": {"type": "integer", "default": 30},
            },
            "required": ["function_name"],
        },
    },
    {
        "name": "get_callees",
        "description": "Find all functions called by a specific function (by func_id).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "func_id": {"type": "integer", "description": "Function ID from find_function results"},
            },
            "required": ["func_id"],
        },
    },
    {
        "name": "get_diagnostics",
        "description": "Get static analysis diagnostics (errors, warnings, info). Filter by severity, rule, or file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["error", "warning", "info"]},
                "rule_id": {"type": "string", "description": "Rule ID like UNDEF_STATE_FIELD, LARGE_METHOD, etc."},
                "file_pattern": {"type": "string", "description": "Filter by file path pattern"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "get_signals",
        "description": "Find pyqtSignal declarations and their .connect() connections.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Signal name (substring match)"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "get_state_fields",
        "description": "List all AppState field accesses grouped by field name. Shows which files and functions access each field.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_widget_refs",
        "description": "Find all references to a UI widget by name. Shows .ui definition and code references.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "widget_name": {"type": "string", "description": "Widget name (e.g., 'btn_detect')"},
            },
            "required": ["widget_name"],
        },
    },
    {
        "name": "search_code",
        "description": "Full-text search across function names, class names, and docstrings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_file_summary",
        "description": "Get a structured summary of a file: classes, functions, imports, diagnostics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rel_path": {"type": "string", "description": "Relative path (e.g., 'core/state.py')"},
            },
            "required": ["rel_path"],
        },
    },
    {
        "name": "cache_knowledge",
        "description": "Store or retrieve persistent knowledge across Claude Code sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "set", "delete"], "description": "Action to perform"},
                "key": {"type": "string", "description": "Knowledge key"},
                "value": {"description": "Value to store (for 'set' action)"},
            },
            "required": ["action", "key"],
        },
    },
]


def handle_tool_call(name: str, arguments: dict) -> dict:
    """Handle a single tool call."""
    svc = get_service()

    try:
        if name == "index_rebuild":
            mode = arguments.get("mode", "incremental")
            if mode == "full":
                stats = svc.full_rebuild()
                return {"content": [{"type": "text", "text": json.dumps(stats.to_dict(), indent=2)}]}
            else:
                result = svc.incremental_update()
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        elif name == "find_function":
            results = svc.find_function(
                name=arguments.get("name"),
                class_name=arguments.get("class_name"),
                file_pattern=arguments.get("file_pattern"),
                limit=arguments.get("limit", 20),
            )
            return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}

        elif name == "find_class":
            results = svc.find_class(
                name=arguments.get("name"),
                base_class=arguments.get("base_class"),
                limit=arguments.get("limit", 20),
            )
            return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}

        elif name == "get_callers":
            results = svc.get_callers(
                arguments["function_name"],
                limit=arguments.get("limit", 30),
            )
            return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}

        elif name == "get_callees":
            results = svc.get_callees(arguments["func_id"])
            return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}

        elif name == "get_diagnostics":
            diags = svc.get_diagnostics(
                severity=arguments.get("severity"),
                rule_id=arguments.get("rule_id"),
                file_pattern=arguments.get("file_pattern"),
                limit=arguments.get("limit", 50),
            )
            results = [d.to_dict() for d in diags]
            return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}

        elif name == "get_signals":
            results = svc.get_signals(
                name=arguments.get("name"),
                limit=arguments.get("limit", 20),
            )
            return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}

        elif name == "get_state_fields":
            results = svc.get_state_fields()
            return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}

        elif name == "get_widget_refs":
            results = svc.get_widget_refs(arguments["widget_name"])
            return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}

        elif name == "search_code":
            results = svc.search_code(
                arguments["query"],
                limit=arguments.get("limit", 20),
            )
            return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}

        elif name == "get_file_summary":
            result = svc.get_file_summary(arguments["rel_path"])
            if result:
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]}
            else:
                return {"content": [{"type": "text", "text": f"File not found: {arguments['rel_path']}"}], "isError": True}

        elif name == "cache_knowledge":
            action = arguments["action"]
            key = arguments["key"]
            if action == "get":
                value = svc.get_knowledge(key)
                return {"content": [{"type": "text", "text": json.dumps({"key": key, "value": value}, default=str)}]}
            elif action == "set":
                svc.cache_knowledge(key, arguments.get("value"))
                return {"content": [{"type": "text", "text": f"Stored: {key}"}]}
            elif action == "delete":
                svc.db.delete_knowledge(key)
                return {"content": [{"type": "text", "text": f"Deleted: {key}"}]}

        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}

    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}


# === stdio MCP protocol ===

def read_message():
    """Read a newline-delimited JSON-RPC message from stdin."""
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def write_message(msg):
    """Write a newline-delimited JSON-RPC message to stdout."""
    print(json.dumps(msg), flush=True)


def main():
    """Main MCP server loop."""
    while True:
        msg_id = None
        try:
            msg = read_message()
            if msg is None:
                break

            method = msg.get("method")
            msg_id = msg.get("id")
            params = msg.get("params", {})

            if method == "initialize":
                write_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "code-index",
                            "version": "1.0.0",
                        },
                    },
                })

            elif method == "notifications/initialized":
                pass  # No response needed

            elif method == "tools/list":
                write_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"tools": TOOLS},
                })

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                result = handle_tool_call(tool_name, tool_args)
                write_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": result,
                })

            elif msg_id is not None:
                write_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                })

        except Exception as e:
            sys.stderr.write(f"MCP Error: {e}\n")
            sys.stderr.flush()
            if msg_id is not None:
                write_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32603, "message": str(e)},
                })


if __name__ == "__main__":
    main()
