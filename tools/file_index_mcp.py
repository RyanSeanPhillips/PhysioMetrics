#!/usr/bin/env python
"""
File Index MCP Server — exposes the lab file index as tools for Claude Code.

Runs as a stdio MCP server. Provides 8 tools for scanning directories,
extracting notes text, and searching the local cache.
"""

import sys
import os
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.services.file_index_service import FileIndexService

# Lazy service singleton
_service = None


def get_service() -> FileIndexService:
    global _service
    if _service is None:
        _service = FileIndexService()
    return _service


# === MCP Protocol ===

TOOLS = [
    {
        "name": "fi_scan",
        "description": "Scan a directory tree, classify and index all files (data, notes, reference, video). Fast — just stat() calls, no file reads.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Root directory to scan"},
                "recursive": {"type": "boolean", "default": True, "description": "Search subdirectories"},
                "max_depth": {"type": "integer", "default": 8, "description": "Maximum directory depth"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "fi_extract",
        "description": "Extract text from notes/reference files into local cache. Incremental by default — only reads new/changed files. Use force=true to re-extract all.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "force": {"type": "boolean", "default": False, "description": "Re-extract all files even if cached"},
                "file_id": {"type": "integer", "description": "Extract specific file by ID (omit for all)"},
            },
        },
    },
    {
        "name": "fi_search",
        "description": "Full-text search across all cached notes content and file names. Returns matching cells with file/sheet/row context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (FTS5 syntax supported)"},
                "scope": {"type": "string", "enum": ["all", "files", "notes"], "default": "all",
                          "description": "Search scope: 'all', 'files' only, or 'notes' content only"},
                "limit": {"type": "integer", "default": 100},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fi_find_file",
        "description": "Find which notes file(s) mention a specific experiment file name. Key use case: given '25121003.abf', find the notes file + sheet + row.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_name": {"type": "string", "description": "File name to search for (extension stripped automatically)"},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["file_name"],
        },
    },
    {
        "name": "fi_get_files",
        "description": "List indexed files, optionally filtered by class (data/notes/reference/video/other), extension, or path substring.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_class": {"type": "string", "description": "Filter by class: data, notes, reference, video, other"},
                "extension": {"type": "string", "description": "Filter by extension (e.g. '.abf')"},
                "path_contains": {"type": "string", "description": "Filter by path substring"},
                "limit": {"type": "integer", "default": 200},
            },
        },
    },
    {
        "name": "fi_get_notes_summary",
        "description": "Get summary of a notes file: sheet names, detected headers, row/col counts. Use file_id from fi_get_files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "integer", "description": "File ID from fi_get_files results"},
            },
            "required": ["file_id"],
        },
    },
    {
        "name": "fi_get_cells",
        "description": "Get cached cell data for a specific notes file (and optionally sheet). Returns all cell values with row/col positions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "integer", "description": "File ID"},
                "sheet_name": {"type": "string", "description": "Optional: filter to specific sheet"},
                "limit": {"type": "integer", "default": 5000},
            },
            "required": ["file_id"],
        },
    },
    {
        "name": "fi_stats",
        "description": "Get file index statistics: file counts by class, extraction progress, scan roots, last scan/extract times.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def handle_tool_call(name: str, arguments: dict) -> dict:
    """Handle a single tool call."""
    svc = get_service()

    try:
        if name == "fi_scan":
            result = svc.scan_directory(
                arguments["path"],
                recursive=arguments.get("recursive", True),
                max_depth=arguments.get("max_depth", 8),
            )
            return _text(result)

        elif name == "fi_extract":
            file_ids = [arguments["file_id"]] if arguments.get("file_id") else None
            result = svc.extract_notes(
                file_ids=file_ids,
                force=arguments.get("force", False),
            )
            return _text(result)

        elif name == "fi_search":
            result = svc.search(
                arguments["query"],
                scope=arguments.get("scope", "all"),
                limit=arguments.get("limit", 100),
            )
            return _text(result)

        elif name == "fi_find_file":
            result = svc.find_file_in_notes(
                arguments["file_name"],
                limit=arguments.get("limit", 50),
            )
            return _text(result)

        elif name == "fi_get_files":
            result = svc.store.get_files(
                file_class=arguments.get("file_class"),
                extension=arguments.get("extension"),
                path_contains=arguments.get("path_contains"),
                limit=arguments.get("limit", 200),
            )
            return _text(result)

        elif name == "fi_get_notes_summary":
            result = svc.get_notes_summary(arguments["file_id"])
            return _text(result)

        elif name == "fi_get_cells":
            result = svc.store.get_cells(
                arguments["file_id"],
                sheet_name=arguments.get("sheet_name"),
                limit=arguments.get("limit", 5000),
            )
            return _text(result)

        elif name == "fi_stats":
            result = svc.get_stats()
            return _text(result)

        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}

    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}


def _text(data) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(data, indent=2, default=str)}]}


# === stdio MCP protocol ===

def read_message():
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def write_message(msg):
    print(json.dumps(msg), flush=True)


def main():
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
                            "name": "file-index",
                            "version": "1.0.0",
                        },
                    },
                })

            elif method == "notifications/initialized":
                pass

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
