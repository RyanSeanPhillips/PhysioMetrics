#!/usr/bin/env python
"""
Batch Analysis MCP Server — headless peak detection and metrics export.

Runs as a stdio MCP server. Works WITHOUT the app running — loads files,
runs the analysis pipeline, writes results CSV.

Configure in .mcp.json:
{
    "mcpServers": {
        "batch": {
            "command": "C:/Users/rphil2/AppData/Local/miniforge3/envs/plethapp/python.exe",
            "args": ["-u", "tools/batch_mcp.py"],
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

import json
from pathlib import Path
from tools.mcp_framework import MCPServer

server = MCPServer("batch", "1.0.0")


@server.tool(
    name="batch_analyze_file",
    description=(
        "Headlessly analyze a single recording file (ABF, SMRX, EDF, MAT). "
        "Runs peak detection + per-breath metrics and writes a results CSV. "
        "Returns summary with output path."
    ),
    params={
        "file_path": {"type": "string", "description": "Path to recording file"},
        "output_dir": {
            "type": "string",
            "description": "Output directory (default: same as input file)",
        },
        "config_json": {
            "type": "string",
            "description": (
                "Optional JSON string with AnalysisConfig overrides. "
                "E.g. '{\"filter\": {\"use_zscore\": false}}'"
            ),
        },
    },
    required=["file_path"],
)
def batch_analyze_file(file_path: str, output_dir: str = None, config_json: str = None):
    from core.services.analysis_service import analyze_file
    from core.domain.analysis.models import AnalysisConfig

    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    config = AnalysisConfig()
    if config_json:
        try:
            overrides = json.loads(config_json)
            config = AnalysisConfig.from_dict(overrides)
        except (json.JSONDecodeError, Exception) as e:
            return {"error": f"Invalid config_json: {e}"}

    out = Path(output_dir) if output_dir else None
    result = analyze_file(path, config, out)

    return result.to_dict()


@server.tool(
    name="batch_analyze_folder",
    description=(
        "Batch-analyze all matching recording files in a folder. "
        "Writes per-file results CSVs and returns a summary."
    ),
    params={
        "folder": {"type": "string", "description": "Folder containing recording files"},
        "pattern": {
            "type": "string",
            "description": "Glob pattern (default: '*.abf')",
        },
        "output_dir": {
            "type": "string",
            "description": "Output directory (default: same as input folder)",
        },
        "config_json": {
            "type": "string",
            "description": "Optional JSON string with AnalysisConfig overrides",
        },
    },
    required=["folder"],
)
def batch_analyze_folder(
    folder: str, pattern: str = "*.abf", output_dir: str = None, config_json: str = None
):
    from core.services.analysis_service import analyze_folder
    from core.domain.analysis.models import AnalysisConfig

    folder_path = Path(folder)
    if not folder_path.is_dir():
        return {"error": f"Not a directory: {folder}"}

    config = AnalysisConfig()
    if config_json:
        try:
            overrides = json.loads(config_json)
            config = AnalysisConfig.from_dict(overrides)
        except (json.JSONDecodeError, Exception) as e:
            return {"error": f"Invalid config_json: {e}"}

    out = Path(output_dir) if output_dir else None

    results = analyze_folder(folder_path, config, pattern, out)

    summary = {
        "total_files": len(results),
        "succeeded": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "results": [r.to_dict() for r in results],
    }
    return summary


# ── CLI entry point ──────────────────────────────────────────────

def main():
    """Run as MCP server or CLI tool."""
    if len(sys.argv) > 1 and sys.argv[1] != "--stdio":
        # CLI mode: batch_mcp.py <file_or_folder> [--pattern *.abf] [--output-dir dir]
        import argparse

        parser = argparse.ArgumentParser(description="Batch analysis CLI")
        parser.add_argument("path", help="File or folder to analyze")
        parser.add_argument("--pattern", default="*.abf", help="File glob pattern")
        parser.add_argument("--output-dir", default=None, help="Output directory")
        parser.add_argument("--config", default=None, help="JSON config file path")
        args = parser.parse_args()

        from core.domain.analysis.models import AnalysisConfig

        config = AnalysisConfig()
        if args.config:
            with open(args.config) as f:
                config = AnalysisConfig.from_dict(json.load(f))

        target = Path(args.path)
        out = Path(args.output_dir) if args.output_dir else None

        if target.is_file():
            from core.services.analysis_service import analyze_file

            def _log(msg):
                print(f"  {msg}")

            result = analyze_file(target, config, out, _log)
            print(json.dumps(result.to_dict(), indent=2, default=str))
        elif target.is_dir():
            from core.services.analysis_service import analyze_folder

            def _progress(cur, total, msg):
                print(f"  [{cur}/{total}] {msg}")

            results = analyze_folder(target, config, args.pattern, out, _progress)
            summary = {
                "total": len(results),
                "succeeded": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
            }
            print(json.dumps(summary, indent=2))
            for r in results:
                status = "OK" if r.success else f"FAIL: {r.error}"
                print(f"  {r.file_path.name}: {status}")
        else:
            print(f"Error: {args.path} not found")
            sys.exit(1)
    else:
        # MCP server mode
        server.run()


if __name__ == "__main__":
    main()
