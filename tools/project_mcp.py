#!/usr/bin/env python
"""
Project MCP Server — exposes project metadata tools for Claude Code.

Runs as a stdio MCP server. Works WITHOUT the app running — operates
directly on .physiometrics JSON files and recording files on disk.

v4: SQLite primary store. Dropped 9 redundant tools (discover_notes,
read_notes, match_notes, cache_pattern, get_patterns, cache_vocabulary,
check_cache, apply_patterns, label_file). Added project_add_custom_column.

Configure in .mcp.json:
{
    "mcpServers": {
        "project": {
            "command": "C:/Users/rphil2/AppData/Local/miniforge3/envs/plethapp/python.exe",
            "args": ["-u", "tools/project_mcp.py"],
            "cwd": "C:/Users/rphil2/Dropbox/python scripts/breath_analysis/pyqt6"
        }
    }
}
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from tools.mcp_framework import MCPServer
from core.services.project_service import ProjectService

# === Server + Service ===

server = MCPServer("project", "4.0.0")

_service = None


def svc() -> ProjectService:
    global _service
    if _service is None:
        _service = ProjectService()
    return _service


def _resolve(path_str: str) -> str:
    """Resolve a relative-or-absolute path against the project data directory."""
    return svc()._to_absolute(path_str)


# ============================================================
# TOOLS (16 total)
# ============================================================


@server.tool(
    name="project_open",
    description="Open or create a .physiometrics project for a folder. If a project file exists, loads it. Otherwise creates a new empty project.",
    params={"folder": {"type": "string", "description": "Path to the project folder"}},
    required=["folder"],
)
def project_open(args):
    return svc().open_project(Path(args["folder"]))


@server.tool(
    name="project_scan",
    description="Discover recording files (ABF, SMRX, EDF, photometry) in the project folder tree. Adds them to the project.",
    params={
        "folder": {"type": "string", "description": "Folder to scan (defaults to project folder)"},
        "recursive": {"type": "boolean", "default": True, "description": "Search subdirectories"},
        "file_types": {
            "type": "array",
            "items": {"type": "string", "enum": ["abf", "smrx", "edf", "photometry"]},
            "description": "File types to scan for (default: all)",
        },
        "load_metadata": {"type": "boolean", "default": True, "description": "Also load protocol/channel metadata"},
    },
)
def project_scan(args):
    folder = args.get("folder")
    new_files = svc().scan_folder(
        root=Path(folder) if folder else None,
        recursive=args.get("recursive", True),
        file_types=args.get("file_types"),
    )
    count = svc().add_files(new_files)
    if args.get("load_metadata", True) and new_files:
        svc().load_file_metadata(file_entries=new_files)
    return {
        "new_files_found": len(new_files),
        "files_added": count,
        "total_files": svc().get_file_count(),
        "sample_files": [
            {"file_name": f["file_name"], "file_type": f.get("file_type", ""), "protocol": f.get("protocol", "")}
            for f in new_files[:10]
        ],
    }


@server.tool(
    name="project_get_files",
    description="List all files in the project with their current metadata. Returns file_path, file_name, experiment, strain, stim_type, power, sex, animal_id, protocol, channel, status, etc. Includes custom column values.",
    params={
        "filter_field": {"type": "string", "description": "Optional field to filter by"},
        "filter_value": {"type": "string", "description": "Value to filter for"},
        "order_by": {"type": "string", "description": "Field to sort by (e.g. 'animal_id', 'experiment')"},
        "offset": {"type": "integer", "default": 0, "description": "Skip first N files (for pagination)"},
        "limit": {"type": "integer", "default": 100, "description": "Max files to return"},
    },
)
def project_get_files(args):
    s = svc()
    files = s.get_project_files()

    filter_field = args.get("filter_field")
    filter_value = args.get("filter_value")
    if filter_field and filter_value:
        files = [f for f in files if str(f.get(filter_field, "")).lower() == filter_value.lower()]

    offset = args.get("offset", 0)
    limit = args.get("limit", 100)
    total = len(files)
    files = files[offset:offset + limit]

    KEEP_KEYS = {
        "file_path", "file_name", "experiment", "strain", "stim_type", "power",
        "sex", "animal_id", "protocol", "channel", "stim_channel", "events_channel",
        "status", "channel_count", "sweep_count", "file_type", "keywords_display", "linked_notes",
    }
    simplified = []
    for f in files:
        entry = {k: v for k, v in f.items() if (k in KEEP_KEYS or k not in (
            'file_id', 'project_id', 'updated_at', 'field_timestamps',
            'path_keywords', 'stim_channels', 'stim_frequency', 'exports',
        )) and v}
        if "file_path" in entry:
            entry["file_path"] = s._to_relative(entry["file_path"])
        subrow_count = f.get("subrow_count")
        if subrow_count:
            entry["subrow_count"] = subrow_count
        simplified.append(entry)

    return {"total": total, "offset": offset, "limit": limit, "files": simplified}


@server.tool(
    name="project_get_files_grouped",
    description="List files grouped by folder with summary stats. Much more context-efficient than flat file list — use this for overview.",
)
def project_get_files_grouped(args):
    return svc().get_files_grouped()


@server.tool(
    name="project_update_file",
    description="Update metadata for a single recording file. Accepts ANY field — standard fields update directly, unknown fields become custom columns. No restriction on editable fields.",
    params={
        "file_path": {"type": "string", "description": "Path to the file to update (relative or absolute)"},
        "updates": {"type": "object", "description": "Dict of field->value pairs to update", "additionalProperties": True},
        "provenance": {"type": "object", "description": "Optional provenance info: {source_type, source_detail, reason, confidence}", "additionalProperties": True},
    },
    required=["file_path", "updates"],
)
def project_update_file(args):
    result = svc().update_file_metadata(
        _resolve(args["file_path"]), args["updates"], provenance=args.get("provenance"),
    )
    if result:
        return {"status": "updated", "file": result.get("file_name", "")}
    raise FileNotFoundError(f"File not found: {args['file_path']}")


@server.tool(
    name="project_batch_update",
    description="Apply the same metadata updates to multiple files at once. Accepts ANY field.",
    params={
        "file_paths": {"type": "array", "items": {"type": "string"}, "description": "List of file paths to update (relative or absolute)"},
        "updates": {"type": "object", "description": "Dict of field->value pairs to apply to all files", "additionalProperties": True},
        "provenance": {"type": "object", "description": "Optional provenance info", "additionalProperties": True},
    },
    required=["file_paths", "updates"],
)
def project_batch_update(args):
    provenance = args.get("provenance")
    count = svc().batch_update(
        [_resolve(fp) for fp in args["file_paths"]],
        args["updates"],
        provenance=provenance,
    )
    return {"files_updated": count, "total_requested": len(args["file_paths"])}


@server.tool(name="project_completeness", description="Get percentage of metadata filled for each column. Shows which fields need attention.")
def project_completeness(args):
    return svc().get_metadata_completeness()


@server.tool(
    name="project_get_unique",
    description="Get sorted unique values for a metadata field. Useful for autocomplete, validation, and detecting patterns.",
    params={"field": {"type": "string", "description": "Field name (e.g., 'strain', 'experiment', 'stim_type')"}},
    required=["field"],
)
def project_get_unique(args):
    values = svc().get_unique_values(args["field"])
    return {"field": args["field"], "unique_values": values, "count": len(values)}


@server.tool(
    name="project_save",
    description="Persist current project state to disk (.physiometrics file). DB is always up-to-date; this exports JSON to network drive.",
    params={"name": {"type": "string", "description": "Optional project name (uses existing name if not provided)"}},
)
def project_save(args):
    return {"saved_to": svc().save_project(name=args.get("name"))}


@server.tool(
    name="project_inspect_channels",
    description="Analyze channels in a recording file — returns FFT-based classification (pleth/stim/noise/empty), dominant frequency, periodicity, SNR, and digital detection for each channel.",
    params={
        "file_path": {"type": "string", "description": "Path to recording file (.abf, .smrx, .edf)"},
        "sample_seconds": {"type": "number", "default": 5.0, "description": "How many seconds to analyze"},
    },
    required=["file_path"],
)
def project_inspect_channels(args):
    from core.services.channel_classifier import classify_channels
    return classify_channels(Path(_resolve(args["file_path"])), sample_seconds=args.get("sample_seconds", 5.0))


@server.tool(
    name="project_classify_channels",
    description="Auto-classify all channels in a file and update the project metadata with channel assignments.",
    params={
        "file_path": {"type": "string", "description": "Path to recording file"},
        "auto_assign": {"type": "boolean", "default": True, "description": "Automatically assign channels in project metadata"},
    },
    required=["file_path"],
)
def project_classify_channels(args):
    from core.services.channel_classifier import classify_channels
    file_path = _resolve(args["file_path"])
    result = classify_channels(Path(file_path))

    if args.get("auto_assign", True) and "channels" in result:
        updates = {}
        for ch in result["channels"]:
            cls = ch.get("classification")
            conf = ch.get("confidence", 0)
            ch_name = ch.get("name", "")
            if cls == "pleth" and conf > 0.6:
                updates["channel"] = ch_name
            elif cls == "stim" and conf > 0.6:
                if not updates.get("stim_channel"):
                    updates["stim_channel"] = ch_name
                else:
                    updates["events_channel"] = ch_name
        if updates:
            svc().update_file_metadata(file_path, updates)
            result["auto_assigned"] = updates

    return result


@server.tool(
    name="project_preview_file",
    description="Generate a multi-channel thumbnail PNG that Claude can view with the Read tool. Shows all channels with classification labels.",
    params={
        "file_path": {"type": "string", "description": "Path to recording file"},
        "output_path": {"type": "string", "description": "Where to save PNG (auto-generates if not specified)"},
    },
    required=["file_path"],
)
def project_preview_file(args):
    from core.services.channel_classifier import generate_thumbnail
    output = args.get("output_path")
    thumb_path = generate_thumbnail(
        Path(_resolve(args["file_path"])),
        output_path=Path(output) if output else None,
    )
    if thumb_path:
        return {"thumbnail_path": thumb_path}
    raise RuntimeError("Failed to generate thumbnail")


@server.tool(
    name="project_add_subrow",
    description="Create a child entry for a multi-animal recording file. Subrows inherit protocol/stim_type/power from parent.",
    params={
        "file_path": {"type": "string", "description": "Path to the parent recording file"},
        "channel": {"type": "string", "description": "Channel for this animal (e.g. 'IN 0', 'AD0')"},
        "animal_id": {"type": "string", "description": "Animal ID for this subrow"},
        "sex": {"type": "string", "description": "Sex (M/F)"},
        "group": {"type": "string", "description": "Experimental group (e.g. 'GFP', 'ChR2')"},
    },
    required=["file_path", "channel"],
)
def project_add_subrow(args):
    subrow = svc().add_subrow(
        file_path=_resolve(args["file_path"]), channel=args["channel"],
        animal_id=args.get("animal_id", ""), sex=args.get("sex", ""), group=args.get("group", ""),
    )
    if subrow:
        return {"status": "created", "subrow": subrow}
    raise FileNotFoundError(f"File not found: {args['file_path']}")


@server.tool(
    name="project_get_subrows",
    description="Get all subrows (child entries) for a multi-animal recording file.",
    params={"file_path": {"type": "string", "description": "Path to the recording file"}},
    required=["file_path"],
)
def project_get_subrows(args):
    subrows = svc().get_subrows(_resolve(args["file_path"]))
    return {"subrows": subrows, "count": len(subrows)}


@server.tool(
    name="project_get_provenance",
    description="Get provenance records showing how metadata values were determined. Shows source type (notes/folder/user/merge), confidence, and correction chains.",
    params={
        "file_path": {"type": "string", "description": "Path to the recording file"},
        "field": {"type": "string", "description": "Optional: filter by specific field"},
    },
    required=["file_path"],
)
def project_get_provenance(args):
    provenance = svc().get_provenance(args["file_path"], field=args.get("field"))
    return {"provenance": provenance, "count": len(provenance)}


@server.tool(
    name="project_add_custom_column",
    description="Define a new custom metadata column for the project. Custom columns appear alongside standard fields and persist across saves.",
    params={
        "column_key": {"type": "string", "description": "Internal key (e.g. 'virus', 'ear_tag', 'fiber_coords')"},
        "display_name": {"type": "string", "description": "Human-readable name (e.g. 'Virus Type')"},
        "column_type": {"type": "string", "default": "text", "description": "Column type: text, number, boolean"},
        "sort_order": {"type": "integer", "default": 0, "description": "Display order (lower = earlier)"},
    },
    required=["column_key", "display_name"],
)
def project_add_custom_column(args):
    col_id = svc().add_custom_column(
        column_key=args["column_key"],
        display_name=args["display_name"],
        column_type=args.get("column_type", "text"),
        sort_order=args.get("sort_order", 0),
    )
    if col_id:
        return {"status": "created", "column_id": col_id, "column_key": args["column_key"]}
    raise RuntimeError("No project open — call project_open first")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    server.run()
