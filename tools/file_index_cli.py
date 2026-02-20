#!/usr/bin/env python
"""
File Index CLI — scan lab directories, extract notes text, search locally.

Usage:
    python -u tools/file_index_cli.py scan PATH [--no-recursive] [--max-depth N]
    python -u tools/file_index_cli.py extract [--force] [--file-id ID]
    python -u tools/file_index_cli.py search QUERY [--scope all|files|notes]
    python -u tools/file_index_cli.py find-file FILENAME
    python -u tools/file_index_cli.py stats
    python -u tools/file_index_cli.py files [--class CLASS] [--ext EXT]
    python -u tools/file_index_cli.py summary FILE_ID
"""

import sys
import os
import json
import argparse
import time

# Unbuffered output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.services.file_index_service import FileIndexService


def get_service() -> FileIndexService:
    return FileIndexService()


def cmd_scan(args):
    svc = get_service()
    print(f"Scanning: {args.path}")
    t0 = time.time()

    def progress(n):
        print(f"  ...indexed {n} files", end="\r")

    result = svc.scan_directory(
        args.path,
        recursive=not args.no_recursive,
        max_depth=args.max_depth,
        progress_callback=progress,
    )
    elapsed = time.time() - t0

    print(f"\nScan complete in {elapsed:.1f}s")
    print(f"  Added:   {result.get('added', 0)}")
    print(f"  Updated: {result.get('updated', 0)}")
    print(f"  Skipped: {result.get('skipped', 0)}")
    print(f"  Deleted: {result.get('deleted', 0)}")
    if result.get("by_class"):
        print(f"  By class: {json.dumps(result['by_class'])}")
    svc.close()


def cmd_extract(args):
    svc = get_service()
    file_ids = [args.file_id] if args.file_id else None
    print("Extracting notes text...")
    t0 = time.time()

    def progress(current, total, name):
        print(f"  [{current}/{total}] {name}")

    result = svc.extract_notes(
        file_ids=file_ids,
        force=args.force,
        progress_callback=progress,
    )
    elapsed = time.time() - t0

    print(f"\nExtraction complete in {elapsed:.1f}s")
    print(f"  Extracted: {result['extracted']} files")
    print(f"  Skipped:   {result['skipped']} files")
    print(f"  Cells:     {result['total_cells']}")
    if result["errors"]:
        print(f"  Errors:    {len(result['errors'])}")
        for err in result["errors"]:
            print(f"    {err['file']}: {err['error']}")
    svc.close()


def cmd_search(args):
    svc = get_service()
    result = svc.search(args.query, scope=args.scope, limit=args.limit)

    if result.get("files"):
        print(f"\n=== Files ({len(result['files'])}) ===")
        for f in result["files"]:
            print(f"  [{f['file_class']}] {f['file_path']}")

    if result.get("notes"):
        print(f"\n=== Notes cells ({len(result['notes'])}) ===")
        for n in result["notes"]:
            loc = f"{n['file_name']}"
            if n["sheet_name"]:
                loc += f" / {n['sheet_name']}"
            loc += f" (row {n['row_num']}, col {n['col_num']})"
            snippet = n.get("snippet", n["value"])
            print(f"  {loc}: {snippet}")
    svc.close()


def cmd_find_file(args):
    svc = get_service()
    matches = svc.find_file_in_notes(args.filename)

    if not matches:
        print(f"No notes found mentioning '{args.filename}'")
    else:
        print(f"Found in {len(matches)} location(s):")
        for m in matches:
            sheet_info = f" / sheet '{m['sheet']}'" if m["sheet"] else ""
            print(f"\n  {m['source_name']}{sheet_info}")
            for hit in m["matches"]:
                print(f"    Row {hit['row']}, Col {hit['col']}: {hit['value']}")
    svc.close()


def cmd_stats(args):
    svc = get_service()
    stats = svc.get_stats()
    print("File Index Statistics")
    print("=" * 40)
    print(f"  Total files:     {stats.get('total_files', 0)}")
    if stats.get("files_by_class"):
        for cls, cnt in sorted(stats["files_by_class"].items()):
            print(f"    {cls:12s}: {cnt}")
    print(f"  Files extracted: {stats.get('files_extracted', 0)} / {stats.get('files_extractable', 0)}")
    print(f"  Total cells:     {stats.get('total_cells', 0)}")
    print(f"  Last scan:       {stats.get('last_scan', 'never')}")
    print(f"  Last extract:    {stats.get('last_extract', 'never')}")
    roots = stats.get("scan_roots", "")
    if roots:
        print(f"  Scan roots:")
        for r in roots.split("|"):
            if r:
                print(f"    {r}")
    svc.close()


def cmd_files(args):
    svc = get_service()
    files = svc.store.get_files(
        file_class=args.file_class,
        extension=args.ext,
        limit=args.limit,
    )
    print(f"Files ({len(files)}):")
    for f in files:
        size_kb = f["file_size"] / 1024
        print(f"  [{f['file_class']:9s}] {f['file_name']:40s} ({size_kb:8.1f} KB) {f['parent_dir']}")
    svc.close()


def cmd_summary(args):
    svc = get_service()
    summary = svc.get_notes_summary(args.file_id)
    if "error" in summary:
        print(f"Error: {summary['error']}")
    else:
        print(f"File: {summary['file_name']}")
        print(f"Path: {summary['file_path']}")
        for s in summary["sheets"]:
            print(f"\n  Sheet: {s['name']}")
            print(f"    Rows: {s['rows']}, Cols: {s['cols']}, Cells: {s['cell_count']}")
            if s["headers"]:
                print(f"    Header row {s['header_row']}: {' | '.join(s['headers'])}")
    svc.close()


def main():
    parser = argparse.ArgumentParser(description="File Index CLI")
    sub = parser.add_subparsers(dest="command")

    # scan
    p = sub.add_parser("scan", help="Scan directory tree")
    p.add_argument("path", help="Root directory to scan")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--max-depth", type=int, default=8)

    # extract
    p = sub.add_parser("extract", help="Extract notes text")
    p.add_argument("--force", action="store_true", help="Re-extract all files")
    p.add_argument("--file-id", type=int, help="Extract specific file by ID")

    # search
    p = sub.add_parser("search", help="Full-text search")
    p.add_argument("query", help="Search query")
    p.add_argument("--scope", default="all", choices=["all", "files", "notes"])
    p.add_argument("--limit", type=int, default=50)

    # find-file
    p = sub.add_parser("find-file", help="Find notes mentioning a file")
    p.add_argument("filename", help="File name to search for")

    # stats
    sub.add_parser("stats", help="Index statistics")

    # files
    p = sub.add_parser("files", help="List indexed files")
    p.add_argument("--class", dest="file_class", help="Filter by class")
    p.add_argument("--ext", help="Filter by extension")
    p.add_argument("--limit", type=int, default=100)

    # summary
    p = sub.add_parser("summary", help="Notes file summary")
    p.add_argument("file_id", type=int, help="File ID")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmds = {
        "scan": cmd_scan,
        "extract": cmd_extract,
        "search": cmd_search,
        "find-file": cmd_find_file,
        "stats": cmd_stats,
        "files": cmd_files,
        "summary": cmd_summary,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
