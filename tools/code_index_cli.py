#!/usr/bin/env python
"""
Code Index CLI — rebuild, query, and run diagnostics.

Usage:
    python tools/code_index_cli.py rebuild [--no-analysis]
    python tools/code_index_cli.py incremental
    python tools/code_index_cli.py diagnostics [--severity=error|warning|info] [--rule=RULE_ID]
    python tools/code_index_cli.py stats
    python tools/code_index_cli.py query find_function --name NAME [--class CLASS] [--file FILE]
    python tools/code_index_cli.py query find_class --name NAME [--base BASE]
    python tools/code_index_cli.py query get_callers --name NAME
    python tools/code_index_cli.py query get_signals [--name NAME]
    python tools/code_index_cli.py query get_state_fields
    python tools/code_index_cli.py query search --query QUERY
    python tools/code_index_cli.py query file_summary --path PATH
"""

import sys
import os
import json
import argparse
import time

# Unbuffered output for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from core.adapters.code_index_sqlite import CodeIndexSQLite
from core.services.code_index_service import CodeIndexService


DB_PATH = Path(PROJECT_ROOT) / "_internal" / "code_index.db"


def get_service() -> CodeIndexService:
    db = CodeIndexSQLite(DB_PATH)
    return CodeIndexService(db, Path(PROJECT_ROOT))


def cmd_rebuild(args):
    print(f"Rebuilding code index at {DB_PATH}...")
    t0 = time.time()
    svc = get_service()
    stats = svc.full_rebuild(run_analysis=not args.no_analysis)
    elapsed = time.time() - t0

    print(f"\nIndex rebuilt in {elapsed:.2f}s")
    print(f"  Files:       {stats.total_files}")
    print(f"  Classes:     {stats.total_classes}")
    print(f"  Functions:   {stats.total_functions}")
    print(f"  Imports:     {stats.total_imports}")
    print(f"  Signals:     {stats.total_signals}")
    print(f"  Connections: {stats.total_connections}")
    print(f"  Calls:       {stats.total_calls}")
    print(f"  UI Widgets:  {stats.total_ui_widgets}")
    print(f"  Parse errors:{stats.parse_errors}")

    if not args.no_analysis:
        print(f"\nDiagnostics:")
        print(f"  Errors:   {stats.errors}")
        print(f"  Warnings: {stats.warnings}")
        print(f"  Info:     {stats.info}")

    svc.db.close()


def cmd_incremental(args):
    print("Running incremental update...")
    svc = get_service()
    result = svc.incremental_update()
    print(f"  Added:   {result['added']}")
    print(f"  Changed: {result['changed']}")
    print(f"  Removed: {result['removed']}")
    svc.db.close()


def cmd_diagnostics(args):
    svc = get_service()
    diags = svc.get_diagnostics(
        severity=args.severity,
        rule_id=args.rule,
        limit=args.limit or 100,
    )

    if not diags:
        print("No diagnostics found.")
        svc.db.close()
        return

    # Group by severity
    for sev in ['error', 'warning', 'info']:
        group = [d for d in diags if d.severity == sev]
        if not group:
            continue
        print(f"\n{'=' * 60}")
        print(f" {sev.upper()} ({len(group)})")
        print(f"{'=' * 60}")
        for d in group:
            loc = f"{d.context}:{d.line_no}" if d.line_no else d.context or "?"
            print(f"  [{d.rule_id}] {loc}")
            print(f"    {d.message}")

    svc.db.close()


def cmd_stats(args):
    svc = get_service()
    stats = svc.get_stats()
    print(json.dumps(stats.to_dict(), indent=2))

    # Last rebuild info
    last = svc.get_knowledge("last_rebuild")
    if last:
        print(f"\nLast rebuild: {last.get('timestamp', '?')} ({last.get('elapsed_seconds', '?')}s)")

    svc.db.close()


def cmd_query(args):
    svc = get_service()

    if args.query_type == 'find_function':
        results = svc.find_function(name=args.name, class_name=args.cls, file_pattern=args.file)
        _print_results(results, "Functions")

    elif args.query_type == 'find_class':
        results = svc.find_class(name=args.name, base_class=args.base)
        _print_results(results, "Classes")

    elif args.query_type == 'get_callers':
        results = svc.get_callers(args.name)
        _print_results(results, "Callers")

    elif args.query_type == 'get_signals':
        results = svc.get_signals(name=args.name)
        _print_results(results, "Signals")

    elif args.query_type == 'get_state_fields':
        results = svc.get_state_fields()
        if results:
            for field_info in results:
                accesses = field_info['accesses']
                files = set(a['file'] for a in accesses)
                print(f"  {field_info['field']} ({len(accesses)} accesses in {len(files)} files)")
        else:
            print("No state field accesses found.")

    elif args.query_type == 'search':
        results = svc.search_code(args.query_text)
        _print_results(results, "Search Results")

    elif args.query_type == 'file_summary':
        result = svc.get_file_summary(args.path)
        if result:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"File not found: {args.path}")

    svc.db.close()


def _print_results(results, label):
    if not results:
        print(f"No {label.lower()} found.")
        return
    print(f"\n{label} ({len(results)}):")
    print(json.dumps(results, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="Code Index CLI")
    sub = parser.add_subparsers(dest="command")

    # rebuild
    p_rebuild = sub.add_parser("rebuild", help="Full rebuild of the code index")
    p_rebuild.add_argument("--no-analysis", action="store_true", help="Skip static analysis")

    # incremental
    sub.add_parser("incremental", help="Incremental update (only changed files)")

    # diagnostics
    p_diag = sub.add_parser("diagnostics", help="Show static analysis diagnostics")
    p_diag.add_argument("--severity", choices=["error", "warning", "info"])
    p_diag.add_argument("--rule", help="Filter by rule ID (e.g., UNDEF_STATE_FIELD)")
    p_diag.add_argument("--limit", type=int, default=100)

    # stats
    sub.add_parser("stats", help="Show index statistics")

    # query
    p_query = sub.add_parser("query", help="Query the index")
    q_sub = p_query.add_subparsers(dest="query_type")

    p_ff = q_sub.add_parser("find_function")
    p_ff.add_argument("--name", required=True)
    p_ff.add_argument("--class", dest="cls")
    p_ff.add_argument("--file")

    p_fc = q_sub.add_parser("find_class")
    p_fc.add_argument("--name", required=True)
    p_fc.add_argument("--base")

    p_gc = q_sub.add_parser("get_callers")
    p_gc.add_argument("--name", required=True)

    p_gs = q_sub.add_parser("get_signals")
    p_gs.add_argument("--name")

    q_sub.add_parser("get_state_fields")

    p_search = q_sub.add_parser("search")
    p_search.add_argument("--query", dest="query_text", required=True)

    p_fs = q_sub.add_parser("file_summary")
    p_fs.add_argument("--path", required=True)

    args = parser.parse_args()

    if args.command == 'rebuild':
        cmd_rebuild(args)
    elif args.command == 'incremental':
        cmd_incremental(args)
    elif args.command == 'diagnostics':
        cmd_diagnostics(args)
    elif args.command == 'stats':
        cmd_stats(args)
    elif args.command == 'query':
        if not args.query_type:
            p_query.print_help()
        else:
            cmd_query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
