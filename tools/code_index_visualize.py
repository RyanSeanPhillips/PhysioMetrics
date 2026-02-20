#!/usr/bin/env python
"""
Code Index Visualizer — generate graphs that reveal codebase structure.

Usage:
    python tools/code_index_visualize.py all              — Generate all overview graphs
    python tools/code_index_visualize.py coupling          — MainWindow coupling web
    python tools/code_index_visualize.py imports            — File import dependency graph
    python tools/code_index_visualize.py signals            — Signal/slot flow graph
    python tools/code_index_visualize.py state              — AppState field access heatmap
    python tools/code_index_visualize.py complexity         — File size vs coupling scatter
    python tools/code_index_visualize.py impact FUNC_NAME   — Call graph around a function
    python tools/code_index_visualize.py module MODULE_PATH — Dependencies of a module/file

Options:
    --no-show   Don't open the image after generating
"""

import sqlite3
import sys
import os
import json
import subprocess
from pathlib import Path
from collections import defaultdict

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')  # Non-interactive: we open the file separately
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np

DB_PATH = PROJECT_ROOT / "_internal" / "code_index.db"
VIZ_DIR = PROJECT_ROOT / "_internal" / "viz"

# Module color map
MODULE_COLORS = {
    'main.py': '#e74c3c',
    'core/': '#3498db',
    'dialogs/': '#2ecc71',
    'plotting/': '#9b59b6',
    'editing/': '#e67e22',
    'export/': '#1abc9c',
    'consolidation/': '#f39c12',
    'viewmodels/': '#00bcd4',
    'views/': '#8bc34a',
}


def get_module_color(rel_path):
    if rel_path == 'main.py':
        return MODULE_COLORS['main.py']
    for prefix, color in MODULE_COLORS.items():
        if rel_path.startswith(prefix):
            return color
    return '#95a5a6'


def get_module_group(rel_path):
    if rel_path == 'main.py':
        return 'main.py'
    parts = rel_path.split('/')
    if len(parts) > 1:
        return parts[0] + '/'
    return 'root'


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def save_and_print(fig, name):
    """Save figure and print path for the caller."""
    path = VIZ_DIR / name
    fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(str(path))
    return path


# ---------------------------------------------------------------------------
# Graph: Coupling Web
# ---------------------------------------------------------------------------

def graph_coupling(conn):
    """MainWindow coupling web — who depends on self.mw and how much."""
    rows = conn.execute("""
        SELECT fi.rel_path, COUNT(*) as ref_count
        FROM ci_attribute_access a
        JOIN ci_files fi ON a.file_id = fi.file_id
        WHERE a.target = 'self.mw'
        AND fi.rel_path != 'main.py'
        GROUP BY fi.file_id
        ORDER BY ref_count DESC
    """).fetchall()

    if not rows:
        print("No self.mw references found.")
        return None

    fig, ax = plt.subplots(figsize=(14, 14))
    G = nx.Graph()
    G.add_node('MainWindow\n(main.py)', node_type='center')

    main_info = conn.execute(
        "SELECT line_count FROM ci_files WHERE rel_path = 'main.py'"
    ).fetchone()
    main_lines = main_info['line_count'] if main_info else 9000

    for row in rows:
        rp = row['rel_path']
        count = row['ref_count']
        label = rp.replace('.py', '').replace('/', '\n')
        G.add_node(label, node_type='manager', ref_count=count, rel_path=rp)
        G.add_edge('MainWindow\n(main.py)', label, weight=count)

    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42, weight='weight')

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + (w / max_w) * 8 for w in edge_weights]
    edge_colors = [plt.cm.Reds(0.3 + 0.7 * w / max_w) for w in edge_weights]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color=edge_colors, alpha=0.6)

    for node in G.nodes():
        x, y = pos[node]
        ndata = G.nodes[node]
        if ndata.get('node_type') == 'center':
            ax.scatter(x, y, s=3000, c='#e74c3c', zorder=5,
                       edgecolors='#c0392b', linewidths=2)
            ax.text(x, y, f'MainWindow\n({main_lines} lines)', ha='center',
                    va='center', fontsize=8, fontweight='bold', color='white',
                    zorder=6)
        else:
            count = ndata.get('ref_count', 1)
            color = get_module_color(ndata.get('rel_path', ''))
            ax.scatter(x, y, s=200 + count * 8, c=color, zorder=5,
                       edgecolors='#2c3e50', linewidths=1)
            ax.text(x, y - 0.06, node, ha='center', va='top', fontsize=6, zorder=6)
            ax.text(x, y + 0.02, str(count), ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color='white', zorder=6)

    total_refs = sum(r['ref_count'] for r in rows)
    ax.set_title('MainWindow Coupling Web\n(edge thickness = self.mw reference count)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    ax.text(0.02, 0.02,
            f'{len(rows)} managers | {total_refs} total self.mw refs\n'
            f'Numbers show reference count per file',
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    return save_and_print(fig, 'coupling_web.png')


# ---------------------------------------------------------------------------
# Graph: Import Dependencies
# ---------------------------------------------------------------------------

def graph_imports(conn):
    """File-level import dependency graph."""
    files = {}
    for row in conn.execute("SELECT file_id, rel_path, line_count FROM ci_files").fetchall():
        files[row['file_id']] = {'rel_path': row['rel_path'],
                                  'line_count': row['line_count']}

    path_to_module = {}
    for fid, info in files.items():
        rp = info['rel_path']
        mod = rp.replace('/', '.').replace('\\', '.').removesuffix('.py')
        path_to_module[mod] = rp
        parts = mod.split('.')
        if len(parts) > 1:
            path_to_module[parts[-1]] = rp

    rows = conn.execute("""
        SELECT fi.rel_path as importer, i.module
        FROM ci_imports i
        JOIN ci_files fi ON i.file_id = fi.file_id
    """).fetchall()

    G = nx.DiGraph()
    edge_count = defaultdict(int)

    for row in rows:
        importer = row['importer']
        module = row['module']
        target = path_to_module.get(module)
        if not target:
            for mod_name, rp in path_to_module.items():
                if module.endswith(mod_name) or mod_name.endswith(module):
                    target = rp
                    break
        if target and target != importer:
            edge_count[(importer, target)] += 1

    connected_files = set()
    for (src, dst) in edge_count:
        connected_files.add(src)
        connected_files.add(dst)

    for f in connected_files:
        G.add_node(f)
    for (src, dst), count in edge_count.items():
        G.add_edge(src, dst, weight=count)

    if not G.nodes():
        print("No internal imports found.")
        return None

    fig, ax = plt.subplots(figsize=(20, 20))
    groups = defaultdict(list)
    for node in G.nodes():
        groups[get_module_group(node)].append(node)

    pos = nx.spring_layout(G, k=1.8, iterations=100, seed=42)

    node_sizes = []
    node_colors = []
    for node in G.nodes():
        finfo = next((v for v in files.values() if v['rel_path'] == node), None)
        lc = finfo['line_count'] if finfo else 100
        node_sizes.append(max(50, min(lc / 2, 2000)))
        node_colors.append(get_module_color(node))

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, arrows=True,
                           arrowsize=6, edge_color='#7f8c8d',
                           connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors='#2c3e50',
                           linewidths=0.5, alpha=0.85)

    labels = {}
    for node in G.nodes():
        finfo = next((v for v in files.values() if v['rel_path'] == node), None)
        lc = finfo['line_count'] if finfo else 0
        if lc > 300 or G.in_degree(node) > 5 or G.out_degree(node) > 8:
            labels[node] = node.replace('.py', '').split('/')[-1]
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6)

    ax.set_title('File Import Dependency Graph\n(node size = line count, color = module)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    legend_handles = []
    for prefix, color in MODULE_COLORS.items():
        count = len(groups.get(prefix, []))
        if count > 0:
            legend_handles.append(mpatches.Patch(color=color,
                                                  label=f'{prefix} ({count} files)'))
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8, framealpha=0.9)

    fig.tight_layout()
    return save_and_print(fig, 'import_graph.png')


# ---------------------------------------------------------------------------
# Graph: Signal Flow
# ---------------------------------------------------------------------------

def graph_signals(conn):
    """Signal/slot connection flow between classes."""
    rows = conn.execute("""
        SELECT co.signal_expr, co.slot_expr, co.line_no, fi.rel_path
        FROM ci_connections co
        JOIN ci_files fi ON co.file_id = fi.file_id
    """).fetchall()

    edge_data = defaultdict(list)
    for row in rows:
        src_file = row['rel_path']
        slot = row['slot_expr']
        if 'self.mw.' in slot:
            dst_file = 'main.py'
        else:
            dst_file = src_file
        if src_file != dst_file:
            edge_data[(src_file, dst_file)].append(
                f"{row['signal_expr']} -> {slot}")

    signal_rows = conn.execute("""
        SELECT s.name, fi.rel_path FROM ci_signals s
        JOIN ci_files fi ON s.file_id = fi.file_id
    """).fetchall()

    signals_per_file = defaultdict(int)
    for row in signal_rows:
        signals_per_file[row['rel_path']] += 1

    conns_per_file = defaultdict(int)
    for row in rows:
        conns_per_file[row['rel_path']] += 1

    relevant_files = set()
    for f, c in signals_per_file.items():
        if c > 0:
            relevant_files.add(f)
    for f, c in conns_per_file.items():
        if c >= 3:
            relevant_files.add(f)

    G = nx.DiGraph()
    for f in relevant_files:
        G.add_node(f, signals=signals_per_file.get(f, 0),
                   connections=conns_per_file.get(f, 0))
    for (src, dst), conns in edge_data.items():
        if src in relevant_files and dst in relevant_files:
            G.add_edge(src, dst, weight=len(conns))

    if not G.nodes():
        print("No signal connections found.")
        return None

    fig, ax = plt.subplots(figsize=(16, 16))
    pos = nx.spring_layout(G, k=2.0, iterations=80, seed=42)

    node_sizes = [max(100, (G.nodes[n].get('signals', 0) +
                             G.nodes[n].get('connections', 0)) * 15) for n in G.nodes()]
    node_colors = [get_module_color(n) for n in G.nodes()]

    if G.edges():
        ew = [G[u][v]['weight'] for u, v in G.edges()]
        mx = max(ew)
        nx.draw_networkx_edges(G, pos, ax=ax, width=[1 + (w / mx) * 5 for w in ew],
                               alpha=0.4, arrows=True, arrowsize=10,
                               edge_color='#e74c3c', connectionstyle='arc3,rad=0.1')

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors='#2c3e50',
                           linewidths=1, alpha=0.85)

    labels = {}
    for n in G.nodes():
        name = n.replace('.py', '').split('/')[-1]
        nd = G.nodes[n]
        labels[n] = f"{name}\n({nd.get('signals', 0)}s/{nd.get('connections', 0)}c)"
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6)

    cross_file = sum(len(c) for c in edge_data.values())
    ax.set_title('Signal/Slot Flow Graph\n(node size = signals+connections, '
                 'red edges = cross-file)', fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    ax.text(0.02, 0.02,
            f'{sum(signals_per_file.values())} signals | '
            f'{sum(conns_per_file.values())} .connect() calls | '
            f'{cross_file} cross-file',
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    return save_and_print(fig, 'signal_flow.png')


# ---------------------------------------------------------------------------
# Graph: State Heatmap
# ---------------------------------------------------------------------------

def graph_state(conn):
    """Heatmap: which files access which AppState fields."""
    rows = conn.execute("""
        SELECT fi.rel_path, a.attr_name, COUNT(*) as count
        FROM ci_attribute_access a
        JOIN ci_files fi ON a.file_id = fi.file_id
        WHERE a.target = 'self.state'
        GROUP BY fi.rel_path, a.attr_name
    """).fetchall()

    if not rows:
        print("No self.state accesses found.")
        return None

    files_set, fields_set = set(), set()
    access_map = defaultdict(lambda: defaultdict(int))
    for row in rows:
        files_set.add(row['rel_path'])
        fields_set.add(row['attr_name'])
        access_map[row['rel_path']][row['attr_name']] = row['count']

    file_totals = {f: sum(access_map[f].values()) for f in files_set}
    field_breadth = defaultdict(int)
    for f in files_set:
        for attr in access_map[f]:
            field_breadth[attr] += 1

    files_sorted = sorted(files_set, key=lambda f: file_totals[f], reverse=True)[:25]
    fields_sorted = sorted(fields_set, key=lambda a: field_breadth[a], reverse=True)[:40]

    matrix = np.zeros((len(files_sorted), len(fields_sorted)))
    for i, f in enumerate(files_sorted):
        for j, attr in enumerate(fields_sorted):
            matrix[i, j] = access_map[f].get(attr, 0)

    matrix_log = np.log1p(matrix)

    fig, ax = plt.subplots(figsize=(max(16, len(fields_sorted) * 0.45),
                                     max(8, len(files_sorted) * 0.4)))

    cmap = LinearSegmentedColormap.from_list('custom',
        ['#ffffff', '#fee8e0', '#fc9272', '#de2d26', '#67000d'])
    im = ax.imshow(matrix_log, cmap=cmap, aspect='auto', interpolation='nearest')

    file_labels = [f.replace('.py', '').replace('/', '/\n') if len(f) > 25
                   else f.replace('.py', '') for f in files_sorted]
    ax.set_yticks(range(len(files_sorted)))
    ax.set_yticklabels(file_labels, fontsize=7)
    ax.set_xticks(range(len(fields_sorted)))
    ax.set_xticklabels(fields_sorted, fontsize=6, rotation=90)

    for i in range(len(files_sorted)):
        for j in range(len(fields_sorted)):
            val = int(matrix[i, j])
            if val > 0:
                color = 'white' if matrix_log[i, j] > 2.0 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=5, color=color)

    ax.set_title('AppState Field Access Heatmap\n'
                 '(rows = files, columns = state fields, values = access count)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.text(0.02, -0.02,
            f'AppState: {len(fields_set)} fields accessed across {len(files_set)} files',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    fig.colorbar(im, ax=ax, label='log(1 + access count)', shrink=0.6)
    fig.tight_layout()
    return save_and_print(fig, 'state_heatmap.png')


# ---------------------------------------------------------------------------
# Graph: Complexity Scatter
# ---------------------------------------------------------------------------

def graph_complexity(conn):
    """Scatter plot: file size vs coupling — identifies god-objects."""
    file_stats = {}
    for row in conn.execute("SELECT file_id, rel_path, line_count FROM ci_files").fetchall():
        file_stats[row['file_id']] = {
            'rel_path': row['rel_path'], 'lines': row['line_count'],
            'functions': 0, 'imports_out': 0, 'connections': 0,
            'state_accesses': 0, 'mw_accesses': 0, 'max_complexity': 0,
        }

    for row in conn.execute("SELECT file_id, COUNT(*) as c FROM ci_functions GROUP BY file_id").fetchall():
        if row['file_id'] in file_stats: file_stats[row['file_id']]['functions'] = row['c']
    for row in conn.execute("SELECT file_id, COUNT(*) as c FROM ci_imports GROUP BY file_id").fetchall():
        if row['file_id'] in file_stats: file_stats[row['file_id']]['imports_out'] = row['c']
    for row in conn.execute("SELECT file_id, COUNT(*) as c FROM ci_connections GROUP BY file_id").fetchall():
        if row['file_id'] in file_stats: file_stats[row['file_id']]['connections'] = row['c']
    for row in conn.execute("SELECT file_id, COUNT(*) as c FROM ci_attribute_access WHERE target='self.state' GROUP BY file_id").fetchall():
        if row['file_id'] in file_stats: file_stats[row['file_id']]['state_accesses'] = row['c']
    for row in conn.execute("SELECT file_id, COUNT(*) as c FROM ci_attribute_access WHERE target='self.mw' GROUP BY file_id").fetchall():
        if row['file_id'] in file_stats: file_stats[row['file_id']]['mw_accesses'] = row['c']
    for row in conn.execute("SELECT file_id, MAX(complexity) as mx FROM ci_functions GROUP BY file_id").fetchall():
        if row['file_id'] in file_stats: file_stats[row['file_id']]['max_complexity'] = row['mx'] or 0

    data = [d for d in file_stats.values() if d['lines'] > 10]
    lines = [d['lines'] for d in data]
    coupling = [d['imports_out'] + d['connections'] + d['state_accesses'] + d['mw_accesses'] for d in data]
    functions = [d['functions'] for d in data]
    colors = [get_module_color(d['rel_path']) for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    ax1 = axes[0]
    ax1.scatter(lines, coupling, s=[max(20, f * 2) for f in functions],
                c=colors, alpha=0.7, edgecolors='#2c3e50', linewidths=0.5)
    for d, x, y in zip(data, lines, coupling):
        if x > 500 or y > 50:
            ax1.annotate(d['rel_path'].replace('.py', '').split('/')[-1],
                         (x, y), fontsize=7, ha='left', xytext=(5, 5),
                         textcoords='offset points')
    ax1.set_xlabel('Lines of Code', fontsize=11)
    ax1.set_ylabel('Total Coupling\n(imports + connections + state + mw refs)', fontsize=11)
    ax1.set_title('File Size vs Coupling\n(bubble size = function count)',
                  fontsize=13, fontweight='bold')
    ax1.set_xscale('log'); ax1.set_yscale('log'); ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='#e74c3c', linestyle='--', alpha=0.3)
    ax1.axvline(x=500, color='#e74c3c', linestyle='--', alpha=0.3)
    ax1.text(600, 55, 'God-object zone', fontsize=8, color='#e74c3c', alpha=0.6)

    ax2 = axes[1]
    max_cx = [d['max_complexity'] for d in data]
    ax2.scatter(lines, max_cx, s=[max(20, f * 2) for f in functions],
                c=colors, alpha=0.7, edgecolors='#2c3e50', linewidths=0.5)
    for d, x, y in zip(data, lines, max_cx):
        if y > 20 or x > 1000:
            ax2.annotate(d['rel_path'].replace('.py', '').split('/')[-1],
                         (x, y), fontsize=7, ha='left', xytext=(5, 5),
                         textcoords='offset points')
    ax2.set_xlabel('Lines of Code', fontsize=11)
    ax2.set_ylabel('Max Cyclomatic Complexity', fontsize=11)
    ax2.set_title('File Size vs Max Complexity\n(bubble size = function count)',
                  fontsize=13, fontweight='bold')
    ax2.set_xscale('log'); ax2.grid(True, alpha=0.3)
    ax2.axhline(y=15, color='#e67e22', linestyle='--', alpha=0.3)
    ax2.axhline(y=30, color='#e74c3c', linestyle='--', alpha=0.3)

    legend_handles = [mpatches.Patch(color=c, label=p) for p, c in MODULE_COLORS.items()]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(MODULE_COLORS),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return save_and_print(fig, 'complexity_scatter.png')


# ---------------------------------------------------------------------------
# Graph: Impact — call graph around a specific function
# ---------------------------------------------------------------------------

def graph_impact(conn, func_name):
    """Show callers and callees of a specific function as a radial graph."""
    # Find the function
    func_rows = conn.execute("""
        SELECT f.func_id, f.name, f.line_start, f.line_end, f.complexity,
               c.name as class_name, fi.rel_path
        FROM ci_functions f
        LEFT JOIN ci_classes c ON f.class_id = c.class_id
        JOIN ci_files fi ON f.file_id = fi.file_id
        WHERE f.name = ?
    """, (func_name,)).fetchall()

    if not func_rows:
        print(f"Function '{func_name}' not found.")
        return None

    # Get callers
    callers = conn.execute("""
        SELECT ca.callee_expr, ca.line_no, f2.name as caller_func,
               c2.name as caller_class, fi.rel_path
        FROM ci_calls ca
        JOIN ci_files fi ON ca.file_id = fi.file_id
        LEFT JOIN ci_functions f2 ON ca.func_id = f2.func_id
        LEFT JOIN ci_classes c2 ON f2.class_id = c2.class_id
        WHERE ca.callee_expr LIKE '%' || ? || '%'
    """, (func_name,)).fetchall()

    # Get callees (what does this function call?)
    func_ids = [r['func_id'] for r in func_rows]
    callees = []
    for fid in func_ids:
        rows = conn.execute("""
            SELECT ca.callee_expr, ca.line_no, fi.rel_path
            FROM ci_calls ca
            JOIN ci_files fi ON ca.file_id = fi.file_id
            WHERE ca.func_id = ?
        """, (fid,)).fetchall()
        callees.extend(rows)

    # Get signal connections TO this function
    slot_conns = conn.execute("""
        SELECT co.signal_expr, co.slot_expr, co.line_no, fi.rel_path
        FROM ci_connections co
        JOIN ci_files fi ON co.file_id = fi.file_id
        WHERE co.slot_expr LIKE '%' || ? || '%'
    """, (func_name,)).fetchall()

    # Build graph
    G = nx.DiGraph()
    center_label = func_name
    for r in func_rows:
        if r['class_name']:
            center_label = f"{r['class_name']}.{func_name}"
            break

    G.add_node(center_label, node_type='target', color='#e74c3c')

    # Add callers
    caller_nodes = set()
    for r in callers:
        if r['caller_func']:
            label = f"{r['caller_class']}.{r['caller_func']}" if r['caller_class'] else r['caller_func']
        else:
            label = r['rel_path'].replace('.py', '')
        if label != center_label:
            G.add_node(label, node_type='caller',
                       color=get_module_color(r['rel_path']), file=r['rel_path'])
            G.add_edge(label, center_label, edge_type='calls')
            caller_nodes.add(label)

    # Add signal connections
    for r in slot_conns:
        label = r['signal_expr']
        if label not in G.nodes():
            G.add_node(label, node_type='signal', color='#9b59b6',
                       file=r['rel_path'])
        G.add_edge(label, center_label, edge_type='signal')

    # Add callees (deduplicated by name)
    callee_names = set()
    for r in callees:
        expr = r['callee_expr']
        name = expr.rsplit('.', 1)[-1] if '.' in expr else expr
        if name == func_name or name in callee_names:
            continue
        if name.startswith('_') and len(name) > 1 and name[1] == '_':
            continue  # Skip dunders
        callee_names.add(name)
        short = expr if len(expr) < 30 else '...' + expr[-25:]
        G.add_node(short, node_type='callee',
                   color=get_module_color(r['rel_path']), file=r['rel_path'])
        G.add_edge(center_label, short, edge_type='calls')

    if len(G.nodes()) <= 1:
        print(f"No callers or callees found for '{func_name}'.")
        return None

    fig, ax = plt.subplots(figsize=(16, 12))

    # Radial layout: center node in middle
    pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)

    # Color and size nodes by type
    for node in G.nodes():
        x, y = pos[node]
        nd = G.nodes[node]
        nt = nd.get('node_type', 'other')

        if nt == 'target':
            size, color, fs = 2000, '#e74c3c', 9
        elif nt == 'caller':
            size, color, fs = 600, nd.get('color', '#3498db'), 7
        elif nt == 'signal':
            size, color, fs = 400, '#9b59b6', 6
        else:  # callee
            size, color, fs = 400, nd.get('color', '#2ecc71'), 6

        ax.scatter(x, y, s=size, c=color, zorder=5, edgecolors='#2c3e50',
                   linewidths=1, alpha=0.85)
        ax.text(x, y - 0.05, node, ha='center', va='top', fontsize=fs,
                zorder=6, fontweight='bold' if nt == 'target' else 'normal')

    # Draw edges
    for u, v, d in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        color = '#9b59b6' if d.get('edge_type') == 'signal' else '#7f8c8d'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.5, alpha=0.5))

    func_info = func_rows[0]
    lines = func_info['line_end'] - func_info['line_start']
    cx = func_info['complexity']

    ax.set_title(f'Impact Graph: {center_label}\n'
                 f'({func_info["rel_path"]}:{func_info["line_start"]}, '
                 f'{lines} lines, complexity {cx})',
                 fontsize=13, fontweight='bold', pad=20)
    ax.axis('off')

    legend_handles = [
        mpatches.Patch(color='#e74c3c', label=f'Target: {center_label}'),
        mpatches.Patch(color='#3498db', label=f'Callers ({len(caller_nodes)})'),
        mpatches.Patch(color='#9b59b6', label=f'Signal connections ({len(slot_conns)})'),
        mpatches.Patch(color='#2ecc71', label=f'Callees ({len(callee_names)})'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    return save_and_print(fig, f'impact_{func_name}.png')


# ---------------------------------------------------------------------------
# Graph: Module — focused view of one module's dependencies
# ---------------------------------------------------------------------------

def graph_module(conn, module_path):
    """Show imports in/out and coupling for a specific file or module."""
    # Find matching files
    if module_path.endswith('/'):
        pattern = module_path + '%'
    elif not module_path.endswith('.py'):
        pattern = module_path + '%'
    else:
        pattern = module_path

    target_files = conn.execute(
        "SELECT file_id, rel_path, line_count FROM ci_files WHERE rel_path LIKE ?",
        (pattern,)
    ).fetchall()

    if not target_files:
        print(f"No files matching '{module_path}'.")
        return None

    target_paths = {r['rel_path'] for r in target_files}
    target_ids = {r['file_id'] for r in target_files}

    # Build module-to-path map
    all_files = conn.execute("SELECT file_id, rel_path, line_count FROM ci_files").fetchall()
    path_to_module = {}
    file_lines = {}
    for r in all_files:
        rp = r['rel_path']
        mod = rp.replace('/', '.').replace('\\', '.').removesuffix('.py')
        path_to_module[mod] = rp
        parts = mod.split('.')
        if len(parts) > 1:
            path_to_module[parts[-1]] = rp
        file_lines[rp] = r['line_count']

    # Get outgoing imports from target
    out_edges = defaultdict(int)
    rows = conn.execute("""
        SELECT fi.rel_path as src, i.module
        FROM ci_imports i
        JOIN ci_files fi ON i.file_id = fi.file_id
        WHERE fi.rel_path LIKE ?
    """, (pattern,)).fetchall()

    for r in rows:
        mod = r['module']
        dst = path_to_module.get(mod)
        if not dst:
            for mn, rp in path_to_module.items():
                if mod.endswith(mn) or mn.endswith(mod):
                    dst = rp
                    break
        if dst and dst not in target_paths:
            out_edges[(r['src'], dst)] += 1

    # Get incoming imports into target
    in_edges = defaultdict(int)
    rows = conn.execute("SELECT fi.rel_path as src, i.module FROM ci_imports i JOIN ci_files fi ON i.file_id = fi.file_id").fetchall()
    for r in rows:
        if r['src'] in target_paths:
            continue
        mod = r['module']
        dst = path_to_module.get(mod)
        if not dst:
            for mn, rp in path_to_module.items():
                if mod.endswith(mn) or mn.endswith(mod):
                    dst = rp
                    break
        if dst and dst in target_paths:
            in_edges[(r['src'], dst)] += 1

    # Get self.mw references
    mw_refs = {}
    for fid in target_ids:
        row = conn.execute(
            "SELECT COUNT(*) as c FROM ci_attribute_access WHERE file_id=? AND target='self.mw'",
            (fid,)
        ).fetchone()
        if row['c'] > 0:
            mw_refs[fid] = row['c']

    # Get self.state references
    state_refs = {}
    for fid in target_ids:
        row = conn.execute(
            "SELECT COUNT(*) as c FROM ci_attribute_access WHERE file_id=? AND target='self.state'",
            (fid,)
        ).fetchone()
        if row['c'] > 0:
            state_refs[fid] = row['c']

    # Build graph
    G = nx.DiGraph()

    for r in target_files:
        rp = r['rel_path']
        lbl = rp.replace('.py', '').split('/')[-1]
        extras = []
        if r['file_id'] in mw_refs:
            extras.append(f"mw:{mw_refs[r['file_id']]}")
        if r['file_id'] in state_refs:
            extras.append(f"st:{state_refs[r['file_id']]}")
        if extras:
            lbl += f"\n({', '.join(extras)})"
        G.add_node(rp, label=lbl, node_type='target',
                   lines=r['line_count'])

    # Add outgoing deps
    for (src, dst), count in out_edges.items():
        if dst not in G.nodes():
            lbl = dst.replace('.py', '').split('/')[-1]
            G.add_node(dst, label=lbl, node_type='dep_out',
                       lines=file_lines.get(dst, 100))
        G.add_edge(src, dst, weight=count, edge_type='imports')

    # Add incoming deps
    for (src, dst), count in in_edges.items():
        if src not in G.nodes():
            lbl = src.replace('.py', '').split('/')[-1]
            G.add_node(src, label=lbl, node_type='dep_in',
                       lines=file_lines.get(src, 100))
        G.add_edge(src, dst, weight=count, edge_type='imported_by')

    if not G.nodes():
        print(f"No dependencies found for '{module_path}'.")
        return None

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Draw edges
    for u, v, d in G.edges(data=True):
        color = '#3498db' if d.get('edge_type') == 'imports' else '#e67e22'
        w = d.get('weight', 1)
        ax.annotate('', xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1 + w * 0.5, alpha=0.5))

    # Draw nodes
    for node in G.nodes():
        x, y = pos[node]
        nd = G.nodes[node]
        nt = nd.get('node_type', 'other')
        lines = nd.get('lines', 100)

        if nt == 'target':
            size, ec = max(300, lines / 2), '#e74c3c'
        elif nt == 'dep_out':
            size, ec = max(100, lines / 4), '#3498db'
        else:
            size, ec = max(100, lines / 4), '#e67e22'

        color = get_module_color(node)
        ax.scatter(x, y, s=size, c=color, zorder=5, edgecolors=ec,
                   linewidths=2 if nt == 'target' else 1, alpha=0.85)
        ax.text(x, y - 0.04, nd.get('label', node), ha='center', va='top',
                fontsize=7 if nt == 'target' else 6, zorder=6,
                fontweight='bold' if nt == 'target' else 'normal')

    ax.set_title(f'Module Dependencies: {module_path}\n'
                 f'({len(target_files)} file(s), '
                 f'{len(out_edges)} outgoing, {len(in_edges)} incoming imports)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.axis('off')

    legend_handles = [
        mpatches.Patch(facecolor='white', edgecolor='#e74c3c', linewidth=2,
                       label=f'Target module ({len(target_files)} files)'),
        mpatches.Patch(color='#3498db', label=f'Imports ({len(out_edges)} outgoing)'),
        mpatches.Patch(color='#e67e22', label=f'Imported by ({len(in_edges)} incoming)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    safe_name = module_path.replace('/', '_').replace('\\', '_').rstrip('_')
    return save_and_print(fig, f'module_{safe_name}.png')


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

GRAPH_MAP = {
    'coupling': ('MainWindow coupling web', graph_coupling),
    'imports': ('File import dependency graph', graph_imports),
    'signals': ('Signal/slot flow graph', graph_signals),
    'state': ('AppState field access heatmap', graph_state),
    'complexity': ('File size vs coupling scatter', graph_complexity),
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Code Index Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Graph types:
  all            Generate all 5 overview graphs
  coupling       MainWindow coupling web
  imports        File import dependency graph
  signals        Signal/slot flow graph
  state          AppState field access heatmap
  complexity     File size vs coupling scatter
  impact NAME    Call graph around a specific function
  module PATH    Dependencies of a file or module (e.g. 'core/' or 'main.py')
""")
    parser.add_argument('graph', nargs='?', default='all',
                        help="Graph type to generate (default: all)")
    parser.add_argument('target', nargs='?', default=None,
                        help="Target for impact/module graphs")
    parser.add_argument('--no-show', action='store_true',
                        help="Don't open the image after generating")

    args = parser.parse_args()

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        print(f"Error: Code index not found at {DB_PATH}")
        print("Run: python tools/code_index_cli.py rebuild")
        sys.exit(1)

    conn = get_conn()
    paths = []

    if args.graph == 'all':
        for name, (desc, func) in GRAPH_MAP.items():
            path = func(conn)
            if path:
                paths.append(path)
    elif args.graph == 'impact':
        if not args.target:
            print("Usage: code_index_visualize.py impact FUNCTION_NAME")
            sys.exit(1)
        path = graph_impact(conn, args.target)
        if path:
            paths.append(path)
    elif args.graph == 'module':
        if not args.target:
            print("Usage: code_index_visualize.py module MODULE_PATH")
            sys.exit(1)
        path = graph_module(conn, args.target)
        if path:
            paths.append(path)
    elif args.graph in GRAPH_MAP:
        _, func = GRAPH_MAP[args.graph]
        path = func(conn)
        if path:
            paths.append(path)
    else:
        print(f"Unknown graph type: {args.graph}")
        parser.print_help()
        sys.exit(1)

    conn.close()

    if paths and not args.no_show:
        for path in paths:
            subprocess.Popen(['start', '', str(path)], shell=True)


if __name__ == '__main__':
    main()
