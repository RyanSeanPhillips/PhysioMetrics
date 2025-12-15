"""
Parallel processing utilities with robust fallback to sequential execution.

This module provides helpers for parallelizing computationally expensive operations
like peak detection and metric computation across multiple sweeps.

Key design principles:
1. Robust fallback: If parallel execution fails for any reason, fall back to sequential
2. Chunked processing: Process in chunks to enable progress reporting
3. Thread-safe workers: Worker functions receive copies of data, not shared state
4. Progress callbacks: UI stays responsive with progress updates
"""

import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, List, Dict, Any, Optional, Tuple
import numpy as np


# Default to using ThreadPoolExecutor (GIL-friendly, lower overhead)
# ProcessPoolExecutor can be faster for CPU-bound work but has higher overhead
# and requires pickling, which can fail for complex objects
DEFAULT_EXECUTOR_TYPE = 'thread'

# Maximum workers - default to CPU count, capped at 8 to avoid memory issues
MAX_WORKERS = min(os.cpu_count() or 4, 8)


def parallel_map_with_fallback(
    worker_func: Callable,
    items: List[Any],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    chunk_size: int = 10,
    executor_type: str = DEFAULT_EXECUTOR_TYPE,
    max_workers: int = MAX_WORKERS,
    description: str = "Processing"
) -> List[Any]:
    """
    Execute worker_func on each item in parallel, with fallback to sequential.

    Args:
        worker_func: Function to call for each item. Should be (item) -> result.
                    Must be thread-safe and not modify shared state.
        items: List of items to process
        progress_callback: Optional callback(current, total, message) for progress updates
        chunk_size: Process this many items before updating progress
        executor_type: 'thread' for ThreadPoolExecutor, 'process' for ProcessPoolExecutor
        max_workers: Maximum number of parallel workers
        description: Description for progress messages

    Returns:
        List of results in the same order as items

    Note:
        If parallel execution fails for any reason (exception, import error, etc.),
        this function will fall back to sequential execution and print a warning.
    """
    n_items = len(items)
    if n_items == 0:
        return []

    # For very small workloads, don't bother with parallelization overhead
    if n_items <= 2:
        return _sequential_map(worker_func, items, progress_callback, description)

    # Try parallel execution
    try:
        return _parallel_map(
            worker_func, items, progress_callback, chunk_size,
            executor_type, max_workers, description
        )
    except Exception as e:
        # Log the error and fall back to sequential
        print(f"[parallel_utils] Parallel execution failed: {e}")
        print(f"[parallel_utils] Falling back to sequential execution...")
        traceback.print_exc()
        return _sequential_map(worker_func, items, progress_callback, description)


def _parallel_map(
    worker_func: Callable,
    items: List[Any],
    progress_callback: Optional[Callable],
    chunk_size: int,
    executor_type: str,
    max_workers: int,
    description: str
) -> List[Any]:
    """Internal parallel execution."""
    n_items = len(items)
    results = [None] * n_items

    # Choose executor type
    if executor_type == 'process':
        ExecutorClass = ProcessPoolExecutor
    else:
        ExecutorClass = ThreadPoolExecutor

    completed = 0

    with ExecutorClass(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(worker_func, item): idx
            for idx, item in enumerate(items)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                # Re-raise to trigger fallback
                raise RuntimeError(f"Worker failed for item {idx}: {e}") from e

            completed += 1

            # Update progress at chunk boundaries
            if progress_callback and (completed % chunk_size == 0 or completed == n_items):
                progress_callback(completed, n_items, f"{description} ({completed}/{n_items})")

    return results


def _sequential_map(
    worker_func: Callable,
    items: List[Any],
    progress_callback: Optional[Callable],
    description: str
) -> List[Any]:
    """Fallback sequential execution."""
    n_items = len(items)
    results = []

    for i, item in enumerate(items):
        results.append(worker_func(item))

        if progress_callback:
            progress_callback(i + 1, n_items, f"{description} ({i + 1}/{n_items})")

    return results


def chunked_parallel_map(
    worker_func: Callable,
    items: List[Any],
    chunk_size: int = 10,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    executor_type: str = DEFAULT_EXECUTOR_TYPE,
    max_workers: int = MAX_WORKERS,
    description: str = "Processing"
) -> List[Any]:
    """
    Process items in chunks, parallelizing within each chunk.

    This is useful when you want more control over progress reporting
    or when you need to batch writes to shared state.

    Args:
        worker_func: Function to call for each item
        items: List of items to process
        chunk_size: Number of items per chunk
        progress_callback: Optional callback(current, total, message)
        executor_type: 'thread' or 'process'
        max_workers: Maximum parallel workers
        description: Description for progress messages

    Returns:
        List of results in same order as items
    """
    n_items = len(items)
    if n_items == 0:
        return []

    results = []

    # Process in chunks
    for chunk_start in range(0, n_items, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_items)
        chunk_items = items[chunk_start:chunk_end]

        # Process this chunk (with internal fallback)
        chunk_results = parallel_map_with_fallback(
            worker_func=worker_func,
            items=chunk_items,
            progress_callback=None,  # We handle progress at chunk level
            chunk_size=len(chunk_items),
            executor_type=executor_type,
            max_workers=max_workers,
            description=description
        )

        results.extend(chunk_results)

        # Update progress after each chunk
        if progress_callback:
            progress_callback(
                chunk_end, n_items,
                f"{description} ({chunk_end}/{n_items})"
            )

    return results


# ============================================================================
# Peak Detection Helpers
# ============================================================================

def create_peak_detection_worker(
    get_processed_func: Callable[[str, int], np.ndarray],
    detect_peaks_func: Callable,
    compute_breath_events_func: Callable,
    label_peaks_func: Callable,
    compute_metrics_func: Callable,
    analyze_chan: str,
    sr_hz: float,
    t: np.ndarray,
    thresh: float,
    min_dist_samples: Optional[int],
    direction: str = "up"
) -> Callable[[int], Dict]:
    """
    Create a worker function for parallel peak detection on a single sweep.

    This creates a closure that captures all the parameters needed to process
    a single sweep, making it suitable for parallel execution.

    Args:
        get_processed_func: Function to get processed signal for (channel, sweep_idx)
        detect_peaks_func: Peak detection function
        compute_breath_events_func: Breath events computation function
        label_peaks_func: Peak labeling function
        compute_metrics_func: Peak metrics computation function
        analyze_chan: Channel to analyze
        sr_hz: Sampling rate
        t: Time array
        thresh: Threshold for labeling
        min_dist_samples: Minimum distance between peaks
        direction: Peak direction ("up" or "down")

    Returns:
        Worker function that takes sweep_idx and returns detection results dict
    """
    def worker(sweep_idx: int) -> Dict:
        """Process a single sweep and return all detection results."""
        # Get processed signal
        y_proc = get_processed_func(analyze_chan, sweep_idx)

        # Step 1: Detect ALL peaks
        all_peak_indices = detect_peaks_func(
            y=y_proc, sr_hz=sr_hz,
            thresh=None,
            prominence=None,
            min_dist_samples=min_dist_samples,
            direction=direction,
            return_all=True
        )

        # Step 2: Compute breath events for ALL peaks
        all_breaths = compute_breath_events_func(
            y_proc, all_peak_indices, sr_hz=sr_hz, exclude_sec=0.030
        )

        # Step 3: Label peaks by threshold
        all_peaks_data = label_peaks_func(
            y=y_proc,
            peak_indices=all_peak_indices,
            thresh=thresh,
            direction=direction
        )
        all_peaks_data['labels_threshold_ro'] = all_peaks_data['labels'].copy()

        # Step 4: Compute peak metrics (for ML and display)
        peak_metrics = compute_metrics_func(
            y=y_proc,
            all_peak_indices=all_peak_indices,
            breath_events=all_breaths,
            sr_hz=sr_hz
        )

        # Step 5: Extract labeled peaks for display
        labeled_mask = all_peaks_data['labels'] == 1
        labeled_indices = all_peak_indices[labeled_mask]

        # Step 6: Compute breath events for labeled peaks only
        labeled_breaths = compute_breath_events_func(
            y_proc, labeled_indices, sr_hz=sr_hz, exclude_sec=0.030
        )

        return {
            'sweep_idx': sweep_idx,
            'y_proc': y_proc,
            'all_peak_indices': all_peak_indices,
            'all_breaths': all_breaths,
            'all_peaks_data': all_peaks_data,
            'peak_metrics': peak_metrics,
            'labeled_indices': labeled_indices,
            'labeled_breaths': labeled_breaths
        }

    return worker


# ============================================================================
# Export Helpers
# ============================================================================

def create_metric_computation_worker(
    compute_metric_func: Callable,
    get_processed_func: Callable[[str, int], np.ndarray],
    analyze_chan: str,
    t: np.ndarray,
    sr_hz: float,
    peaks_by_sweep: Dict[int, np.ndarray],
    breath_by_sweep: Dict[int, Dict],
    keys_to_compute: List[str]
) -> Callable[[int], Dict[str, np.ndarray]]:
    """
    Create a worker function for parallel metric computation on a single sweep.

    Args:
        compute_metric_func: Function to compute a metric trace
        get_processed_func: Function to get processed signal
        analyze_chan: Channel to analyze
        t: Time array
        sr_hz: Sampling rate
        peaks_by_sweep: Dict mapping sweep_idx to peak indices
        breath_by_sweep: Dict mapping sweep_idx to breath events
        keys_to_compute: List of metric keys to compute

    Returns:
        Worker function that takes sweep_idx and returns dict of metric traces
    """
    def worker(sweep_idx: int) -> Tuple[int, Dict[str, np.ndarray]]:
        """Compute all metrics for a single sweep."""
        y_proc = get_processed_func(analyze_chan, sweep_idx)
        pks = peaks_by_sweep.get(sweep_idx, np.array([], dtype=int))
        br = breath_by_sweep.get(sweep_idx, {})

        traces = {}
        for key in keys_to_compute:
            trace = compute_metric_func(key, t, y_proc, sr_hz, pks, br, sweep=sweep_idx)
            traces[key] = trace

        return (sweep_idx, traces)

    return worker


# ============================================================================
# Testing / Diagnostics
# ============================================================================

def test_parallel_performance(n_items: int = 100, work_ms: int = 10):
    """
    Test parallel vs sequential performance.

    Args:
        n_items: Number of items to process
        work_ms: Milliseconds of simulated work per item
    """
    import time

    def dummy_worker(item):
        """Simulate some work."""
        time.sleep(work_ms / 1000.0)
        return item * 2

    items = list(range(n_items))

    # Sequential
    t0 = time.time()
    seq_results = _sequential_map(dummy_worker, items, None, "Sequential")
    seq_time = time.time() - t0

    # Parallel (thread)
    t0 = time.time()
    par_results = parallel_map_with_fallback(
        dummy_worker, items, None, 10, 'thread', MAX_WORKERS, "Parallel"
    )
    par_time = time.time() - t0

    # Verify results match
    assert seq_results == par_results, "Results mismatch!"

    speedup = seq_time / par_time if par_time > 0 else float('inf')

    print(f"[test_parallel_performance]")
    print(f"  Items: {n_items}, Work per item: {work_ms}ms")
    print(f"  Workers: {MAX_WORKERS}")
    print(f"  Sequential: {seq_time:.2f}s")
    print(f"  Parallel:   {par_time:.2f}s")
    print(f"  Speedup:    {speedup:.1f}x")

    return speedup


if __name__ == "__main__":
    # Run performance test
    test_parallel_performance(n_items=50, work_ms=50)
