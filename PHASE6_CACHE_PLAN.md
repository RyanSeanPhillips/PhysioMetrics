# Phase 6: Cache Management - Analysis

## Current State

### Cache System Overview
The application has **two cache systems**:

#### 1. Active Cache: `st.proc_cache` (in core/state.py)
```python
# core/state.py line 58
proc_cache: Dict[Tuple[str,int], np.ndarray] = field(default_factory=dict)
```

**Usage in main.py:**
- Line 764: `st.proc_cache.clear()` - When file loaded
- Line 786: `st.proc_cache.clear()` - When channel changed
- Line 852: `st.proc_cache.clear()` - When filter params changed
- Line 943: `st.proc_cache.clear()` - When z-score toggled
- Line 959-960: Cache read in `_current_trace()`
- Line 984: Cache write in `_current_trace()`
- Line 1061-1062: Cache read in `_get_processed_for()`
- Line 1083: Cache write in `_get_processed_for()`
- Line 1763: `st.proc_cache.clear()` - When global z-score computed

**Key generation:**
- `_proc_key(chan, sweep)` creates tuple of (chan, sweep, filter_params)
- Ensures cache invalidation when settings change

#### 2. Dead Code: `self._global_trace_cache` (in main.py)
```python
# main.py lines 460, 601
self._global_trace_cache = {}  # Initialized but NEVER USED
```

**No actual usage** - This appears to be vestigial code from an old implementation.

**Exception:** `export_manager.py` references `self.window._global_trace_cache` (lines 1000-1001, 1035, 1764-1765), but this appears to be a **different cache** used only during export operations for storing computed metric traces.

## Findings

### Current System is Already Well-Organized!

The cache system is actually in a good state:
1. ✅ Cache is in the right place (`core/state.py`)
2. ✅ Cache key generation is clean (`_proc_key()`)
3. ✅ Cache clearing is appropriately placed
4. ✅ Cache reads/writes are efficient

### Issues Found:

1. **Dead Code**: `self._global_trace_cache = {}` in main.py (lines 460, 601)
   - Initialized twice, never used in main.py
   - Should be removed

2. **Export Manager Cache Confusion**:
   - `export_manager.py` uses `self.window._global_trace_cache` for storing metric traces during export
   - This is a **different cache** with a different purpose (sweep → metric_traces dict)
   - Should be renamed to avoid confusion with the removed dead code

3. **Cache Key in Wrong Place**:
   - `_proc_key()` method is tightly coupled to main.py's filter parameters
   - Could be moved to a cache manager, but current placement is reasonable

## Recommendation

### Option 1: Minimal Cleanup (RECOMMENDED)
**Time: 30 minutes**
**Risk: LOW**

1. Remove dead code: `self._global_trace_cache = {}` from main.py (2 lines)
2. Rename export manager's cache: `_global_trace_cache` → `_export_metric_cache`
3. Add documentation comment explaining the two different caches
4. Mark Phase 6 as "Already Complete" (cache already in core/state.py)

**Rationale**: The cache system is already well-designed. No need to extract anything.

### Option 2: Full Extraction (NOT RECOMMENDED)
**Time: 2-3 hours**
**Risk: MEDIUM**

Create `caching/trace_cache.py`:
- Move `_proc_key()` to cache manager
- Add cache statistics/monitoring
- Add LRU eviction policy
- Add max size limits

**Rationale**: Current system works fine. This would be over-engineering.

### Option 3: Skip Phase 6
**Time: 0 minutes**
**Risk: NONE**

Just document that Phase 6 is "Already Complete" and move on.

## Decision Point

Phase 6 (Cache Management) was already completed when the cache was moved to `core/state.py` (probably during initial architecture setup). The modularization plan assumed cache code was still in main.py, but it's not.

**Recommendation**: Do Option 1 (Minimal Cleanup) to remove dead code and improve clarity, then mark Phase 6 as complete.

---

## If Proceeding with Option 1:

### Files to Modify:
1. `main.py` - Remove lines 460, 601 (`self._global_trace_cache = {}`)
2. `export/export_manager.py` - Rename cache (3 locations: lines 1000-1001, 1035, 1764-1765)
3. Add comment explaining the distinction

### Changes:
```python
# In export/export_manager.py (line 1000):
# OLD:
if not hasattr(self, '_global_trace_cache'):
    self.window._global_trace_cache = {}

# NEW:
if not hasattr(self.window, '_export_metric_cache'):
    self.window._export_metric_cache = {}  # Cache for computed metric traces during export

# Similar changes for lines 1035, 1764-1765
```

### Testing:
- Verify export still works
- Verify caching behavior unchanged
- No performance impact expected

---

## Conclusion

**Phase 6 doesn't need extraction** - the cache is already properly located in `core/state.py`. We just need to clean up dead code and clarify the export-specific cache.

**Estimated time for cleanup: 30 minutes**
**Lines removed: 2 lines**
**Complexity reduced: Minimal**

Should we proceed with Option 1 (cleanup), or skip Phase 6 entirely and move to Phase 5b (Consolidation)?
