# Business Analysis (BA) Caching & History

This document describes how the Business Analysis caching, history, and change-over-time summary work in this project.

## Overview

- **Purpose**: Speed up BA retrieval by reusing recent results and preserve historical runs for comparison.
- **Storage**: Cached results are saved as Markdown under `stock_charts_10k10q/cache/`.
- **Freshness window**: 30 days. If the latest cache for a ticker is newer than 30 days, it is loaded instead of re-running.
- **History**: Each run is saved with a unique timestamped filename so older versions are kept (no overwrite).
- **Change Over Time**: When history exists, a short summary is appended below the BA content.

## File(s) Touched

- `stock_charts_10k10q/gui.py`
  - Added helper methods for cache pathing and history:
    - `_get_ba_cache_file(ticker)` → returns a timestamped markdown path: `cache/{TICKER}_business_analysis_{YYYYMMDD_HHMMSS}.md`
    - `_find_latest_ba_cache_file(ticker)` → returns most recent BA markdown file for the ticker
    - `_list_ba_cache_files(ticker)` → returns all history for the ticker, newest first
    - `_build_ba_change_over_time_section(ticker, max_items=5)` → computes simple diff metrics between adjacent versions and formats as markdown
  - Updated flows:
    - `_run_business_analysis()` → loads the latest fresh cache if available; else runs BA and saves a new timestamped `.md`. Appends “Change Over Time” when history exists.
    - `_load_cached_analysis(ticker)` → loads the latest fresh cache and appends the history section.

## Cache File Naming & Location

- **Directory**: `stock_charts_10k10q/cache/`
- **File format**: `{TICKER}_business_analysis_{YYYYMMDD_HHMMSS}.md`
- **Rationale**: Timestamped filenames prevent overwrites and maintain a complete history for each ticker.

## Freshness Logic

- Uses the latest cached file for a ticker (by modification time).
- If `mtime` is within 30 days, the cached content is loaded immediately instead of re-running analysis.
- Otherwise, the analysis is re-run and a new timestamped file is created.

## Change Over Time Section

- Triggered when at least 2 historical files are present for the ticker.
- Appended under the current BA content, separated by `---`.
- Shows:
  - The timestamps of up to the last 5 versions (latest first)
  - For each adjacent pair, the number of added/removed lines and a similarity ratio

### Example Section (illustrative)

```markdown
## Change Over Time (last 3 versions)
- 2025-10-08 06:45:12 latest
- 2025-09-15 12:08:31 
- 2025-08-07 09:52:00 

- Δ 2025-09-15 12:08:31 → 2025-10-08 06:45:12: +23 / -10, similarity 0.89
- Δ 2025-08-07 09:52:00 → 2025-09-15 12:08:31: +12 / -8, similarity 0.92
```

## UI Controls (Business Analysis Tab)

- **Freshness (days)**: Spinbox to set the cache freshness window. Controls the `days` argument passed to `_is_cache_fresh(...)`.
- **History (max)**: Spinbox to set how many past versions to summarize in the Change Over Time section (`max_items`).
- **Show Change Over Time**: Checkbox to toggle rendering of the summary section.

## How To Adjust

- **Freshness window**: Change the `days` parameter in calls to `_is_cache_fresh(path, days=30)` or adjust via the UI spinbox.
- **History length in summary**: Change the `max_items` default in `_build_ba_change_over_time_section(ticker, max_items=5)` or adjust via the UI spinbox.
- **Disable summary**: Uncheck the UI toggle or skip calling `_build_ba_change_over_time_section()`.

## Known Limitations

- Diff stats are simple, line-based metrics and a similarity ratio; they do not perform semantic understanding of changes.
- Very large historical files may make computing diffs slower; `max_items` helps bound the work.

## Developer Notes

- Caching is only for Business Analysis markdown results.
- All UI updates use thread-safe utilities where necessary (e.g., `safe_update_text_widget`).
- Timestamp header is included in saved `.md` to provide immediate context even outside the app.
