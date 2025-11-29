# PRD – Tooltip Coverage for StockDataGUI

## 1. Background & Problem Statement

- The `StockDataGUI` class orchestrates a large Tkinter UI that manages ticker lists, chart tabs (individual, comparison, seasonality), analytical panes, SEC tooling, and action bars.@stock_charts_10k10q/gui.py#61-450
- Each tab (e.g., Fundamental Analysis, Business Analysis, Market News, Buffett & CANSLIM, SEC Filings) exposes numerous buttons, comboboxes, listboxes, and text areas without inline guidance.@stock_charts_10k10q/gui.py#671-1060
- Users must currently rely on external documentation or trial-and-error to understand purpose, prerequisites, and side effects of controls.
- Tooltips are required to reduce onboarding time, prevent destructive actions, and surface hidden keyboard or threading considerations.

## 2. Goals & Success Metrics

| Goal | Success Metric |
| --- | --- |
| Provide contextual help for every interactive widget | 100% of actionable widgets display a tooltip within 300 ms hover delay |
| Reduce user confusion for complex flows (SEC extraction, business analysis) | Support ticket volume for “how do I …” questions drops by 50% within two releases |
| Maintain discoverability for power-user shortcuts | At least 10 core actions document shortcuts or advanced behaviors in their tooltip copy |

## 3. Scope

### In Scope

1. All interactive widgets created inside `StockDataGUI._create_widgets` and any helper panes created later (e.g., dynamic dialogs, `CustomDateEntry`).@stock_charts_10k10q/gui.py#431-1060 @stock_charts_10k10q/custom_widgets.py#1-170
2. Status labels that act as affordances (clickable or conveying mode) should show read-only explanatory tooltips.
3. Tooltip system must support dynamic content (e.g., ticker-specific messaging) when state changes.

### Out of Scope

1. External scripts (command-line utilities, audio transcription tools) unless they share the same Tk root.
2. Plotly browser views or HTML exports.
3. Reflow of layout purely for tooltip placement (can be handled in future UX pass).

## 4. Users & Pain Points

- **Equity Researchers / DIY Investors**: Need certainty before running heavy downloads or SEC calls.
- **Data Ops Engineers**: Maintain data hygiene; need cues on threading limitations and cache usage.
- **New Analysts**: Require guided discovery of niche tabs (Buffett & CANSLIM, Market News Blog) without reading code.
Pain points: hidden dependencies (e.g., “Use Mock Data”), irreversible actions, inconsistent naming between tabs and buttons.

## 5. User Stories

1. *As a new user, I want each button to explain its action and prerequisites so I can avoid triggering long downloads accidentally.*
2. *As an advanced analyst, I want tooltips to surface keyboard shortcuts or modifier actions so I can work faster.*
3. *As a maintainer, I want a centralized tooltip registry so translations or copy updates are low-effort.*

## 6. Detailed Requirements

### 6.1 Tooltip Coverage Baseline

- Every `ttk.Button`, `ttk.Combobox`, `ttk.Entry`, `ttk.Checkbutton`, `ttk.Menubutton`, `ttk.Spinbox`, `tk.Listbox`, `ttk.Treeview`, and `tk.Text` control must register a tooltip at creation time.
- Multi-element widgets (e.g., paned windows, notebooks) need tooltips for tabs or frame labels when they communicate distinct functionality.

### 6.2 Content Guidelines

- 1–2 concise sentences (<180 chars) for simple actions; up to 3 sentences for destructive or multi-step flows.
- Prefix warnings with **⚠** and success tips with **💡** to provide scannable semantics.
- Reference dependent settings (“Requires ticker selection”, “Uses cached SEC data when available”).

### 6.3 Interaction Behavior

- Default delay: 300 ms hover; disappear 5 s after display or immediately on pointer leave.
- Tooltips must follow cursor within the root window, offsetting (x+12, y+18) to avoid covering controls.
- Support for keyboard focus: when a widget gains focus via Tab, pressing `Shift+F1` should show the tooltip near the focused widget.

### 6.4 Accessibility & Internationalization

- Ensure tooltip text is stored centrally to enable future localization.
- Provide ARIA-like description bridging by setting `.tooltip_text` attribute so screen readers (when supported) can retrieve context.
- Avoid emoji fallbacks when platform font lacks glyphs; provide text-only alternative.

### 6.5 Configurability & Persistence

- Provide a settings toggle (e.g., “Show Tooltips”) persisted via existing config mechanism (e.g., JSON or ini) so power users can disable.
- Future-proof by allowing verbosity levels: `basic`, `detailed` (maybe via environment variable or hidden setting).

### 6.6 Performance & Resilience

- Tooltip manager must be thread-safe with respect to `setup_thread_safe_tkinter` to avoid `TclError` when controls are destroyed.@stock_charts_10k10q/gui.py#71-78
- Guard against orphaned tooltips when widgets are re-rendered (e.g., chart frames cleared in `_display_plotly_chart`).@stock_charts_10k10q/gui.py#313-430

## 7. UX & Component Inventory

| Component Group | Representative Widgets | Tooltip Focus |
| --- | --- | --- |
| **Ticker List Controls** | Filter entry, combobox, refresh button, clipboard actions.@stock_charts_10k10q/gui.py#175-312 | Explain naming conventions, watchlist implications, clipboard format. |
| **Chart Notebook Tabs** | Individual, Comparison, Seasonality, Fundamental, Business, Market News, Buffett & CANSLIM, SEC Filings.@stock_charts_10k10q/gui.py#671-773 | Summaries of data source, refresh cadence, limitations (e.g., seasonality year selection, SEC rate limits). |
| **Buffett & CANSLIM Pane** | Analyze button, zoom controls, explanation text area.@stock_charts_10k10q/gui.py#703-768 | Clarify that analysis reuses selected ticker and may take ~30s. |
| **Business Analysis Controls** | Multiple action buttons, search entry, filters, spinboxes, checkboxes.@stock_charts_10k10q/gui.py#939-1015 | Indicate API usage, rate limits, filter syntax, change-over-time toggle. |
| **Market News Blog** | Text viewer, refresh actions (when available).@stock_charts_10k10q/gui.py#1017-1039 | Clarify update cadence and Finviz parsing state. |
| **Seasonality Controls** | Menubutton for multi-year selection and dynamic menu entries.@stock_charts_10k10q/gui.py#1041-1054 | Describe multi-select usage and chart regeneration triggers. |
| **SEC Filings Tab** | Ticker entry, form type combo, mock data toggle, extract button, table listbox, Treeview, cache controls.@stock_charts_10k10q/gui.py#775-884 | Warn about SEC rate limits, caching behavior, export destinations. |
| **Custom Widgets** | `CustomDateEntry` button and calendar navigation.@stock_charts_10k10q/custom_widgets.py#1-170 | Explain date format, navigation buttons, default values. |

## 8. Technical Considerations

1. **Tooltip Manager**
   - Implement as reusable class (e.g., `TooltipManager`) storing references with weakrefs to avoid memory leaks.
   - Provide helper `attach_tooltip(widget, text, dynamic_callback=None)` so code stays declarative.
2. **Data-Driven Definitions**
   - Maintain dictionary keyed by semantic IDs (e.g., `"ticker_list.refresh"`) to decouple copy from widget instantiation.
   - Support late-binding functions returning strings (for stateful hints like currently selected ticker, cached status).
3. **Thread Safety**
   - All tooltip show/hide operations must run in Tk main thread; use existing queued-call helper from `thread_safe_tkinter` if needed.
4. **Testing Hooks**
   - Provide method to enumerate registered tooltips for automated UI smoke tests (can be CLI command or debug log dump).

## 9. Dependencies & Risks

| Dependency | Risk | Mitigation |
| --- | --- | --- |
| Tkinter base widgets | Lack of native tooltip support | Use custom class or vetted snippet with event bindings |
| Long-running background threads | Hover events may fire on destroyed widgets | Track widget `.winfo_exists()` before showing tooltip |
| Copy accuracy | Tooltips may drift from behavior | Add review step in release checklist; store copy centrally |
| Localization | English-only copy may block future translation | Keep strings in single module and avoid concatenation |

## 10. Rollout Plan & Milestones

1. **Design & Inventory (Week 1)** – Confirm widget list, finalize copy template, approve style guidelines.
2. **Infrastructure (Week 2)** – Implement tooltip manager, global enable/disable, keyboard shortcut.
3. **Phase 1 Coverage (Week 3)** – Apply tooltips to ticker controls + chart tabs.
4. **Phase 2 Coverage (Week 4)** – Apply to analytical tabs (Fundamental, Business, Market News, Buffett & CANSLIM).
5. **Phase 3 Coverage (Week 5)** – Apply to SEC tab, custom dialogs, residual widgets.
6. **QA & Accessibility Review (Week 6)** – Manual smoke tests, screen-reader verification where possible.

## 11. Acceptance Criteria

- [ ] Hovering or focusing any actionable widget reveals a tooltip that matches approved copy.
- [ ] Tooltips respect enable/disable setting and keyboard shortcut display.
- [ ] Tooltip manager handles widget destruction without residual windows or exceptions.
- [ ] Documentation updated (README/WORKLOG) with instructions for adding future tooltips.
- [ ] Automated smoke test or script validates registry count equals number of actionable widgets ±5% (exception list documented).

## 12. Open Questions

1. Should tooltip verbosity tiers be user-facing at launch or hidden behind config?
2. Is localization required in the next two quarters? (Impacts storage format.)
3. Can we surface analytics (e.g., tooltip usage) without adding heavy dependencies?
