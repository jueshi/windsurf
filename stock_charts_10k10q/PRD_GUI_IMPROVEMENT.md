# Product Requirements Document: GUI Improvement

## Project Overview

**Project Name:** Stock Analysis Tool GUI Modernization  
**Branch:** `improve-gui`  
**Date:** November 30, 2025  
**Status:** Draft

### Executive Summary

The current GUI for the stock analysis tool is functional but suffers from information overload, dated visual design, and poor content presentation. This PRD outlines a comprehensive redesign to create a modern, intuitive, and user-friendly interface that enhances productivity for financial analysis workflows.

---

## Problem Statement

### Current Issues

1. **Information Overload** - Interface displays too much information simultaneously
2. **Cluttered Layout** - Multiple narrow panels compete for attention
3. **Dated Aesthetics** - Visual design lacks modern polish
4. **Poor Content Hierarchy** - Key information is not easily scannable
5. **Inconsistent Controls** - Buttons and actions lack clear grouping and prominence

### User Impact

- Slower navigation and task completion
- Cognitive fatigue from visual clutter
- Difficulty finding key information quickly
- Reduced adoption by new users

---

## Goals & Success Metrics

### Primary Goals

1. Reduce visual clutter by 40%
2. Improve information hierarchy and scannability
3. Modernize visual design to professional standards
4. Enhance user workflow efficiency

### Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Time to locate key data | ~10 sec | <3 sec |
| User satisfaction (1-5) | 2.5 | 4.0+ |
| New user onboarding time | 30 min | 10 min |
| Screen real estate utilization | 60% | 85% |

---

## Requirements

### 1. Layout Organization & De-cluttering

#### 1.1 Consolidated Sidebar Panel

**Priority:** P0 (Critical)

| Requirement | Description |
|-------------|-------------|
| **REQ-1.1.1** | Combine "Available Tickers" and "Watch List" into a single tabbed panel |
| **REQ-1.1.2** | Implement tab switching between "Available" and "Watch List" views |
| **REQ-1.1.3** | Add ticker count badge on each tab |
| **REQ-1.1.4** | Maintain all existing functionality (filter, sort, right-click menus) |

**Mockup Concept:**
```
┌─────────────────────────┐
│ [Available (52)] [Watch (12)] │  ← Tabbed header
├─────────────────────────┤
│ Filter: [________] [A-Z]│
├─────────────────────────┤
│ AAPL                    │
│ MSFT                    │
│ GOOGL                   │
│ ...                     │
└─────────────────────────┘
```

#### 1.2 Timeframe Control Bar

**Priority:** P0 (Critical)

| Requirement | Description |
|-------------|-------------|
| **REQ-1.2.1** | Group Start Date, End Date, Apply, Reset into cohesive control bar |
| **REQ-1.2.2** | Position quick-select buttons (6M, 1Y, 3Y, 5Y) adjacent to date controls |
| **REQ-1.2.3** | Add visual separator between timeframe controls and other elements |
| **REQ-1.2.4** | Use compact date picker widgets |

**Mockup Concept:**
```
┌─────────────────────────────────────────────────────────────┐
│ Timeframe: [2024-01-01] to [2025-11-30] [Apply] [Reset]     │
│            [6M] [1Y] [3Y] [5Y] [All]                        │
└─────────────────────────────────────────────────────────────┘
```

#### 1.3 Analysis Tabs Redesign

**Priority:** P0 (Critical)

| Requirement | Description |
|-------------|-------------|
| **REQ-1.3.1** | Style analysis tabs as clear, clickable tab buttons |
| **REQ-1.3.2** | Highlight active tab with primary color and underline |
| **REQ-1.3.3** | Add icons to each tab for quick recognition |
| **REQ-1.3.4** | Reduce tab count by grouping related functions |

**Tab Structure:**
| Tab | Icon | Contains |
|-----|------|----------|
| Charts | 📈 | Individual, Comparison, Seasonality |
| Fundamentals | 📊 | Fundamental Analysis data |
| Business | 💼 | Business Analysis, News Search |
| News | 📰 | Market News Blog |
| Analysis | 🎯 | Buffett & CANSLIM |
| SEC | 📋 | SEC Filings |

#### 1.4 Action Buttons Hierarchy

**Priority:** P1 (High)

| Requirement | Description |
|-------------|-------------|
| **REQ-1.4.1** | Style primary actions (Run BA, Extract Tables) with prominent color |
| **REQ-1.4.2** | Group related actions together with visual separators |
| **REQ-1.4.3** | Make Filter Metric section collapsible |
| **REQ-1.4.4** | Add keyboard shortcuts for frequent actions |

**Button Hierarchy:**
- **Primary** (Blue): Run BA, Extract Tables, Analyze
- **Secondary** (Gray): News Search, AI Search
- **Tertiary** (Outline): Clear, Reset, Cancel

---

### 2. Visual Design Modernization

#### 2.1 Typography System

**Priority:** P1 (High)

| Requirement | Description |
|-------------|-------------|
| **REQ-2.1.1** | Implement Segoe UI (Windows native) or Inter font family |
| **REQ-2.1.2** | Define type scale: H1 (24px), H2 (18px), H3 (14px), Body (12px) |
| **REQ-2.1.3** | Use font weights: Bold (600) for headings, Regular (400) for body |
| **REQ-2.1.4** | Ensure minimum 12px font size for readability |

**Type Scale:**
```
Company Name (H1): 24px Bold - "KYNDRYL HOLDINGS, INC."
Section Header (H2): 18px Bold - "BUSINESS SNAPSHOT"
Subsection (H3): 14px SemiBold - "1. OVERVIEW"
Body Text: 12px Regular
Caption/Label: 11px Regular Gray
```

#### 2.2 Color Palette

**Priority:** P1 (High)

| Requirement | Description |
|-------------|-------------|
| **REQ-2.2.1** | Define primary color: Deep Blue (#1a365d) |
| **REQ-2.2.2** | Define accent color: Teal (#0d9488) for positive indicators |
| **REQ-2.2.3** | Define warning color: Amber (#f59e0b) for alerts |
| **REQ-2.2.4** | Define error color: Red (#dc2626) for negative indicators |
| **REQ-2.2.5** | Use neutral grays for borders (#e5e7eb) and backgrounds (#f9fafb) |

**Color System:**
| Purpose | Color | Hex |
|---------|-------|-----|
| Primary | Deep Blue | #1a365d |
| Primary Light | Light Blue | #3b82f6 |
| Accent/Success | Teal | #0d9488 |
| Warning | Amber | #f59e0b |
| Error/Negative | Red | #dc2626 |
| Background | Off-White | #f9fafb |
| Border | Light Gray | #e5e7eb |
| Text Primary | Dark Gray | #1f2937 |
| Text Secondary | Medium Gray | #6b7280 |

#### 2.3 Spacing & Whitespace

**Priority:** P1 (High)

| Requirement | Description |
|-------------|-------------|
| **REQ-2.3.1** | Define spacing scale: 4px, 8px, 12px, 16px, 24px, 32px |
| **REQ-2.3.2** | Apply consistent padding to all containers (16px default) |
| **REQ-2.3.3** | Add 8px gap between related elements |
| **REQ-2.3.4** | Add 24px gap between sections |
| **REQ-2.3.5** | Ensure minimum 40px touch target for buttons |

---

### 3. Content Presentation

#### 3.1 Business Snapshot Data Grid

**Priority:** P0 (Critical)

| Requirement | Description |
|-------------|-------------|
| **REQ-3.1.1** | Display key metrics in structured 2-column grid |
| **REQ-3.1.2** | Add icons for each metric type |
| **REQ-3.1.3** | Color-code values (green=positive, red=negative) |
| **REQ-3.1.4** | Make grid responsive to panel width |

**Mockup Concept:**
```
┌─────────────────────────────────────────────┐
│ BUSINESS SNAPSHOT                           │
├─────────────────────┬───────────────────────┤
│ 💰 Market Cap       │ $5.2B                 │
│ 📈 Revenue Growth   │ +12.3% ↑             │
│ 💵 Dividend Yield   │ 2.1%                  │
│ 📊 P/E Ratio        │ 18.5                  │
│ 🏢 Sector           │ Technology            │
│ 🏭 Industry         │ IT Services           │
└─────────────────────┴───────────────────────┘
```

#### 3.2 Report Text Formatting

**Priority:** P1 (High)

| Requirement | Description |
|-------------|-------------|
| **REQ-3.2.1** | Render markdown formatting in report text |
| **REQ-3.2.2** | Style section headings with larger font and color |
| **REQ-3.2.3** | Render bullet points with proper indentation |
| **REQ-3.2.4** | Add paragraph spacing (12px between paragraphs) |
| **REQ-3.2.5** | Support code blocks with monospace font |

#### 3.3 Language Toggle

**Priority:** P2 (Medium)

| Requirement | Description |
|-------------|-------------|
| **REQ-3.3.1** | Add language toggle button (EN/中文) in report header |
| **REQ-3.3.2** | Store language preference in settings |
| **REQ-3.3.3** | Apply language filter to AI-generated content |

---

### 4. Toolbar & Status Bar Refinement

#### 4.1 Top Toolbar Reorganization

**Priority:** P1 (High)

| Requirement | Description |
|-------------|-------------|
| **REQ-4.1.1** | Group global controls (List, Load, Create) on left side |
| **REQ-4.1.2** | Group chart controls (D, W, M, Multi-TF) in center |
| **REQ-4.1.3** | Place utility buttons (URLs, Guide, Tips) on right side |
| **REQ-4.1.4** | Add visual separators between groups |
| **REQ-4.1.5** | Reduce toolbar to single row where possible |

**Toolbar Layout:**
```
┌──────────────────────────────────────────────────────────────────────────┐
│ [List ▾] [◀][▶] [Load] │ [D][W][M] [Multi-TF] [SC] │ [URLs▾][Guide▾][?] │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 4.2 Bottom Action Bar

**Priority:** P1 (High)

| Requirement | Description |
|-------------|-------------|
| **REQ-4.2.1** | Separate action buttons from status message |
| **REQ-4.2.2** | Add clear labels or tooltips to all buttons |
| **REQ-4.2.3** | Replace "A-Z" with "Sort" label |
| **REQ-4.2.4** | Replace "?" with "Help" or info icon with tooltip |
| **REQ-4.2.5** | Style status message as distinct notification area |

**Bottom Bar Layout:**
```
┌──────────────────────────────────────────────────────────────────────────┐
│ [⬇ Download] [📊 Visualize] [📄 Report] [📈 Compare] │ [Sort] [Help]    │
├──────────────────────────────────────────────────────────────────────────┤
│ ✓ Saved Business Analysis markdown for KD                    [Force DL] │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Layout Restructuring (Week 1-2)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Combine ticker panels into tabbed view | P0 | 3 days | None |
| Implement timeframe control bar | P0 | 2 days | None |
| Redesign analysis tabs | P0 | 2 days | None |
| Reorganize toolbar groups | P1 | 2 days | None |

### Phase 2: Visual Design (Week 2-3)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Implement typography system | P1 | 1 day | Phase 1 |
| Apply color palette | P1 | 1 day | Phase 1 |
| Add spacing and whitespace | P1 | 2 days | Phase 1 |
| Style buttons with hierarchy | P1 | 1 day | Color palette |

### Phase 3: Content Presentation (Week 3-4)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Create Business Snapshot grid | P0 | 2 days | Phase 2 |
| Implement markdown rendering | P1 | 2 days | Phase 2 |
| Add collapsible filter panel | P1 | 1 day | Phase 1 |
| Language toggle (optional) | P2 | 2 days | Phase 3 |

### Phase 4: Polish & Testing (Week 4)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Refine bottom action bar | P1 | 1 day | Phase 2 |
| Add tooltips to all controls | P1 | 1 day | All phases |
| User testing and feedback | P0 | 2 days | All phases |
| Bug fixes and refinements | P0 | 2 days | Testing |

---

## Technical Considerations

### Tkinter Limitations

- **Custom styling**: Use `ttk.Style()` for consistent theming
- **Fonts**: Limited to system fonts; Segoe UI available on Windows
- **Colors**: Apply via style configuration and widget options
- **Markdown rendering**: Use `tkinter.Text` with tags for formatting

### Recommended Approach

1. Create `gui_styles.py` module for centralized styling
2. Define color constants and font configurations
3. Use `ttk` widgets where possible for native look
4. Implement custom compound widgets for complex UI elements

### Code Structure

```
gui.py                    # Main GUI class
├── gui_styles.py         # Style definitions and theme
├── widgets/
│   ├── tabbed_panel.py   # Combined ticker panel
│   ├── timeframe_bar.py  # Date range controls
│   ├── snapshot_grid.py  # Business snapshot display
│   └── markdown_view.py  # Formatted text display
```

---

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Tkinter styling limitations | Medium | High | Use custom widgets, consider ttkbootstrap |
| Breaking existing functionality | High | Medium | Comprehensive testing, feature flags |
| User resistance to change | Medium | Low | Gradual rollout, user feedback sessions |
| Performance degradation | Medium | Low | Profile and optimize rendering |

---

## Success Criteria

### Must Have (MVP)

- [x] Consolidated tabbed ticker panel
- [x] Cohesive timeframe control bar
- [x] Styled analysis tabs with active indicator
- [x] Consistent color palette applied
- [x] Improved spacing throughout

### Should Have

- [x] Business Snapshot data grid
- [x] Markdown rendering in reports
- [x] Collapsible filter panel
- [x] Reorganized toolbars

### Nice to Have

- [x] Language toggle
- [x] Keyboard shortcuts
- [x] Dark mode support

---

## Workflow-Driven Design

The GUI improvements are specifically designed to support the **5-Phase Stock Research Workflow** documented in USER_GUIDE.md. Each phase has specific UI requirements to streamline the research process.

### Phase 1: Discovery & Screening

**Workflow Goal:** Identify potential investment candidates

| Workflow Step | UI Requirement | Implementation |
|---------------|----------------|----------------|
| Market Overview | Quick access to Market news | **REQ-WF-1.1**: Add prominent "Market News" button in toolbar |
| Browse Lists | Easy list navigation | **REQ-WF-1.2**: ◀/▶ buttons visible, list dropdown prominent |
| Quick Visual Scan | Fast chart gallery access | **REQ-WF-1.3**: D/W/M buttons clearly grouped |
| External Screening | URLs menu accessible | **REQ-WF-1.4**: URLs dropdown in consistent location |

**UI Enhancements:**
- Add "🔍 Discovery" quick-action panel with:
  - Market News button
  - Finviz Screener shortcut
  - AI Search field
- Highlight curated lists (mag7, sp500_top50, etc.) in dropdown

### Phase 2: Technical Analysis

**Workflow Goal:** Evaluate price action and chart patterns

| Workflow Step | UI Requirement | Implementation |
|---------------|----------------|----------------|
| Multi-Timeframe | One-click gallery | **REQ-WF-2.1**: Multi-TF button prominent in toolbar |
| StockCharts | SC buttons accessible | **REQ-WF-2.2**: SC and SC-Line grouped together |
| Seasonality | Tab easily accessible | **REQ-WF-2.3**: 📆 Seasonal tab with icon |
| Comparison | Multi-select + Compare | **REQ-WF-2.4**: Compare button visible in action bar |

**UI Enhancements:**
- Group chart buttons: `[D] [W] [M] | [Multi-TF] [Lines] | [SC] [SC-Line]`
- Add "📊 Technical" section label in toolbar
- Timeframe control bar positioned above chart tabs

### Phase 3: Fundamental Analysis

**Workflow Goal:** Understand the business and valuation

| Workflow Step | UI Requirement | Implementation |
|---------------|----------------|----------------|
| Quick Metrics | Fundamentals tab | **REQ-WF-3.1**: 📋 Fundamentals tab with filter |
| Business Analysis | BA tab with Run BA | **REQ-WF-3.2**: 💼 Business tab with prominent Run BA button |
| News & Sentiment | News search accessible | **REQ-WF-3.3**: Stock news button in action bar |
| Investment Framework | Buffett & CANSLIM | **REQ-WF-3.4**: 🎯 Analysis tab with Analyze button |

**UI Enhancements:**
- Add "Business Snapshot" grid at top of Fundamentals tab showing:
  - 💰 Market Cap | 📈 Revenue Growth | 💵 Dividend Yield
  - 📊 P/E Ratio | 🏢 Sector | 🏭 Industry
- Make "Run BA" button primary style (prominent color)
- Add filter presets: "Value Metrics", "Growth Metrics", "Dividend Metrics"

### Phase 4: SEC Filing Analysis

**Workflow Goal:** Verify financials and identify risks

| Workflow Step | UI Requirement | Implementation |
|---------------|----------------|----------------|
| 10-K Study | Quick access from BA tab | **REQ-WF-4.1**: 10K Study button in Business tab |
| 10-Q Study | Quick access from BA tab | **REQ-WF-4.2**: 10-Q Study button in Business tab |
| Extract Tables | SEC tab with extraction | **REQ-WF-4.3**: 📑 SEC tab with clear workflow |
| Export to Excel | Export button visible | **REQ-WF-4.4**: Export button in SEC tab |

**UI Enhancements:**
- Add "📑 SEC Quick Actions" panel in Business tab:
  - [10-K Study] [10-Q Study] [Extract Tables]
- SEC tab shows clear workflow: Ticker → Form Type → Extract → View → Export
- Add financial table highlighting for key metrics

### Phase 5: Decision & Monitoring

**Workflow Goal:** Make informed decision and track position

| Workflow Step | UI Requirement | Implementation |
|---------------|----------------|----------------|
| Build Thesis | Clipboard AI accessible | **REQ-WF-5.1**: 📋 button in action bar |
| Add to Watch List | Right-click or button | **REQ-WF-5.2**: "Add to Watch" in context menu |
| Ongoing Monitoring | Watch list prominent | **REQ-WF-5.3**: ⭐ Watch tab with count badge |
| Portfolio Review | Visualize all button | **REQ-WF-5.4**: 📊Visualize button in action bar |

**UI Enhancements:**
- Watch list tab shows count: "⭐ Watch (12)"
- Add "Portfolio Actions" group in action bar:
  - [📊 Visualize] [📄 Report] [📈 Compare]
- Status bar shows last action and timestamp

---

## Workflow Quick-Access Panel

**Priority:** P1 (High)

Add a collapsible "Research Workflow" panel that guides users through the 5-phase process:

```
┌─────────────────────────────────────────────────────────────────┐
│ 📋 Research Workflow                                    [−]     │
├─────────────────────────────────────────────────────────────────┤
│ Phase 1: Discovery    [Market News] [Browse Lists] [Screener]   │
│ Phase 2: Technical    [Multi-TF] [SC Charts] [Seasonality]      │
│ Phase 3: Fundamental  [Run BA] [Fundamentals] [News]            │
│ Phase 4: SEC Filing   [10-K Study] [10-Q Study] [Extract]       │
│ Phase 5: Decision     [Add to Watch] [Compare] [Visualize]      │
└─────────────────────────────────────────────────────────────────┘
```

**Requirements:**
| Requirement | Description |
|-------------|-------------|
| **REQ-WF-P.1** | Collapsible panel below toolbar |
| **REQ-WF-P.2** | Each phase has 3 most-used action buttons |
| **REQ-WF-P.3** | Clicking button executes action AND highlights current phase |
| **REQ-WF-P.4** | Panel state (expanded/collapsed) persists in settings |

---

## Quick Research Checklists Integration

Add checklist functionality to support the documented checklists:

### 5-Minute Stock Check
```
☐ Open D chart - Trending?
☐ Fundamentals - P/E reasonable?
☐ Stock news - Red flags?
```

### 30-Minute Deep Dive
```
☐ Multi-TF charts - Trend alignment?
☐ Seasonality - Good entry timing?
☐ Run BA - Business quality?
☐ Buffett & CANSLIM - Investment grade?
☐ 10-Q Study - Recent quarter healthy?
```

**Requirements:**
| Requirement | Description |
|-------------|-------------|
| **REQ-CL.1** | Add "Checklist" dropdown in toolbar |
| **REQ-CL.2** | Selecting checklist opens floating checklist panel |
| **REQ-CL.3** | Clicking checklist item navigates to relevant tab/action |
| **REQ-CL.4** | Checklist state resets when ticker changes |

---

## Appendix

### A. Current vs. Proposed Layout

**Current:**
```
┌─────────────────────────────────────────────────────────────────────┐
│ [Toolbar Row 1 - Many controls mixed together]                      │
│ [Toolbar Row 2 - More controls]                                     │
├────────┬────────┬───────────────────────────────────────────────────┤
│ Avail  │ Watch  │ Chart Display                                     │
│ Ticker │ List   │ [Many tabs in a row]                              │
│        │        │ [Date controls scattered]                         │
│        │        │ [Content area]                                    │
├────────┴────────┴───────────────────────────────────────────────────┤
│ [Action buttons] [Status message mixed in]                          │
└─────────────────────────────────────────────────────────────────────┘
```

**Proposed:**
```
┌─────────────────────────────────────────────────────────────────────┐
│ [Global] │ [Chart Controls] │ [Utilities]                           │
├──────────┼──────────────────────────────────────────────────────────┤
│ Tickers  │ [Charts] [Fundamentals] [Business] [News] [SEC]          │
│ [Avail]  │ ─────────────────────────────────────────────────────────│
│ [Watch]  │ Timeframe: [Start] to [End] [6M][1Y][3Y][5Y]             │
│          │ ─────────────────────────────────────────────────────────│
│ Filter:  │ [Content Area - Well formatted]                          │
│ [____]   │                                                          │
│          │                                                          │
├──────────┴──────────────────────────────────────────────────────────┤
│ [Actions]                                        │ Status: ✓ Saved  │
└─────────────────────────────────────────────────────────────────────┘
```

### B. Color Reference

![Color Palette](https://via.placeholder.com/400x100/1a365d/ffffff?text=Primary)
![Color Palette](https://via.placeholder.com/400x100/0d9488/ffffff?text=Accent)
![Color Palette](https://via.placeholder.com/400x100/f9fafb/1f2937?text=Background)

---

*Document Version: 1.0*  
*Last Updated: November 30, 2025*
