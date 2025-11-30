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

- [ ] Consolidated tabbed ticker panel
- [ ] Cohesive timeframe control bar
- [ ] Styled analysis tabs with active indicator
- [ ] Consistent color palette applied
- [ ] Improved spacing throughout

### Should Have

- [ ] Business Snapshot data grid
- [ ] Markdown rendering in reports
- [ ] Collapsible filter panel
- [ ] Reorganized toolbars

### Nice to Have

- [ ] Language toggle
- [ ] Keyboard shortcuts
- [ ] Dark mode support

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
