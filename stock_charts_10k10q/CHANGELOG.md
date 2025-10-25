# Changelog

## [Unreleased]

### Added

- Support for Chinese ticker comments in stock charts
- GUI toolbar buttons in `gui.py`:
  - Open Ticker File (Notepad++) to open `ticker_lists.py` in Notepad++ or default editor.
  - Copy Current List to copy `self.current_tickers` to clipboard (one per line).
- Methods in `gui.py`:
  - `_open_ticker_list_in_notepadpp()`
  - `_copy_current_list_to_clipboard()`

### Changed

- Updated ticker comments in `ticker_lists.py` to use native Chinese characters
- Modified annotation logic in `multi-ticker_comparison.py` to display Chinese comments when available
- Improved font configuration to support Chinese characters in matplotlib plots

### Improvements

- Enhanced readability of stock chart annotations
- Added more descriptive comments for Chinese stocks

### Fixed

- Resolved `NameError: name 'A' is not defined` by renaming invalid identifiers in `ticker_lists.py`:
  - `A.magic_formula_12_22_25_stocks` -> `magic_formula_12_22_25_stocks`
  - `A.BTC_etfs_stocks` -> `BTC_etfs_stocks`
  - `A.btc_related_stocks` -> `btc_related_stocks`
  - `A.AI_ticker_extractor_tickers` -> `AI_ticker_extractor_tickers`
