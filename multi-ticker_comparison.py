"""
Stock Ticker Multi-Comparison Visualization

This script provides a comprehensive visualization of stock performance 
for multiple tickers across different time frames.

Changelog:
- 2025-01-05: v1.8.0
  * Implemented separate data padding pass
  * Enhanced data consistency across all time frames
  * Added optional CSV export for padded data
  * Improved global start date alignment

- 2025-01-05: v1.7.0
  * Implemented global earliest start date alignment
  * Added padding for tickers using their oldest available price
  * Enhanced data consistency across different tickers
  * Improved historical data representation

- 2025-01-05: v1.6.0
  * Restored 1-year and 5-year charts
  * Fixed data filtering for specific time frames
  * Ensured consistent plotting across all time periods

- 2025-01-05: v1.5.0
  * Implemented true line charts without any fill
  * Ensured clean, precise line representation of stock performance
  * Refined plot styling for maximum clarity

- 2025-01-05: v1.4.0
  * Removed area fill under the curve
  * Simplified plot styling
  * Focused on clean line representation of stock performance

- 2025-01-05: v1.3.0
  * Added padding for tickers with missing data periods
  * Used first trading day price to fill gaps in historical data
  * Improved data alignment across different tickers
  * Enhanced visualization of comparative stock performance

- 2025-01-05: v1.2.0
  * Added support for full historical data comparison
  * Extended visualization to include all-time stock performance
  * Adjusted subplot layout to accommodate three time frames
  * Improved handling of tickers with different data ranges

- 2025-01-05: v1.1.0
  * Added support for 1-year and 5-year stock price performance comparison
  * Implemented subplot visualization for easy comparison
  * Enhanced data preprocessing and normalization
  * Improved error handling and data validation

- 2025-01-03: v1.0.0
  * Initial implementation of stock price comparison script
  * Basic data loading and visualization functionality

Dependencies:
- pandas
- matplotlib
- data_rechiever module

Usage:
Run the script to generate a comparative plot of stock performances.
Customize the stock_tickers list to compare different stocks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from data_rechiever import StockDataManager
import os
from ticker_lists import *

# Add support for Chinese characters
plt.rcParams['font.family'] = 'Microsoft YaHei'  # Windows Chinese font
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # Fallback fonts
plt.rcParams['axes.unicode_minus'] = False  # Resolve minus sign display issue

class StockDataManagerExt(StockDataManager):
    '''moved the plot_multiple_tickers function to root class'''
    pass
# Create a list of tickers
stock_tickers = ['AAPL', 'MSFT', 'GOOG', 'TSLA','ALAB','INTC','NVDA','AVGO','AMD','AMZN']

# Process the tickers
stock_manager = StockDataManagerExt()

# stock_manager.plot_multiple_tickers(tickers=stock_tickers)
stock_manager.plot_multiple_tickers(tickers=chinese_stocks_tickers)
# stock_manager.plot_multiple_tickers(tickers=FUNDS_stocks)
# stock_manager.plot_multiple_tickers(tickers=China_FUNDS_stocks)
