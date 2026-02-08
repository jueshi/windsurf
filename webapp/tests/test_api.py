"""
API endpoint tests for Stock Toolbox Web.
Run with: pytest webapp/tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthCheck:
    """Test basic app functionality."""
    
    def test_homepage_loads(self, client):
        """Test that the homepage loads successfully."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Dashboard" in response.text
    
    def test_static_files_accessible(self, client):
        """Test that static files route is configured."""
        # This should return 404 for non-existent file, not 500
        response = client.get("/static/nonexistent.css")
        assert response.status_code == 404


class TestTickerEndpoints:
    """Test ticker-related endpoints."""
    
    def test_get_ticker_lists(self, client):
        """Test fetching ticker lists."""
        response = client.get("/tickers/list")
        assert response.status_code == 200
    
    def test_search_empty_query(self, client):
        """Test search with empty query returns empty."""
        response = client.get("/tickers/search?q=")
        assert response.status_code == 200
        assert response.text == ""
    
    def test_search_with_query(self, client):
        """Test search with a query."""
        response = client.get("/tickers/search?q=AAPL")
        assert response.status_code == 200


class TestChartEndpoints:
    """Test chart data endpoints."""
    
    @patch('webapp.routers.charts.data_manager')
    def test_chart_data_no_data(self, mock_dm, client):
        """Test chart endpoint when no data exists."""
        mock_dm.load_data.return_value = None
        mock_dm.update_data.return_value = None
        
        response = client.get("/charts/data/INVALID")
        assert response.status_code == 404
        assert "error" in response.json()
    
    @patch('webapp.routers.charts.data_manager')
    def test_chart_data_with_data(self, mock_dm, client):
        """Test chart endpoint with valid data."""
        import pandas as pd
        from datetime import datetime
        
        # Create mock data
        mock_data = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [102.0, 103.0],
            'Low': [99.0, 100.0],
            'Close': [101.0, 102.0],
            'Volume': [1000000, 1100000]
        }, index=pd.to_datetime(['2024-01-01', '2024-01-02']))
        mock_data.index.name = 'Date'
        
        mock_dm.load_data.return_value = mock_data
        
        response = client.get("/charts/data/AAPL")
        assert response.status_code == 200
        
        data = response.json()
        assert data['ticker'] == 'AAPL'
        assert 'dates' in data
        assert 'close' in data
        assert len(data['dates']) == 2


class TestValidation:
    """Test input validation."""
    
    def test_ticker_validation_valid(self):
        """Test valid ticker symbols."""
        from webapp.routers.tickers import validate_ticker_symbol
        
        valid_tickers = ['AAPL', 'MSFT', 'A', 'GOOGL', 'BRK.A', 'BRK.B']
        for ticker in valid_tickers:
            is_valid, result = validate_ticker_symbol(ticker)
            assert is_valid, f"{ticker} should be valid"
            assert result == ticker.upper()
    
    def test_ticker_validation_invalid(self):
        """Test invalid ticker symbols."""
        from webapp.routers.tickers import validate_ticker_symbol
        
        invalid_tickers = ['', '123', 'TOOLONG', 'A1B', '@#$', 'A B']
        for ticker in invalid_tickers:
            is_valid, result = validate_ticker_symbol(ticker)
            assert not is_valid, f"{ticker} should be invalid"


class TestSchemas:
    """Test Pydantic schemas and utilities."""
    
    def test_filter_fundamental_data(self):
        """Test fundamental data filtering."""
        from webapp.schemas import filter_fundamental_data
        
        raw_data = {
            'longName': 'Apple Inc.',
            'marketCap': 3000000000000,  # 3T
            'trailingPE': 28.5,
            'dividendYield': 0.005,  # 0.5%
            'randomField': 'should be filtered out',
            'anotherRandom': 12345
        }
        
        filtered = filter_fundamental_data(raw_data)
        
        assert 'Company Name' in filtered
        assert filtered['Company Name'] == 'Apple Inc.'
        assert 'Market Cap' in filtered
        assert 'T' in filtered['Market Cap']  # Should be formatted as trillions
        assert 'randomField' not in filtered
        assert 'anotherRandom' not in filtered
    
    def test_format_metric_value(self):
        """Test metric value formatting."""
        from webapp.schemas import format_metric_value
        
        # Test percentage
        assert '%' in format_metric_value('dividendYield', 0.025)
        
        # Test large numbers
        assert 'B' in format_metric_value('marketCap', 5000000000)
        assert 'M' in format_metric_value('marketCap', 500000000)
        
        # Test ratios
        result = format_metric_value('trailingPE', 25.5)
        assert result == '25.50'


class TestCache:
    """Test caching functionality."""
    
    def test_simple_cache(self):
        """Test SimpleCache class."""
        from webapp.data_manager import SimpleCache
        import time
        
        cache = SimpleCache(default_ttl=1)  # 1 second TTL
        
        # Test set and get
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        # Test expiry
        cache.set('key2', 'value2', ttl=0)  # Immediate expiry
        time.sleep(0.1)
        assert cache.get('key2') is None
        
        # Test remove
        cache.set('key3', 'value3')
        cache.remove('key3')
        assert cache.get('key3') is None
        
        # Test clear
        cache.set('key4', 'value4')
        cache.clear()
        assert cache.get('key4') is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
