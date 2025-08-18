import os
import sys
from lxml import html

def test_html_import():
    """Test the lxml.html import and text extraction functionality"""
    test_html = """
    <html>
        <body>
            <h1>Test Header</h1>
            <p>This is a test paragraph.</p>
            <div>
                <p>This is another paragraph inside a div.</p>
            </div>
        </body>
    </html>
    """
    
    try:
        # Test html.fromstring
        parsed = html.fromstring(test_html)
        text_content = parsed.text_content()
        
        print("HTML parsing successful!")
        print(f"Extracted text: {text_content}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_html_import()
