import json
import os
import webbrowser

def generate_chart_html(tickers: list[str], columns: int, output_filename: str = "stock_charts.html",tf="d"):
    """
    Generates a self-contained HTML file to display a grid of stock charts.

    Args:
        tickers (list[str]): A list of stock ticker symbols.
        columns (int): The number of columns for the chart grid.
        output_filename (str): The name of the output HTML file.
    """

    # Safely format the Python list of tickers into a JavaScript array string.
    # For example: ['AAPL', 'GOOG'] becomes '["AAPL", "GOOG"]'
    tickers_js_array = json.dumps(tickers)

    # The HTML template is structured with f-strings to inject our dynamic values.
    # The CSS and JavaScript are embedded directly into the template.
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dynamic Stock Chart Grid</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f4f4f9;
            color: #333;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 0;
            padding: 15px;
        }}
        .grid-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 0;
            justify-content: center;
        }}
        .chart-container {{
            background-color: #ffffff;
            border-radius: 0;
            padding: 0;
            text-align: center;
            border: 1px solid #f0f0f0;
            min-width: 300px;
        }}
        .chart-container h2 {{
            margin: 5px 0 5px 0;
            font-size: 1.1em;
            color: #34495e;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
        }}
    </style>
</head>
<body>
    
    <h1>Stock Charts</h1>

    <div class="grid-container" id="chart-grid"></div>

    <script>
        // --- Configuration injected by Python ---
        const tickers = {tickers_js_array};
        const columns = {columns};
        const tf = {tf};
        
        // --- Dynamic Chart Generation Logic ---
        const gridContainer = document.getElementById('chart-grid');
        const flexBasis = 100 / columns;
        
        // Dynamically create and inject the style rule for the columns
        const style = document.createElement('style');
        style.innerHTML = `.chart-container {{ flex: 1 1 ${{flexBasis}}%; }}`;
        document.head.appendChild(style);

        let allChartsHTML = '';
        for (const ticker of tickers) {{
            const chartUrl = `https://charts2-node.finviz.com/chart.ashx?cs=&t=${{ticker.toUpperCase()}}&tf=${{tf}}&s=linear&pm=240&am=1200&ct=candle_stick&tm=d&o[0][ot]=sma&o[0][op]=50&o[0][oc]=FF8F33C6&o[1][ot]=sma&o[1][op]=200&o[1][oc]=DCB3326D&o[2][ot]=patterns&o[2][op]=&o[2][oc]=000`;
            
            allChartsHTML += `
                <div class="chart-container">
                    <h2>${{ticker.toUpperCase()}}</h2>
                    <img src="${{chartUrl}}" alt="${{ticker.toUpperCase()}} Stock Chart">
                </div>
            `;
        }}
        // Insert the complete HTML string into the container at once for better performance
        gridContainer.innerHTML = allChartsHTML;
    </script>

</body>
</html>
"""

    # Write the generated HTML string to the specified file
    try:
        with open(output_filename, "w") as f:
            f.write(html_template)
        print(f"✅ Successfully generated '{output_filename}'")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # Define your list of tickers and desired number of columns here
    my_tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM',
        'V', 'JNJ', 'UNH', 'WMT', 'PG', 'MA'
    ]
    
    number_of_columns = 4

    # Call the function to generate the HTML file
    output_filename = "stock_charts.html"
    generate_chart_html(my_tickers, number_of_columns, output_filename)
    
    # Open the generated HTML in Chrome
    webbrowser.register('chrome', None, webbrowser.BackgroundBrowser('C:/Program Files/Google/Chrome/Application/chrome.exe'))
    webbrowser.get('chrome').open(os.path.abspath(output_filename))
