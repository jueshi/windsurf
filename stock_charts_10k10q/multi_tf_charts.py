import json
import os
import webbrowser

def generate_multi_timeframe_chart_html(
    tickers: list[str], 
    output_filename: str = "multi_timeframe_charts.html"
):
    """
    Generates a self-contained HTML file to display Daily, Weekly, and Monthly 
    charts for each stock, with each stock's charts on one row.

    Args:
        tickers (list[str]): A list of stock ticker symbols.
        output_filename (str): The name of the output HTML file.
    """

    tickers_js_array = json.dumps(tickers)

    # Note: For multi-timeframe per row, we'll hardcode 3 columns for daily/weekly/monthly
    # The CSS flex property will be adjusted to 33.33% for the chart-container.
    
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Timeframe Stock Chart Gallery</title>
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
        .stock-row {{
            display: flex;
            flex-wrap: wrap; /* Allows charts to wrap on smaller screens if they don't fit 3-wide */
            justify-content: center;
            margin-bottom: 10px; /* Space between rows for different stocks */
            border: 1px solid #e0e0e0;
            background-color: #ffffff;
            border-radius: 5px;
            overflow: hidden; /* Ensures inner borders don't spill */
        }}
        .stock-row h2.stock-ticker-header {{
            width: 100%; /* Make the stock ticker span full width of the row */
            text-align: center;
            margin: 10px 0 0px 0;
            font-size: 1.5em;
            color: #2c3e50;
            background-color: #f8f8f8;
            padding: 5px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .chart-container {{
            /* Hardcoded to 33.33% for 3 charts per row */
            flex: 1 1 calc(100% / 3); 
            background-color: #ffffff;
            padding: 0;
            text-align: center;
            border-right: 1px solid #f0f0f0; /* Separator between charts */
            min-width: 320px; /* Minimum width for each chart */
        }}
        .chart-container:last-child {{
            border-right: none; /* No border on the last chart in a row */
        }}
        .chart-container h3 {{ /* Changed from h2 to h3 for chart titles */
            margin: 5px 0 5px 0;
            font-size: 1.0em;
            color: #34495e;
        }}
        img {{
            width: 100%;
            height: 300px;
            object-fit: contain;
            display: block;
            background-color: #ffffff;
        }}
        @media (max-width: 960px) {{ /* Responsive adjustments for smaller screens */
            .chart-container {{
                flex: 1 1 50%; /* Two charts per row on medium screens */
            }}
            .stock-row h2.stock-ticker-header {{
                font-size: 1.3em;
            }}
        }}
        @media (max-width: 640px) {{ /* Single chart per row on small screens */
            .chart-container {{
                flex: 1 1 100%;
                border-right: none;
                border-bottom: 1px solid #f0f0f0;
            }}
            .chart-container:last-child {{
                border-bottom: none;
            }}
        }}
    </style>
</head>
<body>
    
    <h1>Multi-Timeframe Stock Chart Gallery</h1>

    <div class="grid-container" id="chart-grid"></div>

    <script>
        // --- Configuration injected by Python ---
        const tickers = {tickers_js_array};
        const timeFrames = [ // Define all time frames to plot
            {{ tf: "d", label: "Daily", range: "" }},
            {{ tf: "w", label: "Weekly", range: "&r=y2" }},
            {{ tf: "m", label: "Monthly", range: "&r=max" }}
        ];

        const gridContainer = document.getElementById('chart-grid');
        
        let allContentHTML = '';

        for (const ticker of tickers) {{
            // Start a new row for each stock
            allContentHTML += `
                <div class="stock-row">
                    <h2 class="stock-ticker-header">${{ticker.toUpperCase()}}</h2>
            `;
            
            for (const tfConfig of timeFrames) {{
                const chartUrl = `https://charts2-node.finviz.com/chart.ashx?cs=&t=${{ticker.toUpperCase()}}&tf=${{tfConfig.tf}}&s=linear&pm=240&am=1200&ct=candle_stick&tm=d${{tfConfig.range}}&o[0][ot]=sma&o[0][op]=50&o[0][oc]=FF8F33C6&o[1][ot]=sma&o[1][op]=200&o[1][oc]=DCB3326D&o[2][ot]=patterns&o[2][op]=&o[2][oc]=000`;
                
                allContentHTML += `
                    <div class="chart-container">
                        <h3>${{tfConfig.label}}</h3>
                        <img src="${{chartUrl}}" alt="${{ticker.toUpperCase()}} ${{tfConfig.label}} Stock Chart">
                    </div>
                `;
            }}
            
            allContentHTML += `</div>`; // Close the stock-row
        }}
        gridContainer.innerHTML = allContentHTML;
    </script>

</body>
</html>
"""

    try:
        with open(output_filename, "w") as f:
            f.write(html_template)
        print(f"✅ Successfully generated '{output_filename}' with multi-timeframe charts.")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    my_tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META',
        'JPM', 'V', 'JNJ', 'UNH', 'WMT', 'PG', 'MA'
    ]
    
    # No 'columns' or 'time_frame' parameter needed for this function
    # as the layout is fixed to 3 timeframes per row.

    generate_multi_timeframe_chart_html(
        tickers=my_tickers
    )

    
    try:
        # Open the generated HTML in Edge
        output_filename = "multi_timeframe_charts.html"
        webbrowser.register('edge', None, webbrowser.BackgroundBrowser(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'))
        webbrowser.get('edge').open(os.path.abspath(output_filename))
        print(f"✅ Opened '{output_filename}' in Edge")
    except Exception as e:
        print(f"❌ Error opening file in Edge: {e}")
