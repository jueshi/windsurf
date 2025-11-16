import json
import os
import webbrowser

def generate_multi_timeframe_stockcharts_html(
    tickers: list[str], 
    output_filename: str = "multi_timeframe_stockcharts.html"
):
    """
    Generates a self-contained HTML file to display Daily, Weekly, and Monthly 
    charts for each stock using StockCharts.com URLs, with each stock's 
    charts on one row.

    Args:
        tickers (list[str]): A list of stock ticker symbols.
        output_filename (str): The name of the output HTML file.
    """

    tickers_js_array = json.dumps(tickers)

    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Timeframe StockCharts.com Gallery</title>
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
            width: 100%; /* Use full width */
            margin: 0 auto;
            padding: 20px;
            display: flex; /* Use flexbox for the main container too */
            flex-direction: column; /* Stack rows vertically */
            gap: 20px; /* Space between stock rows */
        }}
        .stock-row {{
            display: flex;
            flex-wrap: wrap; 
            justify-content: center;
            border: 1px solid #e0e0e0;
            background-color: #ffffff;
            border-radius: 5px;
            overflow: hidden; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .stock-row h2.stock-ticker-header {{
            width: 100%; 
            text-align: center;
            margin: 0; /* Remove top/bottom margin to tightly fit */
            font-size: 1.5em;
            color: #2c3e50;
            background-color: #f8f8f8;
            padding: 10px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .chart-container {{
            flex: 1 1 calc(100% / 3); 
            background-color: #ffffff;
            padding: 0;
            text-align: center;
            border-right: 1px solid #f0f0f0; 
            min-width: 0; /* Allow charts to shrink as needed */
            display: flex; /* Flexbox for title and image inside container */
            flex-direction: column;
            align-items: center;
            justify-content: flex-start; /* Push title to top */
        }}
        .chart-container:last-child {{
            border-right: none; 
        }}
        .chart-container h3 {{ 
            margin: 10px 0 5px 0; /* More space above title */
            font-size: 1.0em;
            color: #34495e;
            font-weight: 600;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            border-top: 1px solid #f0f0f0; /* Separator between title and image */
        }}
        /* Responsive adjustments */
        @media (max-width: 992px) {{ 
            .chart-container {{
                flex: 1 1 50%; /* Two charts per row on tablets */
            }}
            .chart-container:nth-child(even) {{ /* No right border on second chart in 2-col layout */
                border-right: none;
            }}
            .chart-container:nth-child(odd) {{ /* Keep border on first chart in 2-col layout */
                border-right: 1px solid #f0f0f0;
            }}
            .stock-row h2.stock-ticker-header {{
                font-size: 1.3em;
            }}
        }}
        @media (max-width: 680px) {{ /* Single chart per row on phones */
            .chart-container {{
                flex: 1 1 100%;
                border-right: none;
            }}
            .chart-container:not(:last-child) {{ /* Add bottom border between stacked charts */
                border-bottom: 1px solid #f0f0f0;
            }}
            img {{
                border-top: none; /* Remove top border on image when stacking */
            }}
            .chart-container h3 {{
                border-bottom: 1px solid #f0f0f0; /* Add border under title */
                width: 100%;
                padding-bottom: 5px;
            }}
        }}
    </style>
</head>
<body>
    
    <h1>Multi-Timeframe StockCharts.com Gallery</h1>

    <div class="grid-container" id="chart-grid"></div>

    <script>
        const tickers = {tickers_js_array};
        
        // Define time frames with StockCharts.com specific parameters
        const timeFrames = [
            // Daily Chart: 1 year range
            {{ p: "D", label: "Daily (1yr)", yr: 1, mn: 0, dy: 0, i: "t3784817951c" }}, 
            // Weekly Chart: 5 year range
            {{ p: "W", label: "Weekly (5yr)", yr: 5, mn: 0, dy: 0, i: "t1918602869c" }}, 
            // Monthly Chart: Max historical data (using a very large year value)
            {{ p: "M", label: "Monthly (Max)", yr: 99, mn: 0, dy: 0, i: "t1918602869c" }} 
        ];

        const gridContainer = document.getElementById('chart-grid');
        
        let allContentHTML = '';

        for (const ticker of tickers) {{
            allContentHTML += `
                <div class="stock-row">
                    <h2 class="stock-ticker-header">${{ticker.toUpperCase()}}</h2>
            `;
            
            for (const tfConfig of timeFrames) {{
                // Construct StockCharts.com URL
                const chartUrl = `https://stockcharts.com/c-sc/sc?s=${{ticker.toUpperCase()}}&p=${{tfConfig.p}}&yr=${{tfConfig.yr}}&mn=${{tfConfig.mn}}&dy=${{tfConfig.dy}}&i=${{tfConfig.i}}`;
                
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
        print(f"✅ Successfully generated '{output_filename}' using StockCharts.com URLs.")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")


def generate_multi_timeframe_stockcharts_line_html(
    tickers: list[str],
    output_filename: str = "multi_timeframe_stockcharts_line.html",
):
    """Generate multi-timeframe StockCharts gallery using a line-chart style.

    Uses the same layout as generate_multi_timeframe_stockcharts_html but builds
    URLs based on a line-chart style template id (e.g. from
    https://stockcharts.com/c-sc/sc?s=ALAB&p=D&yr=1&mn=0&dy=0&i=t3101262626c).
    """

    tickers_js_array = json.dumps(tickers)

    html_template = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Multi-Timeframe StockCharts.com Line Gallery</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif;
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
            width: 100%;
            margin: 0 auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .stock-row {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            border: 1px solid #e0e0e0;
            background-color: #ffffff;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .stock-row h2.stock-ticker-header {{
            width: 100%;
            text-align: center;
            margin: 0;
            font-size: 1.5em;
            color: #2c3e50;
            background-color: #f8f8f8;
            padding: 10px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .chart-container {{
            flex: 1 1 calc(100% / 3);
            background-color: #ffffff;
            padding: 0;
            text-align: center;
            border-right: 1px solid #f0f0f0;
            min-width: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
        }}
        .chart-container:last-child {{
            border-right: none;
        }}
        .chart-container h3 {{
            margin: 10px 0 5px 0;
            font-size: 1.0em;
            color: #34495e;
            font-weight: 600;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            border-top: 1px solid #f0f0f0;
        }}
        @media (max-width: 992px) {{
            .chart-container {{
                flex: 1 1 50%;
            }}
            .chart-container:nth-child(even) {{
                border-right: none;
            }}
            .chart-container:nth-child(odd) {{
                border-right: 1px solid #f0f0f0;
            }}
            .stock-row h2.stock-ticker-header {{
                font-size: 1.3em;
            }}
        }}
        @media (max-width: 680px) {{
            .chart-container {{
                flex: 1 1 100%;
                border-right: none;
            }}
            .chart-container:not(:last-child) {{
                border-bottom: 1px solid #f0f0f0;
            }}
            img {{
                border-top: none;
            }}
            .chart-container h3 {{
                border-bottom: 1px solid #f0f0f0;
                width: 100%;
                padding-bottom: 5px;
            }}
        }}
    </style>
</head>
<body>
    
    <h1>Multi-Timeframe StockCharts.com Line Gallery</h1>

    <div class=\"grid-container\" id=\"chart-grid\"></div>

    <script>
        const tickers = {tickers_js_array};

        // Use the same D/W/M timeframes but a line-chart style template id (from the
        // provided daily line URL). Reuse it across timeframes.
        const timeFrames = [
            {{ p: "D", label: "Daily (1yr)",   yr: 1,  mn: 0, dy: 0, i: "t3101262626c" }},
            {{ p: "W", label: "Weekly (5yr)", yr: 5,  mn: 0, dy: 0, i: "t3101262626c" }},
            {{ p: "M", label: "Monthly (Max)", yr: 99, mn: 0, dy: 0, i: "t3101262626c" }}
        ];

        const gridContainer = document.getElementById('chart-grid');
        
        let allContentHTML = '';

        for (const ticker of tickers) {{
            allContentHTML += `
                <div class=\"stock-row\">
                    <h2 class=\"stock-ticker-header\">${{ticker.toUpperCase()}}</h2>
            `;
            
            for (const tfConfig of timeFrames) {{
                const chartUrl = `https://stockcharts.com/c-sc/sc?s=${{ticker.toUpperCase()}}&p=${{tfConfig.p}}&yr=${{tfConfig.yr}}&mn=${{tfConfig.mn}}&dy=${{tfConfig.dy}}&i=${{tfConfig.i}}`;
                
                allContentHTML += `
                    <div class=\"chart-container\">
                        <h3>${{tfConfig.label}}</h3>
                        <img src=\"${{chartUrl}}\" alt=\"${{ticker.toUpperCase()}} ${{tfConfig.label}} Line Chart\">
                    </div>
                `;
            }}
            
            allContentHTML += `</div>`;
        }}
        gridContainer.innerHTML = allContentHTML;
    </script>

</body>
</html>
"""

    try:
        with open(output_filename, "w") as f:
            f.write(html_template)
        print(f"✅ Successfully generated '{output_filename}' using StockCharts.com line-chart URLs.")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    my_tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META',
        'JPM', 'V', 'JNJ', 'UNH', 'WMT', 'PG', 'MA'
    ]
    
    generate_multi_timeframe_stockcharts_html(
        tickers=my_tickers
    )

    try:
        # Open the generated HTML in the default web browser
        output_filename = "multi_timeframe_stockcharts.html"
        webbrowser.open(os.path.abspath(output_filename))
        print(f"✅ Opened '{output_filename}' in web browser")
    except Exception as e:
        print(f"❌ Error opening file in web browser: {e}")
