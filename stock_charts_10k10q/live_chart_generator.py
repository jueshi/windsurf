import json

def generate_chart_html(
    tickers: list[str], 
    columns: int, 
    output_filename: str = "stock_charts.html",
    time_frame: str = "d"
):
    """
    Generates a self-contained HTML file to display a grid of stock charts.

    Args:
        tickers (list[str]): A list of stock ticker symbols.
        columns (int): The number of columns for the chart grid.
        time_frame (str): The chart time frame. Valid options are:
                          'd' (daily), 'w' (weekly), 'm' (monthly).
        output_filename (str): The name of the output HTML file.
    """
    # --- Input Validation ---
    valid_time_frames = ["d", "w", "m"]
    if time_frame not in valid_time_frames:
        raise ValueError(f"Invalid time_frame '{time_frame}'. Please use one of {valid_time_frames}")

    tickers_js_array = json.dumps(tickers)

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
            width: 100%;
            height: 300px;
            object-fit: contain;
            display: block;
            background-color: #ffffff;
        }}
    </style>
</head>
<body>
    
    <h1>Stock Charts (Time Frame: {'Daily' if time_frame == 'd' else 'Weekly' if time_frame == 'w' else 'Monthly'})</h1>

    <div class="grid-container" id="chart-grid"></div>

    <script>
        // --- Configuration injected by Python ---
        const tickers = {tickers_js_array};
        const columns = {columns};
        const timeFrame = "{time_frame}"; // New variable for the time frame

        // --- Dynamic Chart Generation Logic ---
        const gridContainer = document.getElementById('chart-grid');
        const flexBasis = 100 / columns;
        
        const style = document.createElement('style');
        style.innerHTML = `.chart-container {{ flex: 1 1 ${{flexBasis}}%; }}`;
        document.head.appendChild(style);

        let allChartsHTML = '';
        for (const ticker of tickers) {{
            // --- UPDATED LOGIC ---
            // Determine the correct range parameter ('r') based on the time frame
            let rangeParam = '';
            if (timeFrame === 'd') {{
                rangeParam = '&r=y1';
            }} else if (timeFrame === 'w') {{
                rangeParam = '&r=y2';
            }} else if (timeFrame === 'm') {{
                rangeParam = '&r=y5';
            }}

            // Construct the final URL with the correct range parameter
            const chartUrl = `https://charts-node.finviz.com/chart?w=466&h=292&bw=1&bm=1&bb=1&t=${{ticker.toUpperCase()}}&tf=${{timeFrame}}&s=linear&pm=240&am=1200&ct=candle_stick&tm=d${{rangeParam}}&o[0][ot]=sma&o[0][op]=20&o[0][oc]=DC32B363&o[1][ot]=sma&o[1][op]=50&o[1][oc]=FF8F33C6&o[2][ot]=sma&o[2][op]=200&o[2][oc]=DCB3326D&o[3][ot]=patterns&o[3][op]=&o[3][oc]=69C1EAFF&o[4][ot]=vp&o[4][op]=30%2C0.3&o[4][oc]=18B8475B&o[5][ot]=vwap&o[5][op]=&o[5][oc]=9467BDFF&i[0][it]=rsi&i[0][ip]=14&cc[dark][canvasFill]=22262f`;
                  
            allChartsHTML += `
                <div class="chart-container">
                    <h2>${{ticker.toUpperCase()}}</h2>
                    <img src="${{chartUrl}}" alt="${{ticker.toUpperCase()}} Stock Chart">
                </div>
            `;
        }}
        gridContainer.innerHTML = allChartsHTML;
    </script>

</body>
</html>
"""

    try:
        with open(output_filename, "w") as f:
            f.write(html_template)
        print(f"✅ Successfully generated '{output_filename}' with a {time_frame.upper()} time frame.")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    my_tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM',
        'V', 'JNJ', 'UNH', 'WMT', 'PG', 'MA'
    ]
    
    number_of_columns = 4
    
    # Customize your time frame here: 'd' for daily, 'w' for weekly, 'm' for monthly
    time_frame_setting = "w"

    # Call the function with the new time_frame argument
    generate_chart_html(
        tickers=my_tickers, 
        columns=number_of_columns, 
        time_frame=time_frame_setting
    )