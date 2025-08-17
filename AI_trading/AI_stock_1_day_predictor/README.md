# AI Stock Price Predictor

## Overview
This is a Python-based AI stock price prediction system that uses LSTM neural networks to forecast stock prices. The system incorporates historical price data, technical indicators, and sentiment analysis to generate predictions with confidence intervals and risk assessments.

## Features
- Historical stock data collection from Yahoo Finance
- Advanced data preprocessing and feature engineering
- LSTM-based deep learning model
- Prediction confidence intervals
- Risk assessment metrics
- Visualization of results
- Logging of all predictions

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Usage
Run the predictor with a stock ticker symbol:
```bash
python main.py AAPL
```

Optional parameters:
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Training batch size (default: 32)
- `--window_size`: Sliding window size (default: 60)
- `--forecast_horizon`: Number of days to predict (default: 30)

Example:
```bash
python main.py AAPL --epochs 200 --window_size 90 --forecast_horizon 60
```

## Important Disclaimers
1. This is an experimental project and should not be used for actual investment decisions.
2. Stock market predictions are inherently uncertain and past performance is not indicative of future results.
3. The model's predictions come with significant uncertainty and should be interpreted with caution.
4. Always consult with a qualified financial advisor before making any investment decisions.

## License
MIT License

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Support
For support or questions, please open an issue in the repository.
