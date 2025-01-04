import argparse
import warnings
import numpy as np
import torch
from data_collection import get_stock_data, get_news_sentiment
from data_preprocessing import clean_stock_data, create_sliding_windows, train_test_split
from model_training import create_lstm_model, train_model, evaluate_model, save_model
from prediction_generation import generate_predictions, calculate_confidence_intervals, assess_risk, log_predictions
from visualization import plot_predictions, plot_risk_assessment, save_plots
from sklearn.preprocessing import MinMaxScaler

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')

def main(ticker, epochs=100, batch_size=32, window_size=60, forecast_horizon=30):
    try:
        print(f"\nStarting stock prediction for {ticker}...")
        
        # Step 1: Data Collection
        print("1. Collecting stock data...")
        stock_data = get_stock_data(ticker)
        sentiment_data = get_news_sentiment(ticker)
        
        # Step 2: Data Preprocessing
        print("2. Preprocessing data...")
        cleaned_data = clean_stock_data(stock_data)
        X, y = create_sliding_windows(cleaned_data, window_size, forecast_horizon)
        
        # Normalize data
        # Use only numeric columns for scaling
        X_numeric = X.reshape(-1, X.shape[-1])
        
        # Create separate scalers for each feature
        scalers = [MinMaxScaler() for _ in range(X.shape[-1])]
        X_normalized = np.zeros_like(X, dtype=float)
        
        for i in range(X.shape[-1]):
            X_normalized[:, :, i] = scalers[i].fit_transform(X[:, :, i].reshape(-1, 1)).reshape(X.shape[0], X.shape[1])
        
        # Split data
        X_train, X_test = train_test_split(X_normalized)
        y_train, y_test = train_test_split(y)
        
        # Step 3: Model Training
        print("3. Training model...")
        model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
        model, history = train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size)
        
        # Step 4: Prediction Generation
        print("4. Generating predictions...")
        # Use last sequence for prediction
        last_sequence = X_test[-1:]
        predictions_tensor = model(torch.FloatTensor(last_sequence))
        
        # Convert to numpy and reshape
        predictions_np = predictions_tensor.detach().numpy().reshape(-1, 1)
        
        # Inverse transform predictions
        # Use the first scaler (assuming target feature is the first column)
        predictions = scalers[0].inverse_transform(predictions_np)
        
        confidence_intervals = calculate_confidence_intervals(predictions)
        risk_assessment = assess_risk(predictions, stock_data)
        
        # Step 5: Visualization
        print("5. Creating visualizations...")
        prediction_plot = plot_predictions(stock_data, predictions, confidence_intervals)
        risk_plot = plot_risk_assessment(risk_assessment)
        
        # Save results
        save_plots(prediction_plot, f'{ticker}_predictions')
        save_plots(risk_plot, f'{ticker}_risk_assessment')
        
        # Log predictions
        log_predictions(predictions, confidence_intervals, risk_assessment, ticker)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Price Predictor')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g. AAPL)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--window_size', type=int, default=60, help='Sliding window size')
    parser.add_argument('--forecast_horizon', type=int, default=30, help='Forecast horizon')
    
    args = parser.parse_args()
    
    main(
        args.ticker,
        epochs=args.epochs,
        batch_size=args.batch_size,
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon
    )
