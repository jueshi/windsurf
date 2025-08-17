import argparse
import warnings
import numpy as np
import torch
from data_collection import get_stock_data, get_news_sentiment
from data_preprocessing import clean_stock_data, create_sliding_windows, normalize_data
from model import StockPricePredictionModel
from prediction_generation import generate_predictions, calculate_confidence_intervals, assess_risk, rescale_predictions
from visualization import plot_predictions, plot_prediction_error, plot_risk_assessment, save_plots
from training import train_model
import pandas as pd

# command to run it: venv\Scripts\python.exe main.py AAPL --epochs 100 --window_size 60 --forecast_horizon 30
def main(ticker, epochs=100, batch_size=32, window_size=60, forecast_horizon=30):
    """
    Main function to predict stock prices
    
    Args:
        ticker (str): Stock ticker symbol
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        window_size (int): Size of sliding window for input features
        forecast_horizon (int): Number of days to forecast
    """
    try:
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        print(f"Starting stock prediction for {ticker}...")
        
        # Step 1: Collect stock data
        print("1. Collecting stock data...")
        stock_data = get_stock_data(ticker)
        
        # Optional: Get news sentiment data
        # news_sentiment = get_news_sentiment(ticker)
        
        # Step 2: Preprocess data
        print("2. Preprocessing data...")
        cleaned_stock_data = clean_stock_data(stock_data)
        
        # Create sliding windows
        X_train, X_test, y_train, y_test, scalers, validation_indices, validation_prices, original_data, train_split = create_sliding_windows(
            cleaned_stock_data, 
            window_size=window_size, 
            forecast_horizon=forecast_horizon
        )
        
        # Normalize input data
        X_train_normalized, _ = normalize_data(X_train)
        X_test_normalized, _ = normalize_data(X_test)
        
        # Step 3: Train model
        print("3. Training model...")
        model = StockPricePredictionModel(
            input_size=X_train.shape[2], 
            hidden_size=64, 
            num_layers=2
        )
        
        # Train the model
        train_losses, val_losses = train_model(
            model, 
            X_train_normalized, 
            y_train, 
            X_test_normalized, 
            y_test, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # Step 4: Generate predictions
        print("4. Generating predictions...")
        normalized_predictions, normalized_future_predictions = generate_predictions(
            model, 
            X_test_normalized, 
            forecast_horizon=forecast_horizon
        )
        
        # Rescale predictions
        predictions_rescaled = rescale_predictions(normalized_predictions, original_data, scalers[0])
        future_predictions_rescaled = rescale_predictions(normalized_future_predictions, original_data, scalers[0])
        
        # Print detailed prediction information for the first validation window
        if validation_indices and len(validation_indices) > 0:
            first_validation_index = validation_indices[0]
            
            # Prediction details
            prediction_window_start = stock_data.index[first_validation_index]
            prediction_window_end = stock_data.index[first_validation_index + window_size]
            prediction_target_date = stock_data.index[first_validation_index + window_size + 1]
            
            # Actual prices
            actual_price_at_prediction_window_end = stock_data.loc[prediction_window_end, 'Close']
            actual_price_at_prediction_target = stock_data.loc[prediction_target_date, 'Close']
            
            # Predicted price
            predicted_price = predictions_rescaled[0]
            
            print("\n===== First Validation Window Prediction Details =====")
            print(f"Prediction Window Start Date: {prediction_window_start}")
            print(f"Prediction Window End Date: {prediction_window_end}")
            print(f"Prediction Target Date: {prediction_target_date}")
            print(f"Actual Price at Prediction Window End: ${actual_price_at_prediction_window_end:.2f}")
            print(f"Predicted Price for Target Date: ${predicted_price:.2f}")
            print(f"Actual Price at Prediction Target Date: ${actual_price_at_prediction_target:.2f}")
            print(f"Prediction Error: ${abs(predicted_price - actual_price_at_prediction_target):.2f}")
            print("===================================================\n")
        
        # Prepare predictions for confidence intervals and risk assessment
        predictions = normalized_predictions
        
        confidence_intervals = calculate_confidence_intervals(predictions)
        risk_assessment = assess_risk(predictions, stock_data)
        
        # Step 5: Visualization
        print("5. Creating visualizations...")
        y_test_rescaled = rescale_predictions(y_test, original_data, scalers[0])
        
        # Create validation dates
        validation_dates = stock_data.index[train_split:train_split+len(y_test_rescaled)]
        
        # Calculate future dates starting from the next trading day after the last day
        last_trading_day = stock_data.index[-1]
        future_dates = pd.date_range(
            start=last_trading_day + pd.offsets.BDay(1), 
            periods=forecast_horizon
        )
        
        plot_predictions(
            actual_prices=y_test_rescaled, 
            predicted_prices=predictions_rescaled, 
            future_predictions=future_predictions_rescaled,
            future_dates=future_dates,
            validation_dates=validation_dates,
            ticker=ticker
        )
        
        error_plot = plot_prediction_error(stock_data, predictions_rescaled, {}, X_train, X_test, y_train, y_test)
        risk_plot = plot_risk_assessment(risk_assessment)
        
        # Save results
        save_plots(error_plot, f'{ticker}_prediction_error')
        save_plots(risk_plot, f'{ticker}_risk_assessment')
        
        # Log predictions
        log_predictions(predictions, confidence_intervals, risk_assessment, ticker)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

def log_predictions(predictions, confidence_intervals, risk_assessment, ticker):
    """
    Log prediction details to a file
    
    Args:
        predictions (np.ndarray): Model predictions
        confidence_intervals (dict): Confidence interval data
        risk_assessment (dict): Risk assessment metrics
        ticker (str): Stock ticker symbol
    """
    # Implementation of logging function
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Price Predictor')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g. AAPL)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--window_size', type=int, default=60, help='Size of sliding window for input features')
    parser.add_argument('--forecast_horizon', type=int, default=30, help='Number of days to forecast')
    
    args = parser.parse_args()
    
    main(
        args.ticker, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        window_size=args.window_size, 
        forecast_horizon=args.forecast_horizon
    )
