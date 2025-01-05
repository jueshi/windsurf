import torch
import torch.nn as nn

class StockPricePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout_rate=0.2):
        """
        LSTM-based Stock Price Prediction Model
        
        Args:
            input_size (int): Number of features in input
            hidden_size (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate for regularization
        """
        super(StockPricePredictionModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Fully connected layer for prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            torch.Tensor: Predicted stock prices
        """
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step
        last_time_step = lstm_out[:, -1, :]
        
        # Fully connected layer for final prediction
        prediction = self.fc(last_time_step)
        
        return prediction
