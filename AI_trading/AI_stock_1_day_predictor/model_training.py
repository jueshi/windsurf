import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class StockPriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, output_dim=1):
        """
        Initialize LSTM model for stock price prediction
        
        Args:
            input_dim (int): Number of features in input
            hidden_dim (int): Number of hidden units in LSTM layers
            num_layers (int): Number of LSTM layers
            output_dim (int): Number of output features
        """
        super(StockPriceLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step
        last_time_step = lstm_out[:, -1, :]
        
        # Fully connected layer
        predictions = self.fc(last_time_step)
        
        return predictions

def create_lstm_model(input_shape):
    """Create LSTM model with given input shape"""
    model = StockPriceLSTM(input_dim=input_shape[1])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """
    Train LSTM model with early stopping and learning rate scheduling
    
    Args:
        model (nn.Module): PyTorch model to train
        X_train (np.ndarray): Training input features
        y_train (np.ndarray): Training target values
        X_test (np.ndarray): Validation input features
        y_test (np.ndarray): Validation target values
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    
    Returns:
        Trained model and training history
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Best model tracking
    best_val_loss = float('inf')
    patience = 10
    no_improve_epochs = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training mode
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        train_predictions = model(X_train_tensor)
        train_loss = criterion(train_predictions, y_train_tensor)
        
        # Backward pass and optimization
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test_tensor)
            val_loss = criterion(val_predictions, y_test_tensor)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store losses
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_stock_predictor.pth')
        else:
            no_improve_epochs += 1
        
        # Stop if no improvement
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('models/best_stock_predictor.pth'))
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    model.eval()
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    with torch.no_grad():
        predictions = model(X_test_tensor)
        mse = nn.MSELoss()(predictions, y_test_tensor)
        rmse = torch.sqrt(mse)
    
    return {
        'MSE': mse.item(),
        'RMSE': rmse.item()
    }

def save_model(model, filename='stock_predictor.pth'):
    """Save model to file"""
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{filename}')

def load_model(model, path='models/best_stock_predictor.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
