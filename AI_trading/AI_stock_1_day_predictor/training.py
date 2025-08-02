import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """
    Train the stock price prediction model
    
    Args:
        model (torch.nn.Module): Neural network model
        X_train (np.ndarray): Training input features
        y_train (np.ndarray): Training target values
        X_test (np.ndarray): Test input features
        y_test (np.ndarray): Test target values
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    
    Returns:
        tuple: Training and validation losses
    """
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Lists to store losses
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Training mode
        model.train()
        
        # Shuffle training data
        indices = torch.randperm(X_train_tensor.size(0))
        X_train_shuffled = X_train_tensor[indices]
        y_train_shuffled = y_train_tensor[indices]
        
        # Mini-batch training
        for i in range(0, X_train_tensor.size(0), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Compute loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        
        # Validation mode
        model.eval()
        with torch.no_grad():
            # Training loss
            train_pred = model(X_train_tensor)
            train_loss = criterion(train_pred, y_train_tensor)
            train_losses.append(train_loss.item())
            
            # Validation loss
            val_pred = model(X_test_tensor)
            val_loss = criterion(val_pred, y_test_tensor)
            val_losses.append(val_loss.item())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return train_losses, val_losses
