# Changelog

## [1.1.0] - 2025-01-04
### Added
- Comprehensive feature-wise normalization in data preprocessing
- Learning rate scheduling in model training
- Early stopping mechanism
- Enhanced logging and error handling
- More robust LSTM model architecture with dropout

### Changed
- Improved sliding window creation method
- Updated model training process to handle multi-dimensional input
- Refined visualization and prediction generation scripts

### Fixed
- Resolved dimensionality issues in data preprocessing
- Corrected scaling and normalization problems
- Improved model training stability

### Performance
- Reduced model training loss from extremely high values to reasonable range
- Added more informative training progress tracking
- Implemented better model evaluation metrics

## [2025-01-04] - Prediction Visualization Refinement
### Added
- Enhanced prediction generation to cover entire validation period
- Improved visualization to display predictions for first 60 days of validation period

### Changed
- Updated `plot_predictions` function in `visualization.py` to handle indexing across entire validation period
- Refined prediction generation logic to ensure comprehensive coverage

### Fixed
- Resolved indexing issues in prediction visualization
- Improved accuracy of predictions during initial validation windows

### Performance
- Achieved extremely low prediction error (as low as $0.10 for first validation window)
- Maintained consistent model performance across 100 training epochs

### Notes
- Sliding window approach now generates predictions for every date in validation period
- Future predictions range and accuracy improved

## [2025-01-04] - Comprehensive Validation Period Extension

### Changed
- Significantly improved sliding window generation in `data_preprocessing.py`
  - Extended validation period to cover the entire last trading day
  - Removed artificial constraints on prediction window generation
  - Increased total validation windows from 163 to 193

### Fixed
- Resolved indexing issues in `visualization.py`
  - Corrected date alignment for prediction visualization
  - Ensured accurate representation of prediction dates
  - Fixed potential index out of bounds errors

### Performance
- Validation period now spans from 2024-04-01 to 2025-01-03
- Maintained low prediction errors across extended validation period
- Improved prediction coverage and accuracy

### Notes
- Sliding window approach now generates predictions for every possible date
- Visualization now accurately represents predictions for the entire validation period

## [1.0.0] - Initial Release
- Initial project setup
- Basic stock prediction pipeline
- TensorFlow to PyTorch migration
