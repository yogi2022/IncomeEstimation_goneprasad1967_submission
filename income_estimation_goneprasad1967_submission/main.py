import pandas as pd
import numpy as np
import json
import os
import logging
from sklearn.model_selection import train_test_split
from models.feature_engineering import AdvancedFeatureEngineer
from models.model_ensemble import OptimizedEnsembleModel
from models.utils import calculate_evaluation_metrics, plot_predictions
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path='config/config.json'):
    """Load configuration with error handling"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise

def validate_config(config):
    """Validate configuration parameters"""
    required_keys = ['model_params', 'feature_engineering', 'training']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    if config['training']['cv_folds'] < 2:
        raise ValueError("cv_folds must be at least 2")
    
    logger.info("Configuration validation passed")

def create_output_directories():
    """Create necessary output directories"""
    directories = ['models/saved', 'output', 'logs', 'plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("Output directories created")

def main():
    """Enhanced main training pipeline with comprehensive logging"""
    try:
        logger.info("Starting Income Estimation Model Training...")
        
        # Load and validate configuration
        config = load_config()
        validate_config(config)
        
        # Create output directories
        create_output_directories()
        
        # Load data with validation
        data_path = 'data/Hackathon_bureau_data_400.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info("Loading data...")
        df = pd.read_csv(data_path)
        logger.info(f"Data shape: {df.shape}")
        
        # Data quality checks
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if 'target' not in df.columns:
            raise ValueError("Target column 'target' not found in dataset")
        
        # Initialize feature engineer
        feature_engineer = AdvancedFeatureEngineer()
        
        # Feature engineering with progress tracking
        logger.info("Performing feature engineering...")
        df_processed = feature_engineer.fit_transform(df)
        logger.info(f"Feature engineering completed. New shape: {df_processed.shape}")
        
        # Prepare data
        X = df_processed.drop(['target'], axis=1)
        y = df_processed['target']
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Target statistics - Mean: {y.mean():.2f}, Std: {y.std():.2f}")
        
        # Split data with stratification if enabled
        test_size = config['training']['test_size']
        random_state = config['training']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Initialize and train ensemble model
        logger.info("Initializing ensemble model...")
        ensemble_model = OptimizedEnsembleModel(config)
        
        logger.info("Training ensemble model...")
        ensemble_model.fit(X_train, y_train)
        
        # Make predictions
        logger.info("Making predictions...")
        train_pred = ensemble_model.predict(X_train)
        test_pred = ensemble_model.predict(X_test)
        
        # Evaluate model with detailed metrics
        logger.info("\n" + "="*50)
        logger.info("TRAINING METRICS:")
        train_metrics = calculate_evaluation_metrics(y_train, train_pred)
        for metric, value in train_metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        logger.info("\n" + "="*50)
        logger.info("TEST METRICS:")
        test_metrics = calculate_evaluation_metrics(y_test, test_pred)
        for metric, value in test_metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        # Calculate overfitting metrics
        r2_diff = train_metrics['r2'] - test_metrics['r2']
        rmse_ratio = test_metrics['rmse'] / train_metrics['rmse']
        
        logger.info(f"\nOVERFITTING ANALYSIS:")
        logger.info(f"R² difference (train - test): {r2_diff:.4f}")
        logger.info(f"RMSE ratio (test / train): {rmse_ratio:.4f}")
        
        if r2_diff > 0.15:
            logger.warning("High overfitting detected! Consider regularization.")
        
        # Save models and results
        logger.info("Saving models and results...")
        joblib.dump(feature_engineer, 'models/saved/feature_engineer.pkl')
        ensemble_model.save_model('models/saved/ensemble_model.pkl')
        
        # Save metrics to file
        results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'overfitting_analysis': {
                'r2_difference': r2_diff,
                'rmse_ratio': rmse_ratio
            },
            'config': config
        }
        
        with open('output/training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to output/training_results.json")
        
        # Plot results
        try:
            plot_predictions(y_test, test_pred, "Test Set: Predictions vs Actual")
            logger.info("Prediction plots generated")
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
        
        return ensemble_model, feature_engineer, test_metrics
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
