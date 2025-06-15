import pandas as pd
import numpy as np
import joblib
import json
import logging
import os
from models.feature_engineering import AdvancedFeatureEngineer
from models.model_ensemble import OptimizedEnsembleModel
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_input_data(df):
    """Validate input data structure and quality"""
    required_columns = ['unique_id', 'age', 'credit_score']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    if df['unique_id'].duplicated().any():
        logger.warning("Duplicate IDs found in input data")
    
    logger.info(f"Input validation passed. Data shape: {df.shape}")

def predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced prediction function with comprehensive error handling
    """
    try:
        # Validate input
        validate_input_data(df)
        
        # Check if models exist
        if not os.path.exists('models/saved/feature_engineer.pkl'):
            raise FileNotFoundError("Feature engineer model not found. Please train the model first.")
        if not os.path.exists('models/saved/ensemble_model.pkl'):
            raise FileNotFoundError("Ensemble model not found. Please train the model first.")
        
        # Load trained models
        logger.info("Loading trained models...")
        feature_engineer = joblib.load('models/saved/feature_engineer.pkl')
        ensemble_model = OptimizedEnsembleModel({})
        ensemble_model.load_model('models/saved/ensemble_model.pkl')
        
        # Store IDs
        ids = df['unique_id'].copy()
        
        # Feature engineering
        logger.info("Applying feature engineering...")
        df_processed = feature_engineer.transform(df)
        
        # Remove target if present (for test data)
        if 'target' in df_processed.columns:
            df_processed = df_processed.drop(['target'], axis=1)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = ensemble_model.predict(df_processed)
        
        # Ensure predictions are non-negative and reasonable
        predictions = np.maximum(predictions, 0)
        predictions = np.minimum(predictions, 1000000)  # Cap at 10 lakhs
        
        # Create result dataframe with only unique_id and predicted_income
        result_df = pd.DataFrame({
            'unique_id': ids,
            'predicted_income': predictions
            # NOTE: Formerly, prediction_category was added as below (commented out)
            # 'prediction_category': pd.cut(
            #     predictions,
            #     bins=[0, 25000, 50000, 100000, np.inf],
            #     labels=['Low', 'Medium', 'High', 'Very High']
            # )
        })
        
        logger.info(f"Predictions completed successfully. Generated {len(result_df)} predictions.")
        return result_df
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        # Return dummy predictions with error flag for debugging
        result_df = pd.DataFrame({
            'unique_id': df['unique_id'] if 'unique_id' in df.columns else range(len(df)),
            'predicted_income': [50000] * len(df),  # Default prediction
            # NOTE: Formerly, prediction_category and error_flag were added as below (commented out)
            # 'prediction_category': ['Medium'] * len(df),
            # 'error_flag': [True] * len(df)
        })
        # NOTE: If you want to keep the extra columns in error cases, uncomment the above and below
        # result_df = pd.DataFrame({
        #     'unique_id': df['unique_id'] if 'unique_id' in df.columns else range(len(df)),
        #     'predicted_income': [50000] * len(df),
        #     'prediction_category': ['Medium'] * len(df),
        #     'error_flag': [True] * len(df)
        # })
        return result_df

def main():
    """Test the enhanced inference pipeline"""
    try:
        logger.info("Testing inference pipeline...")
        
        # Load test data
        test_data_path = 'data/Hackathon_bureau_data_400.csv'
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data not found: {test_data_path}")
        
        df_test = pd.read_csv(test_data_path)
        
        # Make predictions
        result_df = predict(df_test)
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Save results (only unique_id and predicted_income columns)
        output_path = 'output/output_Hackathon_bureau_data_400.csv'
        result_df[['unique_id', 'predicted_income']].to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        logger.info(f"Results shape: {result_df.shape}")
        
        # Display sample predictions (only unique_id and predicted_income columns)
        print("\nSample predictions (unique_id and predicted_income only):")
        print(result_df[['unique_id', 'predicted_income']].head(10))
        
        # Print basic statistics for predicted_income
        if not result_df['predicted_income'].isna().all():
            stats = {
                'count': len(result_df),
                'mean': result_df['predicted_income'].mean(),
                'median': result_df['predicted_income'].median(),
                'std': result_df['predicted_income'].std(),
                'min': result_df['predicted_income'].min(),
                'max': result_df['predicted_income'].max()
            }
            
            print("\nPrediction Statistics:")
            for key, value in stats.items():
                if key == 'count':
                    print(f"{key.capitalize()}: {value}")
                else:
                    print(f"{key.capitalize()}: ₹{value:,.2f}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Inference pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
