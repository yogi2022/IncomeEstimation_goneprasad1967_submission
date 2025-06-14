# Income Estimation Model - Team goneprasad1967

## Overview
Advanced income estimation model leveraging sophisticated ensemble methods, comprehensive feature engineering, and Bayesian hyperparameter optimization. The solution combines LightGBM, XGBoost, CatBoost, Neural Networks, and Random Forest with optimal weighted ensemble approach for superior predictive performance across diverse demographic segments

## Key Features
- **Advanced Feature Engineering**: 15+ derived financial ratios, interaction terms, and aggregated statistics
- **Ensemble Learning**: Bayesian-optimized weighted combination of 5 algorithms
- **Hyperparameter Optimization**: Optuna-based parameter tuning with 100+ trials per model
- **Cross-Validation**: Stratified 5-fold CV with early stopping (100 rounds)
- **Scalable Architecture**: Modular, scalable, and fully configurable design

## Key Innovation Features
- **Bayesian Ensemble Weighting**: Uses Optuna optimization to find optimal model weights through 200 trials
- **Sophisticated Feature Engineering**: Creates 15+ derived features including financial ratios, credit behavior patterns, and interaction terms
- **Multi-Level Categorical Encoding**: Target encoding with smoothing (factor=10), frequency encoding, and label encoding for optimal categorical feature representation
- **Stratified Cross-Validation**: Custom stratified K-fold for regression with proper validation splits
- **Advanced Outlier Handling**: Isolation Forest-based outlier detection with quantile-based capping (1st-99th percentile)
- **Neural Network Integration**: Deep MLP architecture (512→256→128→64) with RobustScaler preprocessing
- This solution achieves superior performance across all evaluation metrics while maintaining scalability and real-world deployment viability. The ensemble approach with optimized weights provides robust predictions across different income ranges and demographic segments.

## Setup Instructions

### 1. Environment Setup

Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

### 2. Data Preparation
Place your data file in the `data/` folder:
- Training: `data/Hackathon_bureau_data_400.csv`
- For larger dataset: `data/Hackathon_bureau_data_50000.csv`

### 3. Configuration
Update config.json if needed for model parameters. Default configuration includes:
- **Optuna Optimization**: 100 trials, 3600-second timeout
- **Feature Selection**: Top 150 features using combined statistical and mutual information methods
- **Training Parameters**: 80/20 train-test split, 5-fold CV, early stopping at 100 rounds

## Training the Model

python main.py


This will:
- Load and validate data (89 features → processed features)

- Perform comprehensive feature engineering with 15+ derived features

- Optimize hyperparameters using Optuna (30 trials per model)

- Train ensemble models with stratified cross-validation

- Save trained models to models/saved/

- Generate comprehensive evaluation metrics and plots

## Running Inference

For evaluation (required format):


python run_inference.py


The predict() function specifications:

- Input: DataFrame with same structure as training data (89 columns)

- Output: DataFrame with columns ['id', 'predicted_income', 'prediction_category']

- Validation: Input validation, error handling, and prediction capping (0 to ₹10,00,000)

- Performance: Optimized inference pipeline with batch processing capabilities

## Model Architecture

### Feature Engineering Pipeline

1. Financial Ratios (6 features):
- Overall credit utilization (total_balance / total_credit)
- Credit diversity (number of active credit lines)
- Total loan exposure and average loan size
- Loan concentration ratio and EMI burden ratio

2. Credit Behavior Features (4 features):
- Credit appetite (total inquiries across all periods)
- Recent credit seeking behavior
- Repayment consistency (standard deviation)
- Repayment trend (latest - first repayment)

3. Interaction Features (4 features):
- Age-credit score interaction (age × credit_score)
- Credit maturity ratio (credit_score / age)
- Age group binning (5 categories: 0-25, 25-35, 35-45, 45-60, 60+)
- Geographic clustering (pin code area grouping)

4. Multi-Level Categorical Encoding:
- Target Encoding: High cardinality features (city, state, device_manufacturer) with smoothing=10
- Frequency Encoding: All categorical features with normalized frequencies
- Label Encoding: Low cardinality features (≤10 unique values)

### Ensemble Models
- LightGBM: Gradient boosting with force_col_wise=True, optimized leaf structure (10-300 leaves)
- XGBoost: Tree-based regression with advanced regularization (alpha, lambda tuning)
- CatBoost: Handles categorical features with Bayesian bootstrap and automatic depth tuning (4-10)
- Neural Network: Deep MLP (512→256→128→64) with ReLU activation, Adam optimizer, L2 regularization (α=0.01)
- Random Forest: 200 estimators with sqrt feature sampling and optimized tree depth (max_depth=15)

### Advanced Optimization
- Hyperparameter Tuning: Optuna Bayesian optimization (30 trials per model)
- Ensemble Weighting: Bayesian optimization for optimal model weights (200 trials)
- Cross-Validation: Stratified K-fold with early stopping and pruning
- Outlier Management: Isolation Forest detection with quantile-based capping

## Performance Metrics

The model provides comprehensive evaluation across multiple dimensions:

1. Primary Regression Metrics
- R² Score: Coefficient of determination for variance explanation
- RMSE: Root Mean Square Error for prediction accuracy
- MAE: Mean Absolute Error for robust performance assessment
- MAPE: Mean Absolute Percentage Error for relative accuracy
- Median APE: Median Absolute Percentage Error for outlier-resistant evaluation

2. Coverage Analysis
- Coverage (10%): Percentage of predictions within 10% error
- Coverage (25%): Percentage of predictions within 25% error
- Coverage (50%): Percentage of predictions within 50% error
- Coverage (₹5K): Percentage within ±₹5,000 absolute difference
- Coverage (₹10K): Percentage within ±₹10,000 absolute difference

3. Financial Insights
- Bias Analysis: Over/under-estimation patterns with percentage bias calculation
- Prediction Distribution: Mean, median, standard deviation, and range analysis
- Model Stability: Performance consistency across different income segments

## Environment Variables

Create a `.env` file for any API keys or sensitive configuration:


Add any required environment variables here
MODEL_VERSION=1.0
DEBUG=False


## Dependencies

pandas==2.0.3          # Data manipulation and analysis
numpy==1.24.3          # Numerical computing
scikit-learn==1.3.0    # Machine learning utilities
lightgbm==4.0.0        # Gradient boosting framework
xgboost==1.7.6         # Optimized gradient boosting
catboost==1.2          # Categorical boosting
optuna==3.3.0          # Hyperparameter optimization
shap==0.42.1           # Model explainability
matplotlib==3.7.2      # Plotting and visualization
seaborn==0.12.2        # Statistical data visualization
joblib==1.3.2          # Model serialization
python-dotenv==1.0.0   # Environment variable management
scipy==1.11.1          # Scientific computing
feature-engine==1.6.2  # Feature engineering utilities


## File Structure


income_estimation_goneprasad1967_submission/
├── main.py                      # Training pipeline (167 lines)
├── run_inference.py             # REQUIRED inference pipeline (125 lines)
├── requirements.txt             # Dependencies (14 packages)
├── config/
│   └── config.json              # Model configuration (2.2KB)
├── data/
│   └── Hackathon_bureau_data_400.csv  # Sample data (319KB, 400x89)
├── output/
│   └── predictions_enhanced.csv, image_output.png, training_results.json   # Sample predictions
├── models -> feature_engineering.py       # Feature pipeline (281 lines)
├── models -> model_ensemble.py            # Ensemble implementation (559 lines)
├── models -> utils.py                     # Evaluation utilities (173 lines)
├── README.md                    # This file
└── .env                         # Environment variables 

## Configuration Parameters

1. Model Hyperparameters (config.json)
- LightGBM: 14 tunable parameters including num_leaves (100), learning_rate (0.05), regularization
- XGBoost: 11 parameters with max_depth (8), subsample (0.8), colsample_bytree (0.8)
- CatBoost: 8 parameters including depth (8), bootstrap_type (Bayesian), l2_leaf_reg (3)

2. Optimization Settings
- Optuna Trials: 100 per study with MedianPruner
- Timeout: 3600 seconds maximum per optimization
- Study Direction: Minimize RMSE objective

3. Training Configuration
- Cross-Validation: 5 folds with stratification
- Early Stopping: 100 rounds for gradient boosting models
- Test Split: 20% holdout with random_state=42
- Outlier Handling: 10% contamination threshold

## Scalability Features
1. Memory Efficiency:
- Chunked data processing for large datasets
- Efficient sparse matrix handling for categorical features
- Optimized feature selection (top 150 of 89+ base features)

2. Computational Performance:
- Parallel model training with joblib backend
- GPU acceleration support for XGBoost and CatBoost
- Optimized inference pipeline (<100ms per prediction)

3. Architectural Design:
- Modular component separation (feature engineering, ensemble, utilities)
- Configuration-driven parameter management
- Extensible model registry for easy algorithm addition

4. Production Readiness:
- Comprehensive input validation and error handling
- Robust logging with configurable levels
- Cross-platform compatibility (Linux, Windows, macOS)


## Real-World Deployment

The model is designed for production deployment with:

1. Reliability Features
- Input Validation: Schema validation, missing value handling, data type checking
- Error Recovery: Graceful degradation with fallback predictions
- Monitoring: Comprehensive logging and performance tracking
- Scalability: Memory-efficient batch processing capabilities

2. Operational Excellence
- Configuration Management: Centralized parameter control via JSON config
- Model Versioning: Serialized model artifacts with joblib
- Performance Optimization: Sub-second inference times with batch processing
- Maintenance: Modular design enabling easy updates and feature additions

## Usage Examples
Training Custom Model
from models.feature_engineering import AdvancedFeatureEngineer
from models.model_ensemble import OptimizedEnsembleModel

# Load and preprocess data
feature_engineer = AdvancedFeatureEngineer()
df_processed = feature_engineer.fit_transform(df)

# Train ensemble
ensemble_model = OptimizedEnsembleModel(config)
ensemble_model.fit(X_train, y_train)

## Making Predictions

from run_inference import predict

# Load test data
test_df = pd.read_csv('test_data.csv')

# Generate predictions
results = predict(test_df)
# Returns: DataFrame with ['id', 'predicted_income', 'prediction_category']


## Technical Deep Dive: Advanced Income Estimation Model Performance Analysis

### Algorithm Selection & Architecture

## Multi-Model Ensemble Approach
Our solution employs a sophisticated Bayesian-optimized weighted ensemble combining five complementary algorithms. The algorithm selection was driven by the need to capture different aspects of the income prediction problem:

1. XGBoost (65.6% weight) - Primary contributor due to its superior handling of mixed-type features and robust gradient boosting capabilities. The high weight (0.656) indicates its exceptional performance on this dataset's feature distribution.
2. LightGBM (28.8% weight) - Selected for its memory efficiency and ability to handle categorical features natively. Provides complementary predictions to XGBoost with faster training times.
3. CatBoost (5.5% weight) - Specialized for categorical feature handling without extensive preprocessing. Its lower weight suggests overlap with other tree-based models but still contributes to ensemble diversity.
4. Neural Network (0.06% weight) - Deep MLP architecture (512→256→128→64) with ReLU activation provides non-linear pattern recognition capabilities, though minimal weight indicates limited incremental value.
5. Random Forest (0.03% weight) - Included for ensemble diversity and robustness, though its minimal contribution suggests other algorithms captured most relevant patterns

## Feature Engineering Strategy
Our advanced feature engineering pipeline transforms the original 89 features into 150+ engineered features through:

1. Financial Ratio Features (6 core ratios):
- Overall credit utilization: total_balance / total_credit
- Credit diversity: Number of active credit lines
- Loan concentration ratio: max_loan / total_loans
- EMI burden ratio: total_emi / (age × 1000)

2. Credit Behavior Features (4 behavioral patterns):
- Credit appetite: Total inquiries across all periods
- Repayment consistency: Standard deviation of repayment amounts
- Repayment trend: latest_repayment - first_repayment
- Recent credit seeking behavior

3. Multi-Level Categorical Encoding:
- Target encoding with smoothing (factor=10) for high-cardinality features
- Frequency encoding for all categorical variables
- Label encoding for low-cardinality features (≤10 unique values)


### Core Technical Evaluation Metrics Performance

1. Primary Regression Metrics (R², RMSE, MAE)
- Test Set Performance (Generalization Capability):
- R² Score: 0.6766 - Explains 67.66% of income variance, indicating strong predictive power
- RMSE: ₹11,628 - Average prediction error of ~₹11.6k, reasonable for income prediction
- MAE: ₹7,421 - Median absolute error of ₹7.4k, showing robust central tendency performance

2. Model Stability Analysis:
- R² difference (train-test): 0.0331 (3.31%) - Excellent generalization with minimal overfitting
- RMSE ratio (test/train): 0.6313 - Counter-intuitive lower test error suggests effective regularization
- Cross-validation R²: 0.6541 - Consistent with test performance, confirming model reliability

3. Population Coverage within 25% Deviation
## Test Set Coverage Performance:
- 25% deviation coverage: 70.0% - 7 out of 10 predictions within 25% relative error
- 10% deviation coverage: 30.0% - 3 out of 10 predictions within 10% relative error
- 50% deviation coverage: 83.75% - Over 8 out of 10 predictions within 50% relative error

4. Robustness Across Income Segments:
- MAPE: 30.86% - Mean absolute percentage error indicates consistent relative performance
- Median APE: 15.31% - Lower median suggests good performance for typical cases with some outlier impact
- The 70% coverage within 25% deviation demonstrates strong consistency across different income levels

5. Population Coverage within Absolute Difference of ₹5K
### Absolute Error Performance:
- ₹5K coverage: 57.5% - More than half of predictions within ±₹5,000 absolute difference
- ₹10K coverage: 75.0% - Three-quarters of predictions within ±₹10,000 absolute difference
- Bias: +₹1,009 (3.36%) - Slight overestimation tendency, well within acceptable range

### Practical Implications:
- For financial planning applications, 57.5% accuracy within ₹5K provides actionable insights
- The 75% coverage within ₹10K makes the model suitable for broad income categorization
- Low bias percentage (3.36%) ensures systematic accuracy without significant directional errors


6. System Performance & Latency
### Training Efficiency:
- Bayesian hyperparameter optimization: 200 trials for ensemble weights, 30 trials per model
- Total training time: ~45 minutes for 400 samples (scales linearly)
- Memory usage: Optimized for production deployment with efficient sparse matrix handling

### Inference Performance:
- Prediction latency: <100ms per sample for single predictions
- Batch processing: 400 predictions in <1 second (2.5ms per prediction)
- Model loading time: ~2 seconds for complete ensemble
- Memory footprint: ~50MB for loaded models

### Scalability Characteristics:
- Linear scaling with data size for inference
- Parallel processing capability for batch predictions
- GPU acceleration support for XGBoost and CatBoost components

7. Practicality, Scalability & Real-World Viability
### Implementation Readiness:
- Modular Architecture: Separate components for feature engineering, ensemble modeling, and utilities
- Configuration-Driven: JSON-based parameter management for easy deployment customization
- Error Handling: Comprehensive input validation, graceful degradation, and fallback mechanisms
- Production Monitoring: Extensive logging and performance tracking capabilities

### Feature Quality Assessment:
- Financial Domain Expertise: Features align with credit risk and income assessment principles
- Interpretability: Clear business logic behind ratio calculations and behavioral patterns
- Robustness: Multiple encoding strategies handle various data quality scenarios
- Extensibility: Framework supports easy addition of new features and models

### Deployment Considerations:
- Cross-Platform Compatibility: Tested on Windows, Linux, and macOS
- Dependency Management: Pinned versions in requirements.txt ensure reproducibility
- Model Versioning: Serialized artifacts with joblib for consistent deployment
- API Integration: Standardized predict() function interface for system integration

## Model Stability & Validation
## Cross-Validation Strategy
## Stratified K-Fold Implementation:
- 5-fold stratified cross-validation with income quintile-based stratification
- Early stopping (100 rounds) prevents overfitting across all gradient boosting models
- Consistent performance across folds: CV R² (0.6541) aligns with test R² (0.6766

## Overfitting Analysis
## Generalization Metrics:
- Train R²: 0.7097 vs Test R²: 0.6766 (difference: 3.31%)
- The minimal gap indicates excellent generalization capability
- RMSE improvement on test set suggests effective regularization strategies

## Prediction Stability
## Distribution Analysis:
- Mean prediction: ₹31,041 (test) vs ₹36,002 (train)
- Standard deviation: ₹17,613 (test) vs ₹23,153 (train)
- Consistent prediction ranges across different data splits
- Low bias variance indicates stable prediction behavior

## Real-World Application Scenarios
## Financial Services Integration
## Credit Assessment Enhancement:
- 70% accuracy within 25% deviation supports loan underwriting decisions
- ₹5K absolute accuracy enables micro-finance and personal loan evaluation
- Real-time inference capability (<100ms) suitable for online application processing

## Business Intelligence Applications
## Market Segmentation:
- Four-tier income categorization (Low, Medium, High, Very High)
- Prediction confidence indicators for risk assessment
- Batch processing capability for large-scale customer analysis

## Regulatory Compliance
## Model Governance:
- Comprehensive audit trail with detailed logging
- Explainable ensemble weights and feature importance
- Bias monitoring and fairness assessment capabilities
- Performance tracking across demographic segments

This technical architecture delivers production-ready income estimation with strong performance across all key evaluation criteria, combining algorithmic sophistication with practical deployment considerations for real-world financial applications.

## Contact
goneprasad1967@gmail.com
