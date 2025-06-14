import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import optuna
from optuna.integration import LightGBMPruningCallback
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OptimizedEnsembleModel:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.weights = {}
        self.cv_scores = {}
        self.best_params = {}
        self.scalers = {}
        self.outlier_detector = None
        
    def create_stratified_folds(self, y, n_splits=5):
        """Create stratified folds for regression to address overfitting"""
        try:
            y_binned = pd.qcut(y, q=n_splits, duplicates='drop', labels=False)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            folds = []
            for train_idx, val_idx in skf.split(np.zeros(len(y)), y_binned):
                folds.append((train_idx, val_idx))
            return folds
        except:
            # Fallback to regular KFold if stratification fails
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            return list(kf.split(y))
    
    def detect_and_handle_outliers(self, X, y, contamination=0.1):
        """Advanced outlier detection and handling with categorical column support"""
        # Select only numeric columns for outlier detection
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for outlier detection")
            return X.copy(), y.copy()
        
        if self.outlier_detector is None:
            self.outlier_detector = IsolationForest(contamination=contamination, random_state=42)
            # Use only numeric columns for outlier detection
            outliers = self.outlier_detector.fit_predict(X[numeric_cols].fillna(0))
        else:
            outliers = self.outlier_detector.predict(X[numeric_cols].fillna(0))
        
        # Cap outliers instead of removing them - only for numeric columns
        X_processed = X.copy()
        y_processed = y.copy()
        
        for col in numeric_cols:
            if col in X.columns:
                q1 = X[col].quantile(0.01)
                q99 = X[col].quantile(0.99)
                X_processed[col] = X[col].clip(lower=q1, upper=q99)
        
        # Cap target outliers
        y_q1 = y.quantile(0.01)
        y_q99 = y.quantile(0.99)
        y_processed = y.clip(lower=y_q1, upper=y_q99)
        
        return X_processed, y_processed
    
    def validate_feature_types(self, X):
        """Validate that all features are numeric before model training"""
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            logger.warning(f"Found non-numeric columns: {list(non_numeric_cols)}")
            # Convert to numeric or drop
            for col in non_numeric_cols:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    logger.info(f"Converted {col} to numeric")
                except:
                    X = X.drop(columns=[col])
                    logger.warning(f"Dropped non-convertible column: {col}")
        return X
    
    def optimize_lightgbm_params(self, X, y, n_trials=50):
        """Optimize LightGBM with Optuna Bayesian optimization - FIXED VERSION"""
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
                'random_state': 42,
                'verbose': -1,
                'force_col_wise': True
            }
            
            folds = self.create_stratified_folds(y, n_splits=3)
            rmse_scores = []
            
            for train_idx, val_idx in folds:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Fixed: Use callbacks instead of deprecated parameters [17][18]
                callbacks = [
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0)  # 0 means no logging
                ]
                
                model = lgb.train(
                    params, 
                    train_data, 
                    valid_sets=[val_data], 
                    num_boost_round=1000,
                    callbacks=callbacks
                )
                
                y_pred = model.predict(X_val, num_iteration=model.best_iteration)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
        
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params
    
    def optimize_xgboost_params(self, X, y, n_trials=50):
        """Optimize XGBoost with Optuna"""
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'random_state': 42,
                'verbosity': 0
            }
            
            folds = self.create_stratified_folds(y, n_splits=3)
            rmse_scores = []
            
            for train_idx, val_idx in folds:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params, n_estimators=1000)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params
    
    def optimize_catboost_params(self, X, y, n_trials=50):
        """Optimize CatBoost with Optuna"""
        def objective(trial):
            params = {
                'loss_function': 'RMSE',
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'random_state': 42,
                'verbose': False
            }
            
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
            elif params['bootstrap_type'] == 'Bernoulli':
                params['subsample'] = trial.suggest_float('subsample', 0.1, 1)
                
            folds = self.create_stratified_folds(y, n_splits=3)
            rmse_scores = []
            
            for train_idx, val_idx in folds:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = cb.CatBoostRegressor(**params, iterations=1000)
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params

    def train_lightgbm(self, X, y, params):
        """Train LightGBM model with stratified CV - FIXED VERSION"""
        folds = self.create_stratified_folds(y, n_splits=self.config['training']['cv_folds'])
        predictions = np.zeros(len(X))
        models = []

        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
            # Fixed: Use callbacks instead of deprecated parameters [17][18]
            callbacks = [
                lgb.early_stopping(self.config['training']['early_stopping_rounds']),
                lgb.log_evaluation(0)  # Silent logging
            ]
            
            model = lgb.train(
                params, 
                train_data,
                valid_sets=[val_data],
                num_boost_round=2000,
                callbacks=callbacks
            )
        
            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            predictions[val_idx] = val_pred
            models.append(model)
        
        self.models['lgb'] = models
        return predictions
    
    def train_xgboost(self, X, y, params):
        """Train XGBoost model with stratified CV"""
        folds = self.create_stratified_folds(y, n_splits=self.config['training']['cv_folds'])
        predictions = np.zeros(len(X))
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params, n_estimators=2000)
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)], 
                     early_stopping_rounds=self.config['training']['early_stopping_rounds'],
                     verbose=False)
            
            val_pred = model.predict(X_val)
            predictions[val_idx] = val_pred
            models.append(model)
        
        self.models['xgb'] = models
        return predictions
    
    def train_catboost(self, X, y, params):
        """Train CatBoost model with stratified CV"""
        folds = self.create_stratified_folds(y, n_splits=self.config['training']['cv_folds'])
        predictions = np.zeros(len(X))
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = cb.CatBoostRegressor(**params, iterations=2000)
            model.fit(X_train, y_train, 
                     eval_set=(X_val, y_val), 
                     early_stopping_rounds=self.config['training']['early_stopping_rounds'],
                     verbose=False)
            
            val_pred = model.predict(X_val)
            predictions[val_idx] = val_pred
            models.append(model)
        
        self.models['catboost'] = models
        return predictions
    
    def train_neural_network(self, X, y):
        """Train Neural Network model with improved architecture"""
        # Scale features for neural network
        if 'nn_scaler' not in self.scalers:
            self.scalers['nn_scaler'] = RobustScaler()
            X_scaled = self.scalers['nn_scaler'].fit_transform(X)
        else:
            X_scaled = self.scalers['nn_scaler'].transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        folds = self.create_stratified_folds(y, n_splits=self.config['training']['cv_folds'])
        predictions = np.zeros(len(X))
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = MLPRegressor(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                alpha=0.01,  # L2 regularization
                random_state=42
            )
            
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            predictions[val_idx] = val_pred
            models.append(model)
        
        self.models['nn'] = models
        return predictions
    
    def train_random_forest(self, X, y):
        """Train Random Forest model for diversity"""
        folds = self.create_stratified_folds(y, n_splits=self.config['training']['cv_folds'])
        predictions = np.zeros(len(X))
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            predictions[val_idx] = val_pred
            models.append(model)
        
        self.models['rf'] = models
        return predictions

    def calculate_optimal_weights(self, predictions_dict, y_true):
        """Calculate optimal ensemble weights using Bayesian optimization"""
        def objective(trial):
            weights = []
            for i, model_name in enumerate(predictions_dict.keys()):
                weight = trial.suggest_float(f'weight_{model_name}', 0.0, 1.0)
                weights.append(weight)
            
            # Normalize weights
            weights = np.array(weights)
            if weights.sum() == 0:
                weights = np.ones(len(weights)) / len(weights)
            else:
                weights = weights / weights.sum()
            
            ensemble_pred = np.zeros(len(y_true))
            for i, (model_name, preds) in enumerate(predictions_dict.items()):
                ensemble_pred += weights[i] * preds
            
            return mean_squared_error(y_true, ensemble_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=200, show_progress_bar=False)
        
        # Extract optimized weights
        optimal_weights = []
        for model_name in predictions_dict.keys():
            optimal_weights.append(study.best_params[f'weight_{model_name}'])
        
        # Normalize
        optimal_weights = np.array(optimal_weights)
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        return optimal_weights
    
    def fit(self, X, y):
        """Train ensemble model with all optimizations"""
        print("Starting advanced ensemble training with optimizations...")
        
        # Validate feature types first
        X = self.validate_feature_types(X)
        
        # Handle outliers with categorical-aware method
        X_processed, y_processed = self.detect_and_handle_outliers(X, y)
        
        # Optimize hyperparameters
        print("Optimizing hyperparameters with Optuna...")
        print("Optimizing LightGBM...")
        self.best_params['lgb'] = self.optimize_lightgbm_params(X_processed, y_processed, n_trials=30)
        
        print("Optimizing XGBoost...")
        self.best_params['xgb'] = self.optimize_xgboost_params(X_processed, y_processed, n_trials=30)
        
        print("Optimizing CatBoost...")
        self.best_params['catboost'] = self.optimize_catboost_params(X_processed, y_processed, n_trials=30)
        
        # Train models with optimized parameters
        print("Training models with optimized parameters...")
        lgb_preds = self.train_lightgbm(X_processed, y_processed, self.best_params['lgb'])
        xgb_preds = self.train_xgboost(X_processed, y_processed, self.best_params['xgb'])
        cat_preds = self.train_catboost(X_processed, y_processed, self.best_params['catboost'])
        nn_preds = self.train_neural_network(X_processed, y_processed)
        rf_preds = self.train_random_forest(X_processed, y_processed)
        
        # Store predictions
        predictions_dict = {
            'lgb': lgb_preds,
            'xgb': xgb_preds,
            'catboost': cat_preds,
            'nn': nn_preds,
            'rf': rf_preds
        }
        
        # Calculate optimal weights using Bayesian optimization
        print("Optimizing ensemble weights...")
        optimal_weights = self.calculate_optimal_weights(predictions_dict, y_processed)
        self.weights = dict(zip(predictions_dict.keys(), optimal_weights))
        
        # Calculate CV scores
        for model_name, preds in predictions_dict.items():
            rmse = np.sqrt(mean_squared_error(y_processed, preds))
            mae = mean_absolute_error(y_processed, preds)
            r2 = r2_score(y_processed, preds)
            self.cv_scores[model_name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        
        # Ensemble score
        ensemble_pred = np.zeros(len(y_processed))
        for model_name, preds in predictions_dict.items():
            ensemble_pred += self.weights[model_name] * preds
        
        rmse = np.sqrt(mean_squared_error(y_processed, ensemble_pred))
        mae = mean_absolute_error(y_processed, ensemble_pred)
        r2 = r2_score(y_processed, ensemble_pred)
        self.cv_scores['ensemble'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        
        print(f"Optimized Ensemble CV RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        print(f"Model weights: {self.weights}")
        return self
    
    def predict(self, X):
        """Make predictions using optimized ensemble"""
        # Validate feature types
        X = self.validate_feature_types(X)
        
        # Handle outliers in prediction data
        X_processed = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in X.columns:
                q1 = X[col].quantile(0.01)
                q99 = X[col].quantile(0.99)
                X_processed[col] = X[col].clip(lower=q1, upper=q99)
        
        predictions = {}
        
        # Get predictions from each model
        for model_name, model_list in self.models.items():
            model_preds = np.zeros(len(X))
            for model in model_list:
                if model_name == 'lgb':
                    model_preds += model.predict(X_processed, num_iteration=model.best_iteration)
                elif model_name == 'nn':
                    X_scaled = self.scalers['nn_scaler'].transform(X_processed)
                    model_preds += model.predict(X_scaled)
                else:
                    model_preds += model.predict(X_processed)
            model_preds /= len(model_list)
            predictions[model_name] = model_preds
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(X))
        for model_name, preds in predictions.items():
            ensemble_pred += self.weights[model_name] * preds
        
        # Ensure non-negative predictions
        ensemble_pred = np.maximum(ensemble_pred, 0)
        
        return ensemble_pred
    
    def save_model(self, filepath):
        """Save the entire optimized ensemble model"""
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'cv_scores': self.cv_scores,
            'config': self.config,
            'best_params': self.best_params,
            'scalers': self.scalers,
            'outlier_detector': self.outlier_detector
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load the optimized ensemble model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.weights = model_data['weights']
        self.cv_scores = model_data['cv_scores']
        self.config = model_data['config']
        self.best_params = model_data.get('best_params', {})
        self.scalers = model_data.get('scalers', {})
        self.outlier_detector = model_data.get('outlier_detector', None)
