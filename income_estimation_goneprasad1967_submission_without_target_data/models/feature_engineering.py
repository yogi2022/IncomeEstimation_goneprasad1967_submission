import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.selected_features = None
        self.outlier_detector = None
        self.feature_selector = None
        self.expected_columns = None
        self.column_medians = {}
        self.is_fitted = False

    def _ensure_required_columns(self, df):
        """Ensure all required columns exist with appropriate defaults"""
        if self.expected_columns is None:
            # First time (training) - store expected columns
            self.expected_columns = df.columns.tolist()
            return df
        
        df_processed = df.copy()
        
        # Add missing columns with appropriate defaults
        for col in self.expected_columns:
            if col not in df_processed.columns:
                if col == 'target':
                    continue  # Skip target during inference
                elif col in ['unique_id']:
                    continue  # Skip ID columns
                elif col in ['score_comments', 'score_type']:
                    default_val = 'Not Available' if col == 'score_comments' else 'PERFORM CONSUMER 2.0'
                    df_processed[col] = default_val
                elif col in ['gender', 'marital_status', 'city', 'state', 'residence_ownership', 
                           'device_model', 'device_category', 'platform', 'device_manufacturer']:
                    df_processed[col] = 'Unknown'
                else:
                    # Numerical columns - use stored median or 0
                    default_val = self.column_medians.get(col, 0.0)
                    df_processed[col] = default_val
        
        return df_processed

    def create_financial_ratios(self, df):
        """Create advanced financial ratio features"""
        df_processed = df.copy()

        # Credit utilization ratios
        balance_cols = [col for col in df.columns if 'balance_' in col and col != 'business_balance']
        credit_limit_cols = [col for col in df.columns if 'credit_limit_' in col]

        if balance_cols and credit_limit_cols:
            total_balance = df[balance_cols].sum(axis=1)
            total_credit = df[credit_limit_cols].sum(axis=1)
            df_processed['overall_utilization'] = total_balance / (total_credit + 1e-8)
            df_processed['credit_diversity'] = (df[credit_limit_cols] > 0).sum(axis=1)

        # Loan features
        loan_cols = [col for col in df.columns if 'loan_amt_' in col]
        if loan_cols:
            df_processed['total_loan_exposure'] = df[loan_cols].sum(axis=1)
            df_processed['avg_loan_size'] = df[loan_cols].mean(axis=1)
            df_processed['loan_concentration'] = df[loan_cols].max(axis=1) / (df[loan_cols].sum(axis=1) + 1e-8)

        # EMI ratios
        emi_cols = [col for col in df.columns if 'total_emi_' in col]
        if emi_cols and 'age' in df.columns:
            total_emi = df[emi_cols].sum(axis=1)
            df_processed['emi_burden_ratio'] = total_emi / (df['age'] * 1000 + 1e-8)

        # Credit behavior
        inquiry_cols = [col for col in df.columns if 'total_inquir' in col]
        if inquiry_cols:
            df_processed['credit_appetite'] = df[inquiry_cols].sum(axis=1)
            df_processed['recent_credit_seeking'] = df.get('total_inquiries_recent', 0)

        # Repayment patterns
        repayment_cols = [col for col in df.columns if 'repayment_' in col]
        if repayment_cols:
            df_processed['repayment_consistency'] = df[repayment_cols].std(axis=1)
            if len(repayment_cols) > 1:
                df_processed['repayment_trend'] = df[repayment_cols].iloc[:, -1] - df[repayment_cols].iloc[:, 0]
            else:
                df_processed['repayment_trend'] = 0

        return df_processed

    def create_interaction_features(self, df):
        """Create interaction features"""
        df_processed = df.copy()

        # Age interactions
        if 'age' in df.columns:
            if 'credit_score' in df.columns:
                df_processed['age_credit_interaction'] = df['age'] * df['credit_score']
                df_processed['credit_maturity'] = df['credit_score'] / (df['age'] + 1e-8)

            # Age bins
            df_processed['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 60, 100], labels=[1, 2, 3, 4, 5])

        # Geographic patterns
        if 'pin code' in df.columns:
            df_processed['pin_code_area'] = df['pin code'].astype(str).str[:3]

        return df_processed

    def encode_categorical_features(self, df, fit=True):
        """Enhanced categorical encoding"""
        df_processed = df.copy()

        categorical_cols = ['gender', 'marital_status', 'city', 'state',
                           'residence_ownership', 'device_category', 'platform',
                           'device_manufacturer', 'score_type', 'score_comments']

        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    # Target encoding for high cardinality
                    if col in ['city', 'state', 'device_manufacturer'] and 'target' in df.columns:
                        global_mean = df['target'].mean()
                        counts = df.groupby(col).size()
                        target_mean = df.groupby(col)['target'].mean()

                        smoothing = 10
                        regularized_target = (target_mean * counts + global_mean * smoothing) / (counts + smoothing)
                        df_processed[f'{col}_target_encoded'] = df[col].map(regularized_target).astype('float64')
                        self.encoders[f'{col}_target'] = regularized_target

                    # Frequency encoding
                    freq_map = df[col].value_counts(normalize=True)
                    df_processed[f'{col}_frequency'] = df[col].map(freq_map).astype('float64')
                    self.encoders[f'{col}_freq'] = freq_map

                    # Label encoding for low cardinality
                    if df[col].nunique() <= 10:
                        le = LabelEncoder()
                        df_processed[f'{col}_encoded'] = le.fit_transform(df[col].astype(str)).astype('float64')
                        self.encoders[f'{col}_label'] = le

                else:
                    # Apply stored encodings
                    if f'{col}_target' in self.encoders:
                        df_processed[f'{col}_target_encoded'] = df[col].map(self.encoders[f'{col}_target']).fillna(0).astype('float64')

                    if f'{col}_freq' in self.encoders:
                        df_processed[f'{col}_frequency'] = df[col].map(self.encoders[f'{col}_freq']).fillna(0).astype('float64')

                    if f'{col}_label' in self.encoders:
                        try:
                            df_processed[f'{col}_encoded'] = self.encoders[f'{col}_label'].transform(df[col].astype(str)).astype('float64')
                        except ValueError:
                            # Handle unseen categories
                            encoded_values = []
                            for val in df[col].astype(str):
                                try:
                                    encoded_values.append(self.encoders[f'{col}_label'].transform([val])[0])
                                except ValueError:
                                    encoded_values.append(-1)
                            df_processed[f'{col}_encoded'] = pd.Series(encoded_values, index=df.index).astype('float64')

        return df_processed

    def advanced_feature_selection(self, X, y, k=150):
        """Advanced feature selection"""
        if self.feature_selector is None:
            # Combine statistical and mutual information
            stat_selector = SelectKBest(score_func=f_regression, k=min(k//2, X.shape[1]//2))
            mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(k//2, X.shape[1]//2))

            stat_features = stat_selector.fit_transform(X, y)
            mi_features = mi_selector.fit_transform(X, y)

            stat_mask = stat_selector.get_support()
            mi_mask = mi_selector.get_support()
            combined_mask = stat_mask | mi_mask

            self.selected_features = combined_mask
            return X.loc[:, combined_mask], X.columns[combined_mask]
        else:
            return X.loc[:, self.selected_features], X.columns[self.selected_features]

    def handle_missing_values(self, df):
        """Enhanced missing value handling"""
        df_processed = df.copy()

        # Numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['target', 'unique_id']:
                median_val = df[col].median()
                
                # Store median for future use
                if col not in self.column_medians:
                    self.column_medians[col] = median_val
                
                df_processed[col] = df[col].fillna(median_val)

                # Create missing indicator
                if df[col].isna().sum() > 0:
                    df_processed[f'{col}_was_missing'] = df[col].isna().astype(int)

        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'unique_id':
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df_processed[col] = df[col].fillna(mode_val)

        return df_processed

    def fit_transform(self, df):
        """Complete feature engineering pipeline for training"""
        # Store expected columns
        self.expected_columns = df.columns.tolist()
        self.is_fitted = True
        
        # Process pipeline
        df_processed = self.handle_missing_values(df)
        df_processed = self.create_financial_ratios(df_processed)
        df_processed = self.create_interaction_features(df_processed)
        df_processed = self.encode_categorical_features(df_processed, fit=True)

        # Remove original categorical columns
        cols_to_drop = ['unique_id', 'gender', 'marital_status', 'city', 'state',
                       'residence_ownership', 'device_category', 'platform',
                       'device_manufacturer', 'score_type', 'score_comments',
                       'device_model', 'pin code', 'age_group', 'pin_code_area']
        cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
        df_processed = df_processed.drop(columns=cols_to_drop)

        # Feature selection
        if 'target' in df_processed.columns:
            X = df_processed.drop(['target'], axis=1)
            y = df_processed['target']
            X_selected, selected_cols = self.advanced_feature_selection(X, y)
            df_processed = pd.concat([X_selected, y], axis=1)

        self.feature_names = df_processed.columns.tolist()
        return df_processed

    def transform(self, df):
        """Transform new data using fitted pipeline"""
        # Ensure required columns exist
        df_processed = self._ensure_required_columns(df)
        
        # Apply transformations
        df_processed = self.handle_missing_values(df_processed)
        df_processed = self.create_financial_ratios(df_processed)
        df_processed = self.create_interaction_features(df_processed)
        df_processed = self.encode_categorical_features(df_processed, fit=False)

        # Remove categorical columns
        cols_to_drop = ['unique_id', 'gender', 'marital_status', 'city', 'state',
                       'residence_ownership', 'device_category', 'platform',
                       'device_manufacturer', 'score_type', 'score_comments',
                       'device_model', 'pin code', 'age_group', 'pin_code_area']
        cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
        df_processed = df_processed.drop(columns=cols_to_drop)

        # Ensure same features as training
        if hasattr(self, 'feature_names') and self.feature_names:
            missing_cols = set(self.feature_names) - set(df_processed.columns)
            for col in missing_cols:
                if col != 'target':
                    df_processed[col] = 0

            feature_cols = [col for col in self.feature_names if col != 'target' and col in df_processed.columns]
            df_processed = df_processed[feature_cols]

        return df_processed
