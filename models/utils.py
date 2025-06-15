import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

def calculate_evaluation_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics with financial context"""
    metrics = {}
    
    # Basic regression metrics
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Financial-specific metrics
    percentage_errors = np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
    metrics['mape'] = np.mean(percentage_errors)
    metrics['median_ape'] = np.median(percentage_errors)
    
    # Coverage metrics (predictions within acceptable ranges)
    metrics['coverage_10_pct'] = np.mean(percentage_errors <= 10) * 100
    metrics['coverage_25_pct'] = np.mean(percentage_errors <= 25) * 100
    metrics['coverage_50_pct'] = np.mean(percentage_errors <= 50) * 100
    
    # Absolute difference metrics
    abs_differences = np.abs(y_true - y_pred)
    metrics['coverage_5k'] = np.mean(abs_differences <= 5000) * 100
    metrics['coverage_10k'] = np.mean(abs_differences <= 10000) * 100
    
    # Income-specific insights
    metrics['mean_prediction'] = np.mean(y_pred)
    metrics['std_prediction'] = np.std(y_pred)
    metrics['prediction_range'] = np.max(y_pred) - np.min(y_pred)
    
    # Bias analysis
    bias = np.mean(y_pred - y_true)
    metrics['bias'] = bias
    metrics['bias_percentage'] = (bias / np.mean(y_true)) * 100
    
    return metrics

def plot_predictions(y_true, y_pred, title="Predictions vs Actual", save_path=None):
    """Enhanced prediction plotting with multiple visualizations"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Main scatter plot
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=30)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Income')
    axes[0, 0].set_ylabel('Predicted Income')
    axes[0, 0].set_title(f'{title}\nR² = {r2_score(y_true, y_pred):.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Income')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Percentage errors
    percentage_errors = np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
    axes[1, 1].hist(percentage_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=25, color='r', linestyle='--', label='25% threshold')
    axes[1, 1].set_xlabel('Absolute Percentage Error (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Percentage Errors')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def analyze_feature_importance(models_dict, feature_names, top_n=20):
    """Analyze and plot feature importance across models"""
    importance_df = pd.DataFrame()
    
    for model_name, model_list in models_dict.items():
        if model_name in ['lgb', 'xgb', 'catboost', 'rf']:
            avg_importance = np.zeros(len(feature_names))
            
            for model in model_list:
                if hasattr(model, 'feature_importances_'):
                    avg_importance += model.feature_importances_
                elif hasattr(model, 'feature_importance'):
                    avg_importance += model.feature_importance()
            
            avg_importance /= len(model_list)
            importance_df[model_name] = avg_importance
    
    importance_df.index = feature_names
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance_mean = importance_df.mean(axis=1).sort_values(ascending=False)
    top_features = importance_mean.head(top_n)
    
    plt.barh(range(len(top_features)), top_features.values)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Average Feature Importance')
    plt.title(f'Top {top_n} Feature Importances (Average across models)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df

def generate_model_report(metrics_dict, save_path='output/model_report.txt'):
    """Generate a comprehensive model performance report"""
    report = []
    report.append("="*60)
    report.append("INCOME ESTIMATION MODEL PERFORMANCE REPORT")
    report.append("="*60)
    
    for phase, metrics in metrics_dict.items():
        report.append(f"\n{phase.upper()} PERFORMANCE:")
        report.append("-" * 30)
        
        # Key metrics
        report.append(f"R² Score: {metrics['r2']:.4f}")
        report.append(f"RMSE: ₹{metrics['rmse']:,.2f}")
        report.append(f"MAE: ₹{metrics['mae']:,.2f}")
        report.append(f"MAPE: {metrics['mape']:.2f}%")
        
        # Coverage analysis
        report.append(f"\nPrediction Accuracy Coverage:")
        report.append(f"  Within 10%: {metrics['coverage_10_pct']:.1f}%")
        report.append(f"  Within 25%: {metrics['coverage_25_pct']:.1f}%")
        report.append(f"  Within ₹5k: {metrics['coverage_5k']:.1f}%")
        
        # Bias analysis
        if metrics['bias'] > 0:
            bias_direction = "overestimating"
        else:
            bias_direction = "underestimating"
        
        report.append(f"\nBias Analysis:")
        report.append(f"  Model is {bias_direction} by ₹{abs(metrics['bias']):,.2f}")
        report.append(f"  Bias percentage: {metrics['bias_percentage']:.2f}%")
    
    report_text = "\n".join(report)
    
    # Save report
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    logger.info(f"Model report saved to {save_path}")
    
    return report_text
