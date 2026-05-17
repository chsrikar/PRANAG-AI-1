import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

def plot_training_losses(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot Training Loss Curves: total, data, physics, boundary losses vs epochs."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history.get('total', [])) + 1)
    if not epochs:
        return
        
    plt.plot(epochs, history.get('total', []), label='Total Loss', color='black', linewidth=2)
    if 'data' in history: plt.plot(epochs, history['data'], label='Data Loss', alpha=0.8)
    if 'physics' in history: plt.plot(epochs, history['physics'], label='Physics Loss', alpha=0.8)
    if 'boundary' in history: plt.plot(epochs, history['boundary'], label='Boundary Loss', alpha=0.8)
    
    plt.title('Training Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_lambda_evolution(lambda_history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot Lambda Evolution (from AdaptiveLoss): λ1, λ2, λ3 vs epochs."""
    plt.figure(figsize=(10, 6))
    if not lambda_history or 'lambda1' not in lambda_history:
        print("No lambda history available to plot.")
        plt.close()
        return
        
    epochs = range(1, len(lambda_history['lambda1']) + 1)
    plt.plot(epochs, lambda_history['lambda1'], label='λ1 (Data)', color='blue')
    plt.plot(epochs, lambda_history['lambda2'], label='λ2 (Physics)', color='orange')
    plt.plot(epochs, lambda_history['lambda3'], label='λ3 (Boundary)', color='green')
    
    plt.title('Lambda Evolution (Adaptive Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
    """Plot Prediction vs Actual: scatter plot (y_true vs y_pred)."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color='teal')
    
    # 45-degree line
    if len(y_true) > 0 and len(y_pred) > 0:
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title('Prediction vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
    """Plot Residual Plot: error distribution."""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='purple', alpha=0.7, edgecolor='black')
    
    plt.title('Residual Error Distribution')
    plt.xlabel('Residual (y_true - y_pred)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_constraint_violation(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot Constraint Violation Plot: constraint loss over time."""
    plt.figure(figsize=(10, 6))
    if 'constraint' not in history or not history['constraint']:
        print("No constraint history available to plot.")
        plt.close()
        return
        
    epochs = range(1, len(history['constraint']) + 1)
    plt.plot(epochs, history['constraint'], label='Constraint Violation Penalty', color='red', linewidth=2)
    
    plt.title('Constraint Violation Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Constraint Penalty (Loss)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_surrogate_performance(r2_scores: Dict[str, float], times: Dict[str, float], save_path: Optional[str] = None):
    """Plot Surrogate Model Performance: R² score bar chart, prediction time vs accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    models = list(r2_scores.keys())
    scores = [r2_scores[m] for m in models]
    
    # R2 Score Bar Chart
    bars = ax1.bar(models, scores, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974'][:len(models)])
    ax1.set_title('Surrogate Model R² Performance')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1.1)
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    # Time vs Accuracy Scatter
    pred_times = [times.get(m, 0) for m in models]
    ax2.scatter(pred_times, scores, s=100, color='darkred', zorder=5)
    
    for i, model in enumerate(models):
        ax2.annotate(model, (pred_times[i], scores[i]), xytext=(5, 5), textcoords='offset points')
        
    ax2.set_title('Prediction Time vs Accuracy')
    ax2.set_xlabel('Inference Time (ms)')
    ax2.set_ylabel('R² Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_all(history: Dict[str, List[float]], 
             lambda_history: Dict[str, List[float]], 
             y_true: np.ndarray, 
             y_pred: np.ndarray, 
             r2_scores: Dict[str, float], 
             times: Dict[str, float],
             output_dir: str = "."):
    """Generate and save all plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    plot_training_losses(history, os.path.join(output_dir, "training_losses.png"))
    plot_prediction_vs_actual(y_true, y_pred, os.path.join(output_dir, "prediction_vs_actual.png"))
    plot_lambda_evolution(lambda_history, os.path.join(output_dir, "lambda_evolution.png"))
    plot_surrogate_performance(r2_scores, times, os.path.join(output_dir, "surrogate_performance.png"))
