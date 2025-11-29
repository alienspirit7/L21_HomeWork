"""
Comparison module for NumPy vs Sklearn Naive Bayes implementations.
Generates visualizations and analyzes differences between methods.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


def compare_predictions(y_test: np.ndarray, y_pred_numpy: np.ndarray,
                       y_pred_sklearn: np.ndarray) -> None:
    """
    Compare predictions from both implementations.
    
    Args:
        y_test: True labels
        y_pred_numpy: Predictions from NumPy implementation
        y_pred_sklearn: Predictions from sklearn implementation
    """
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("COMPARISON: NumPy vs Sklearn Implementations")
    logger.info("="*60)
    
    # Calculate metrics for both
    acc_numpy = accuracy_score(y_test, y_pred_numpy)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    logger.info(f"\nAccuracy Comparison:")
    logger.info(f"  NumPy Implementation:  {acc_numpy:.4f} ({acc_numpy*100:.2f}%)")
    logger.info(f"  Sklearn Implementation: {acc_sklearn:.4f} ({acc_sklearn*100:.2f}%)")
    logger.info(f"  Difference: {abs(acc_numpy - acc_sklearn):.6f}")
    
    # Detailed metrics
    precision_np, recall_np, f1_np, _ = precision_recall_fscore_support(
        y_test, y_pred_numpy, average='weighted'
    )
    precision_sk, recall_sk, f1_sk, _ = precision_recall_fscore_support(
        y_test, y_pred_sklearn, average='weighted'
    )
    
    logger.info(f"\nDetailed Metrics (Weighted Average):")
    logger.info(f"  NumPy - Precision: {precision_np:.4f}, Recall: {recall_np:.4f}, F1: {f1_np:.4f}")
    logger.info(f"  Sklearn - Precision: {precision_sk:.4f}, Recall: {recall_sk:.4f}, F1: {f1_sk:.4f}")
    
    # Prediction agreement
    agreement = np.mean(y_pred_numpy == y_pred_sklearn)
    logger.info(f"\nPrediction Agreement: {agreement:.4f} ({agreement*100:.2f}%)")
    
    disagreements = np.where(y_pred_numpy != y_pred_sklearn)[0]
    if len(disagreements) > 0:
        logger.info(f"Number of disagreements: {len(disagreements)}")
        logger.info(f"Disagreement indices: {disagreements[:10]}..." if len(disagreements) > 10 
                   else f"Disagreement indices: {disagreements}")
    else:
        logger.info("Perfect agreement! All predictions match.")
    
    logger.info("="*60 + "\n")


def plot_confusion_matrices(y_test: np.ndarray, y_pred_numpy: np.ndarray,
                            y_pred_sklearn: np.ndarray,
                            save_path: str = "confusion_matrices.png") -> None:
    """
    Plot confusion matrices for both implementations side by side.
    
    Args:
        y_test: True labels
        y_pred_numpy: Predictions from NumPy implementation
        y_pred_sklearn: Predictions from sklearn implementation
        save_path: Path to save the plot
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating confusion matrices...")
    
    cm_numpy = confusion_matrix(y_test, y_pred_numpy)
    cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
    
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
    
    # NumPy confusion matrix
    im1 = ax1.imshow(cm_numpy, cmap='Blues', interpolation='nearest')
    ax1.set_title('NumPy Implementation', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(class_names, rotation=45)
    ax1.set_yticklabels(class_names)
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, str(cm_numpy[i, j]), ha='center', va='center',
                    color='white' if cm_numpy[i, j] > cm_numpy.max()/2 else 'black',
                    fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Sklearn confusion matrix
    im2 = ax2.imshow(cm_sklearn, cmap='Greens', interpolation='nearest')
    ax2.set_title('Sklearn Implementation', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=11)
    ax2.set_ylabel('True Label', fontsize=11)
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.set_yticklabels(class_names)
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, str(cm_sklearn[i, j]), ha='center', va='center',
                    color='white' if cm_sklearn[i, j] > cm_sklearn.max()/2 else 'black',
                    fontweight='bold')
    
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved confusion matrices to: {save_path}")
    plt.close()


def explain_differences() -> None:
    """
    Explain potential differences between NumPy and sklearn implementations.
    """
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("ANALYSIS: Why Results Might Differ")
    logger.info("="*60)
    
    explanation = """
    Expected Outcome: Both implementations should produce VERY SIMILAR or IDENTICAL results.
    
    Reasons for Small Differences (if any):
    
    1. NUMERICAL PRECISION
       - NumPy implementation uses float64 by default
       - Sklearn may use different precision in internal calculations
       - Log-probability calculations can accumulate small rounding errors
    
    2. VARIANCE ESTIMATION
       - NumPy: Uses np.var() which divides by N (biased estimator)
       - Sklearn: May use sample variance (N-1) for better generalization
       - This affects the Gaussian likelihood calculations
    
    3. REGULARIZATION
       - Sklearn adds small epsilon (1e-9) to variance by default
       - Prevents division by zero for features with low variance
       - Our NumPy implementation also adds epsilon for stability
    
    4. PROBABILITY CALCULATIONS
       - NumPy: We use log probabilities to avoid underflow
       - Sklearn: Also uses log probabilities internally
       - Minor differences in log computation can propagate
    
    Expected Results:
    - Accuracy should be within 0.01-0.02 (1-2%) for both methods
       - Iris is a well-separated dataset, typically >95% accuracy
    - Most predictions should agree (>95% agreement)
    - Confusion matrices should be nearly identical
    - Any differences are likely in borderline cases near decision boundaries
    
    If results are SIGNIFICANTLY different (>5% accuracy difference):
    - Check for implementation bugs in NumPy version
    - Verify same train/test split is used
    - Ensure feature scaling is consistent
    """
    
    logger.info(explanation)
    logger.info("="*60 + "\n")
