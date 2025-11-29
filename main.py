"""
Main orchestration script for Iris Classification with Naive Bayes.
Compares manual NumPy implementation with sklearn's GaussianNB.

Author: Backend Developer
Date: 2025-11-30
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import load_and_split_data
from naive_bayes_numpy import GaussianNaiveBayesNumPy, plot_feature_distributions
from naive_bayes_sklearn import SklearnNaiveBayes
from comparison import compare_predictions, plot_confusion_matrices, explain_differences


def setup_logging() -> None:
    """Configure logging to both console and file."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler('logs/iris_classification.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def main():
    """Main execution pipeline."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info(" IRIS CLASSIFICATION: NAIVE BAYES COMPARISON ".center(70, '='))
    logger.info("="*70)
    logger.info("Comparing Manual NumPy vs Sklearn Implementations\n")
    
    # Step 1: Load and split data
    logger.info("STEP 1: Loading and Splitting Data")
    logger.info("-" * 70)
    csv_path = 'Iris.csv'
    X_train, X_test, y_train, y_test = load_and_split_data(csv_path, test_size=0.25)
    
    # Step 2: Train NumPy implementation
    logger.info("\nSTEP 2: Training Manual NumPy Implementation")
    logger.info("-" * 70)
    nb_numpy = GaussianNaiveBayesNumPy()
    nb_numpy.fit(X_train, y_train)
    
    # Generate feature distribution plots
    plot_feature_distributions(X_train, y_train, 'logs/numpy_feature_distributions.png')
    
    # Step 3: Test NumPy implementation
    logger.info("\nSTEP 3: Testing NumPy Implementation")
    logger.info("-" * 70)
    y_pred_numpy = nb_numpy.predict(X_test)
    acc_numpy = nb_numpy.score(X_test, y_test)
    logger.info(f"NumPy Model Test Accuracy: {acc_numpy:.4f} ({acc_numpy*100:.2f}%)")
    logger.info(f"Correct predictions: {sum(y_pred_numpy == y_test)}/{len(y_test)}\n")
    
    # Step 4: Train Sklearn implementation
    logger.info("\nSTEP 4: Training Sklearn Implementation")
    logger.info("-" * 70)
    nb_sklearn = SklearnNaiveBayes()
    nb_sklearn.fit(X_train, y_train)
    
    # Step 5: Test Sklearn implementation
    logger.info("\nSTEP 5: Testing Sklearn Implementation")
    logger.info("-" * 70)
    y_pred_sklearn = nb_sklearn.predict(X_test)
    acc_sklearn = nb_sklearn.score(X_test, y_test)
    logger.info(f"Sklearn Model Test Accuracy: {acc_sklearn:.4f} ({acc_sklearn*100:.2f}%)")
    logger.info(f"Correct predictions: {sum(y_pred_sklearn == y_test)}/{len(y_test)}\n")
    
    # Detailed evaluation
    nb_sklearn.detailed_evaluation(X_test, y_test)
    
    # Step 6: Compare results
    logger.info("\nSTEP 6: Comparing Implementations")
    logger.info("-" * 70)
    compare_predictions(y_test, y_pred_numpy, y_pred_sklearn)
    
    # Step 7: Generate visualizations
    logger.info("\nSTEP 7: Generating Comparison Visualizations")
    logger.info("-" * 70)
    plot_confusion_matrices(y_test, y_pred_numpy, y_pred_sklearn, 
                           'logs/confusion_matrices.png')
    
    # Step 8: Explain differences
    logger.info("\nSTEP 8: Analysis and Explanation")
    logger.info("-" * 70)
    explain_differences()
    
    # Final summary
    logger.info("="*70)
    logger.info(" EXECUTION COMPLETE ".center(70, '='))
    logger.info("="*70)
    logger.info(f"\nResults Summary:")
    logger.info(f"  NumPy Accuracy:   {acc_numpy*100:.2f}%")
    logger.info(f"  Sklearn Accuracy: {acc_sklearn*100:.2f}%")
    logger.info(f"  Agreement Rate:   {sum(y_pred_numpy == y_pred_sklearn)/len(y_test)*100:.2f}%")
    logger.info(f"\nGenerated Files:")
    logger.info(f"  - logs/iris_classification.log")
    logger.info(f"  - logs/numpy_feature_distributions.png")
    logger.info(f"  - logs/confusion_matrices.png")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
