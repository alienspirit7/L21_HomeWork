"""
Manual Gaussian Naive Bayes implementation using NumPy.
Implements Bayes theorem with Gaussian likelihood for classification.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple


class GaussianNaiveBayesNumPy:
    """
    Gaussian Naive Bayes classifier implemented from scratch.
    
    Uses Bayes' theorem: P(y|x) = P(x|y) * P(y) / P(x)
    Assumes features are conditionally independent given class.
    """
    
    def __init__(self):
        self.classes = None
        self.class_priors = {}      # P(y)
        self.means = {}             # μ for each class and feature
        self.variances = {}         # σ² for each class and feature
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Naive Bayes classifier.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
        """
        self.logger.info("="*60)
        self.logger.info("Training Manual NumPy Naive Bayes Classifier")
        self.logger.info("="*60)
        
        self.classes = np.unique(y)
        n_samples = len(y)
        
        # Calculate priors and parameters for each class
        for c in self.classes:
            X_c = X[y == c]
            
            # Prior probability: P(y=c)
            self.class_priors[c] = len(X_c) / n_samples
            
            # Mean and variance for each feature
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)
            
            self.logger.info(f"\nClass {c}:")
            self.logger.info(f"  Samples: {len(X_c)}")
            self.logger.info(f"  Prior P(y={c}): {self.class_priors[c]:.4f}")
            self.logger.info(f"  Feature means: {self.means[c]}")
            self.logger.info(f"  Feature variances: {self.variances[c]}")
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Training Complete!")
        self.logger.info("="*60 + "\n")
        
    def _gaussian_pdf(self, x: np.ndarray, mean: float, var: float) -> np.ndarray:
        """
        Calculate Gaussian probability density function.
        
        PDF: f(x) = 1/√(2πσ²) * exp(-(x-μ)²/(2σ²))
        
        Args:
            x: Input values
            mean: Mean (μ)
            var: Variance (σ²)
            
        Returns:
            Probability densities
        """
        eps = 1e-9  # Prevent division by zero
        coeff = 1.0 / np.sqrt(2.0 * np.pi * (var + eps))
        exponent = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))
        return coeff * exponent
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Test features, shape (n_samples, n_features)
            
        Returns:
            Predicted labels, shape (n_samples,)
        """
        predictions = []
        
        for x in X:
            posteriors = []
            
            for c in self.classes:
                # Start with log prior to avoid numerical underflow
                log_posterior = np.log(self.class_priors[c])
                
                # Add log likelihoods for each feature (independence assumption)
                for feature_idx in range(len(x)):
                    likelihood = self._gaussian_pdf(
                        x[feature_idx],
                        self.means[c][feature_idx],
                        self.variances[c][feature_idx]
                    )
                    log_posterior += np.log(likelihood + 1e-9)
                
                posteriors.append(log_posterior)
            
            # Predict class with maximum posterior probability
            predictions.append(self.classes[np.argmax(posteriors)])
        
        return np.array(predictions)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy (fraction of correct predictions)
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


def plot_feature_distributions(X_train: np.ndarray, y_train: np.ndarray,
                               save_path: str = "numpy_feature_distributions.png") -> None:
    """
    Create histograms showing feature distributions for each class.
    
    Args:
        X_train: Training features
        y_train: Training labels
        save_path: Path to save the plot
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating feature distribution histograms...")
    
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feature Distributions by Class (NumPy Implementation)', 
                 fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        for class_label in range(3):
            X_class = X_train[y_train == class_label]
            ax.hist(X_class[:, idx], bins=15, alpha=0.6, 
                   label=class_names[class_label], color=colors[class_label])
        
        ax.set_xlabel(feature_names[idx], fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{feature_names[idx]} Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved feature distribution plot to: {save_path}")
    plt.close()
