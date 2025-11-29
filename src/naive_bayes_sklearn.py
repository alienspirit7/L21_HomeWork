"""
Sklearn-based Gaussian Naive Bayes implementation.
Uses scikit-learn's built-in GaussianNB classifier for comparison.
"""
import numpy as np
import logging
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix


class SklearnNaiveBayes:
    """
    Wrapper for sklearn's Gaussian Naive Bayes classifier.
    Provides logging and parameter inspection.
    """
    
    def __init__(self):
        self.model = GaussianNB()
        self.logger = logging.getLogger(__name__)
        self.classes = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the sklearn Naive Bayes classifier.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
        """
        self.logger.info("="*60)
        self.logger.info("Training Sklearn Naive Bayes Classifier")
        self.logger.info("="*60)
        
        # Train the model
        self.model.fit(X, y)
        self.classes = self.model.classes_
        
        # Log learned parameters
        self.logger.info(f"\nClasses: {self.classes}")
        self.logger.info(f"Number of features: {X.shape[1]}")
        
        for idx, c in enumerate(self.classes):
            self.logger.info(f"\nClass {c}:")
            self.logger.info(f"  Prior P(y={c}): {self.model.class_prior_[idx]:.4f}")
            self.logger.info(f"  Feature means (θ): {self.model.theta_[idx]}")
            self.logger.info(f"  Feature variance (σ²): {self.model.var_[idx]}")
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Training Complete!")
        self.logger.info("="*60 + "\n")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Test features, shape (n_samples, n_features)
            
        Returns:
            Predicted labels, shape (n_samples,)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X: Test features
            
        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy (fraction of correct predictions)
        """
        return self.model.score(X, y)
    
    def detailed_evaluation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Print detailed evaluation metrics.
        
        Args:
            X: Test features
            y: True labels
        """
        predictions = self.predict(X)
        
        self.logger.info("="*60)
        self.logger.info("Sklearn Model - Detailed Evaluation")
        self.logger.info("="*60)
        
        # Confusion Matrix
        cm = confusion_matrix(y, predictions)
        self.logger.info(f"\nConfusion Matrix:\n{cm}")
        
        # Classification Report
        class_names = ['Setosa', 'Versicolor', 'Virginica']
        report = classification_report(y, predictions, target_names=class_names)
        self.logger.info(f"\nClassification Report:\n{report}")
        
        self.logger.info("="*60 + "\n")
