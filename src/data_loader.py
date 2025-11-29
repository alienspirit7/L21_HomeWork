"""
Data loading and splitting module for Iris classification.
Handles CSV loading, preprocessing, and train-test split.
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple


def load_and_split_data(csv_path: str, test_size: float = 0.25, 
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                          np.ndarray, np.ndarray]:
    """
    Load Iris dataset from CSV and split into train/test sets.
    
    Args:
        csv_path: Path to Iris CSV file
        test_size: Proportion of data for testing (default: 0.25)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading Iris dataset from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Extract features (columns 1-4) and labels
    X = df.iloc[:, 1:5].values  # Skip Id column, take 4 features
    y = df.iloc[:, 5].values    # Species column
    
    # Encode species labels to integers
    species_map = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    y_encoded = np.array([species_map[species] for species in y])
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Feature names: ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']")
    logger.info(f"Classes: {list(species_map.keys())}")
    logger.info(f"Class distribution: {np.bincount(y_encoded)}")
    
    # Stratified split to maintain class distribution
    train_indices, test_indices = stratified_split(
        y_encoded, test_size=test_size, random_state=random_state
    )
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y_encoded[train_indices]
    y_test = y_encoded[test_indices]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Train/Test Split (test_size={test_size}):")
    logger.info(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"  Test set:     {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    logger.info(f"  Train class distribution: {np.bincount(y_train)}")
    logger.info(f"  Test class distribution:  {np.bincount(y_test)}")
    logger.info(f"{'='*60}\n")
    
    return X_train, X_test, y_train, y_test


def stratified_split(y: np.ndarray, test_size: float = 0.25, 
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform stratified train-test split to maintain class distribution.
    
    Args:
        y: Labels array
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        train_indices, test_indices
    """
    np.random.seed(random_state)
    
    train_indices = []
    test_indices = []
    
    # Split each class independently
    for class_label in np.unique(y):
        class_indices = np.where(y == class_label)[0]
        np.random.shuffle(class_indices)
        
        n_test = int(len(class_indices) * test_size)
        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])
    
    return np.array(train_indices), np.array(test_indices)
