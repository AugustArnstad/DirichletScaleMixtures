import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_abalone_regression_data(
    path="datasets/abalone/abalone.csv",
    target="Rings",
    frac=0.5,
    standardized = False,
    random_state=42,
    test_size=0.2
):
    """
    Load and preprocess the abalone dataset for regression.

    Parameters:
        path (str): Path to the abalone CSV file.
        frac (float): Fraction of data to sample.
        random_state (int): Random seed.
        test_size (float): Proportion of data to use for testing.

    Returns:
        X_train, X_test, y_train, y_test
    """
    column_names = [
        "Sex", "Length", "Diameter", "Height",
        "Whole weight", "Shucked weight", "Viscera weight",
        "Shell weight", "Rings"
    ]

    abalone = pd.read_csv(path, header=None, names=column_names)

    # Map sex to integers: I=1, F=2, M=3
    abalone['Sex'] = abalone['Sex'].map({'I': 1, 'F': 2, 'M': 3})
    abalone = abalone.sample(frac=frac, random_state=random_state).reset_index(drop=True)
    
    X = abalone.drop([target], axis=1)
    numeric_cols = X.select_dtypes(include='number').columns.drop('Sex')
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    y_raw = abalone[target].astype(int)
    y = y_raw.values.astype(float)
    
    if standardized:
        return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
