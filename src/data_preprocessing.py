"""
Data preprocessing utilities for the Bank Customer Churn Prediction project.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Churn_Modelling.csv"
TARGET_COLUMN = "Exited"
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the churn dataset."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records and non-predictive identifier columns."""
    df = df.copy()
    df = df.drop_duplicates()

    columns_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=columns_to_drop)

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical variables."""
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def get_train_test_data(test_size: float = 0.2, random_state: int = 42):
    """Load, clean, and split the data."""
    df = load_data()
    df = clean_data(df)
    X, y = split_features_target(df)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
print("\n\nTesting Before Preprocessing:\n")

df = load_data()
df_clean = clean_data(df)

print(df_clean.head())
print(df_clean.shape)

print("\n\nTrain/Test Split:\n")

X_train, X_test, y_train, y_test = get_train_test_data()

print(X_train.head())
print(y_train.head())