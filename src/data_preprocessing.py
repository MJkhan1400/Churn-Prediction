import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def preprocess_data(file_path):
    """
    Preprocess the raw churn dataset.
    - Handles missing values
    - Encodes categorical variables
    - Applies scaling to numeric features
    - Balances the dataset using SMOTE
    """
    # Load raw data
    df = pd.read_csv(file_path)

    # Drop irrelevant columns
    if 'customer_id' in df.columns:
        df.drop(columns=['customer_id'], inplace=True)

    # Separate target variable
    X = df.drop(columns=['churn'])
    y = df['churn']

    # Encode binary categorical features (e.g., 'gender', 'SeniorCitizen')
    binary_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for feature in binary_features:
        if feature in X.columns:
            X[feature] = X[feature].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    # One-hot encode multi-class categorical features
    categorical_features = ['InternetService', 'Contract', 'PaymentMethod']
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Define transformers for preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Handle missing values
    X = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X), columns=X.columns)

    # Apply transformations
    X = pd.DataFrame(preprocessor.fit_transform(X).toarray())

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled

