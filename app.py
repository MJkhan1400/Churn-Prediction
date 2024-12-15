import os
import pandas as pd
import joblib
from preprocessing.data_cleaning import clean_data
from preprocessing.feature_engineering import encode_features, scale_features
from utils.oversampling import balance_classes
from train.model_training import train_random_forest
from train.model_evaluation import evaluate_model

# Paths for model and scaler
MODEL_PATH = './models/customer_churn_model.pkl'
SCALER_PATH = './models/scaler.pkl'

# Load the data
data = pd.read_csv('./data/telecom_data.csv')
print("Dataset Loaded Successfully!")

# Preprocess data
data = clean_data(data)
data, label_encoder = encode_features(data)

# Split data into train-test sets
from sklearn.model_selection import train_test_split
X = data.drop('Churn', axis=1)
y = data['Churn']

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    # Load the model and scaler
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully!")

    # Scale the features
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X, _ = scale_features(X, numerical_cols, scaler=scaler)

    # Predict using the loaded model
    y_pred = rf_model.predict(X)
    y_pred_proba = rf_model.predict_proba(X)[:, 1]

    # Show prediction results
    churn_count = sum(y_pred)
    non_churn_count = len(y_pred) - churn_count
    print(f"\nTotal Records: {len(y_pred)}")
    print(f"Predicted Churn: {churn_count}")
    print(f"Predicted Non-Churn: {non_churn_count}")

    # Display a sample of predictions
    results = pd.DataFrame({
        'Churn (Actual)': y,
        'Churn (Predicted)': y_pred,
        'Churn Probability': y_pred_proba
    })
    print("\nSample Predictions:")
    print(results.head(10))

else:
    print("Model and Scaler not found. Proceeding with training...")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balance classes
    X_train, y_train = balance_classes(X_train, y_train)

    # Scale numerical features
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train, scaler = scale_features(X_train, numerical_cols)
    X_test, _ = scale_features(X_test, numerical_cols, scaler=scaler)

    # Train model and save both model and scaler
    rf_model = train_random_forest(X_train, y_train, MODEL_PATH, SCALER_PATH, scaler=scaler)

    # Evaluate model
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

