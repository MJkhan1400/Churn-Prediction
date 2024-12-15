import os
import pandas as pd
import joblib
from preprocessing.data_cleaning import clean_data
from preprocessing.feature_engineering import encode_features, scale_features
from train.model_training import train_random_forest
from train.model_evaluation import evaluate_model

# Paths for model and scaler
MODEL_PATH = './models/customer_churn_model.pkl'
SCALER_PATH = './models/scaler.pkl'

# Load the training dataset
data = pd.read_csv('./data/telecom_data.csv')
print("Training Dataset Loaded Successfully!")

# Load the actual dataset for prediction
actual_data = pd.read_csv('./data/Actual_Data.csv')
print("Actual Dataset for Prediction Loaded Successfully!")

# Preprocess the training dataset
data = clean_data(data)  # Clean the training data

# Separate the target column for training
X = data.drop('Churn', axis=1)
y = data['Churn']

# Encode the training dataset features
X, label_encoder = encode_features(X)  # Remove the target column before encoding

# Train-test split for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    # Load the pre-trained model and scaler
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully!")

    # Process actual_data for prediction (no cleaning, only encoding and scaling)
    actual_data_encoded, _ = encode_features(actual_data)  # Encode features for actual_data
    numerical_cols = actual_data_encoded.select_dtypes(include=['float64', 'int64']).columns
    actual_data_scaled, _ = scale_features(actual_data_encoded, numerical_cols, scaler=scaler)

    # Predict using the loaded model
    y_pred = rf_model.predict(actual_data_scaled)
    y_pred_proba = rf_model.predict_proba(actual_data_scaled)[:, 1]

    # Show prediction results
    churn_count = sum(y_pred)
    non_churn_count = len(y_pred) - churn_count
    print(f"\nTotal Records in Actual Dataset: {len(y_pred)}")
    print(f"Predicted Churn: {churn_count}")
    print(f"Predicted Non-Churn: {non_churn_count}")

    # Display a sample of predictions
    results = pd.DataFrame({
        'Customer ID': actual_data.index,  # Replace with a unique identifier column if available
        'Churn (Predicted)': y_pred,
        'Churn Probability': y_pred_proba
    })
    print("\nSample Predictions:")
    print(results.head(10))

    # Optionally, save the results to a CSV file
    results.to_csv('./data/Predicted_Churn_Results.csv', index=False)
    print("\nPredictions saved to 'Predicted_Churn_Results.csv'.")

else:
    print("Model and Scaler not found. Training the model first...")

    # Scale numerical features for training
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train, scaler = scale_features(X_train, numerical_cols)
    X_test, _ = scale_features(X_test, numerical_cols, scaler=scaler)

    # Train the Random Forest model and save it along with the scaler
    rf_model = train_random_forest(X_train, y_train, MODEL_PATH, SCALER_PATH, scaler=scaler)

    # Evaluate the trained model
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

