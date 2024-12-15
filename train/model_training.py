import os
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_random_forest(X_train, y_train, model_path, scaler_path, scaler=None):
    # Train the Random Forest model
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(rf_model, model_path)
    if os.path.exists(model_path):
        print(f"Model saved successfully at: {model_path}")
    else:
        print(f"Error: Model file not created at: {model_path}")

    # Save the scaler
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
        if os.path.exists(scaler_path):
            print(f"Scaler saved successfully at: {scaler_path}")
        else:
            print(f"Error: Scaler file not created at: {scaler_path}")
    else:
        print("Warning: No scaler provided to save.")
    
    return rf_model

