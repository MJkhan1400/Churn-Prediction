import pickle
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from .data_preprocessing import preprocess_data

def train_and_save_model(file_path, model_path, columns_path):
    """
    Train the XGBoost model and save it as a pickle file.
    - Outputs a classification report and ROC-AUC score.
    - Saves the model and column metadata.
    """
    # Preprocess data
    X, y = preprocess_data(file_path)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba)}")

    # Save model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    # Save column metadata
    columns = {'data_columns': X.columns.tolist()}
    with open(columns_path, 'w') as file:
        json.dump(columns, file)

    print(f"Model saved to {model_path}")
    print(f"Column metadata saved to {columns_path}")

