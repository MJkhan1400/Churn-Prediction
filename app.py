import os
import pickle
import pandas as pd
from src.data_preprocessing import preprocess_data
from src.train_model import train_and_save_model
from src.evaluate_model import evaluate_model

def main_menu():
    print("\nWelcome to the Churn Prediction Project")
    print("=======================================")
    print("1. Train Model")
    print("2. Evaluate Model")
    print("3. Predict Churn")
    print("4. Exit")

    choice = input("Enter your choice (1-4): ")
    if choice == '1':
        train_model_ui()
    elif choice == '2':
        evaluate_model_ui()
    elif choice == '3':
        predict_churn_ui()
    elif choice == '4':
        print("Goodbye!")
        exit()
    else:
        print("Invalid choice. Please try again.")
        main_menu()

def train_model_ui():
    print("\n==== Train Model ====")
    data_path = input("Enter the path to the raw dataset (e.g., data/raw/churn_data.csv): ")
    model_path = 'model/churn_model.pkl'
    columns_path = 'model/columns.json'

    try:
        train_and_save_model(data_path, model_path, columns_path)
        print(f"Model trained and saved to {model_path}.")
        print(f"Column metadata saved to {columns_path}.")
    except Exception as e:
        print(f"Error during training: {e}")
    
    main_menu()

def evaluate_model_ui():
    print("\n==== Evaluate Model ====")
    model_path = 'model/churn_model.pkl'
    data_path = input("Enter the path to the processed dataset (e.g., data/processed/churn_data_processed.csv): ")

    try:
        # Load processed data
        df = pd.read_csv(data_path)
        X = df.drop(columns=['Churn'])
        y = df['Churn']

        evaluate_model(model_path, X, y)
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    main_menu()

def predict_churn_ui():
    print("\n==== Predict Churn ====")
    model_path = 'model/churn_model.pkl'
    columns_path = 'model/columns.json'

    # Load model and columns
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(columns_path, 'r') as file:
            columns = pd.read_json(file)['data_columns'].tolist()
    except Exception as e:
        print(f"Error loading model or columns: {e}")
        main_menu()

    # Input data for prediction
    print("\nEnter customer details for churn prediction:")
    user_input = {}
    for column in columns:
        value = input(f"{column}: ")
        user_input[column] = float(value) if value.replace('.', '', 1).isdigit() else value

    # Convert input to DataFrame and preprocess
    input_df = pd.DataFrame([user_input])
    try:
        X_processed, _ = preprocess_data(input_df)  # Process single input
        prediction = model.predict(X_processed)
        print("\nPrediction: ", "Customer will churn" if prediction[0] == 1 else "Customer will not churn")
    except Exception as e:
        print(f"Error during prediction: {e}")
    
    main_menu()

if __name__ == '__main__':
    main_menu()

