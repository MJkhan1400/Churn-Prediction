from sklearn.utils import resample
import pandas as pd

def balance_classes(X_train, y_train):
    train_data = pd.concat([X_train, y_train], axis=1)
    majority = train_data[train_data['Churn'] == 0]
    minority = train_data[train_data['Churn'] == 1]
    
    minority_oversampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42
    )
    balanced_train_data = pd.concat([majority, minority_oversampled])
    
    X_train = balanced_train_data.drop('Churn', axis=1)
    y_train = balanced_train_data['Churn']
    
    return X_train, y_train

