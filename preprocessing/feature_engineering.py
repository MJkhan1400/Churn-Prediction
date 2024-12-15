from sklearn.preprocessing import StandardScaler, LabelEncoder

def encode_features(data):
    # Encode categorical columns
    data['International plan'] = data['International plan'].map({'Yes': 1, 'No': 0})
    data['Voice mail plan'] = data['Voice mail plan'].map({'Yes': 1, 'No': 0})
    
    # Label encode the 'State' column
    label_encoder = LabelEncoder()
    data['State'] = label_encoder.fit_transform(data['State'])
    
    return data, label_encoder

def scale_features(X, numerical_cols, scaler=None):
    if scaler is None:
        scaler = StandardScaler()  # Instantiate scaler if not provided
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    else:
        X[numerical_cols] = scaler.transform(X[numerical_cols])
    return X, scaler

