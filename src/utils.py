import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocesses the raw rent data for model training.

    This function performs one-hot encoding on categorical features,
    scales the numeric features, and selects final features for the model.

    Args:
        df (pd.DataFrame): The raw DataFrame.
    
    Returns:
        tuple: A tuple containing the processed features (X) and the target (y).

    """

    # --- Feature Engineering ---
    data = df.copy()
    if 'Size_per_BHK' not in data.columns:
        data['Size_per_BHK'] = data['Size'] / data['BHK']

    # --- Feature Selection ---
    features = ['BHK', 'Size', 'Bathroom', 'Size_per_BHK', 'City', 'Furnishing Status', 'Tenant Preferred']
    target = 'Rent'
    data_subset = data[features + [target]].copy()

    # --- One-Hot Encoding for Categorical Features ---
    categorical_cols = ['City', 'Furnishing Status', 'Tenant Preferred']
    data_encoded = pd.get_dummies(data_subset, columns=categorical_cols, drop_first=True)

    # --- Define Final Features (X) and Target (y) ---
    X = data_encoded.drop(target, axis=1)
    y = data_encoded[target]

    # --- Scale Numeric Features ---
    numeric_cols = ['BHK', 'Size', 'Bathroom', 'Size_per_BHK']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, scaler