import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocesses the raw rent data for model training.

    This function performs one-hot encoding on categorical features,
    scales numeric features, and extracts engineered interaction terms.

    Args:
        df (pd.DataFrame): The raw DataFrame.
    
    Returns:
        tuple: (X, y, scaler)
    """
    data = df.copy()
    if 'Size_per_BHK' not in data.columns:
        data['Size_per_BHK'] = data['Size'] / data['BHK']

    # --- Feature Selection ---
    categorical_cols = ['State', 'City', 'City_Tier', 'Locality_Zone', 'Furnishing Status', 'Tenant Preferred']
    # Filter to only existing columns
    categorical_cols = [c for c in categorical_cols if c in data.columns]

    numeric_cols = ['BHK', 'Size', 'Bathroom', 'Size_per_BHK']
    target = 'Rent'

    data_subset = data[numeric_cols + categorical_cols + [target]].copy()

    # --- One-Hot Encoding for Categorical Features ---
    data_encoded = pd.get_dummies(data_subset, columns=categorical_cols, drop_first=True)

    # --- Define Final Features (X) and Target (y) ---
    X = data_encoded.drop(target, axis=1)
    y = data_encoded[target]

    # --- Scale Numeric Features ---
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, scaler