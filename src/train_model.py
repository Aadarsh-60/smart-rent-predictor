import sys
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure src can import utils
src_path = os.path.abspath(os.path.dirname(__file__))
if src_path not in sys.path:
    sys.path.append(src_path)

from utils import preprocess_data

def train_and_evaluate():
    # Base directory resolution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'raw', 'cleaned_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    df_cleaned = pd.read_csv(data_path)

    # Preprocess the data
    X, y, scaler = preprocess_data(df_cleaned)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Candidate Models (Linear, Regularized Linear, Single Tree, Bagged Ensemble)
    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=8),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    comparison_results = {}
    fitted_models = {}

    print("\n" + "=" * 60)
    print("MODEL BENCHMARKING & EVALUATION")
    print("=" * 60)
    print(f"{'Model':<20} | {'MAE (INR)':<12} | {'RMSE (INR)':<12} | {'R² Score':<10} | {'Train Time':<10}")
    print("-" * 75)

    for name, model in candidate_models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        fitted_models[name] = model

        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        comparison_results[name] = {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4),
            'TrainTime_s': round(train_time, 4)
        }

        print(f"{name:<20} | ₹{mae:>10,.2f} | ₹{rmse:>10,.2f} | {r2:>10.4f} | {train_time:>8.3f}s")

    # Champion model is Random Forest
    champion_model = fitted_models['Random Forest']
    champion_metrics = comparison_results['Random Forest']
    print("=" * 60)
    print(f"Champion Model: Random Forest (R² = {champion_metrics['R2']:.4f}, MAE = ₹{champion_metrics['MAE']:,.2f})")

    # Feature Importance for Random Forest
    importances = champion_model.feature_importances_
    feat_imp = {col: round(float(imp), 4) for col, imp in sorted(
        zip(X.columns, importances), key=lambda x: x[1], reverse=True
    )}

    # Sample actual vs predicted for evaluation plot
    y_test_pred = champion_model.predict(X_test)
    test_sample_df = pd.DataFrame({
        'Actual': y_test.values[:300],
        'Predicted': np.round(y_test_pred[:300], 2)
    })

    # Save artifacts
    print("\nSaving artifacts to models/ directory...")
    joblib.dump(champion_model, os.path.join(models_dir, 'random_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(models_dir, 'training_columns.pkl'))

    with open(os.path.join(models_dir, 'model_comparison.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2)

    with open(os.path.join(models_dir, 'feature_importance.json'), 'w') as f:
        json.dump(feat_imp, f, indent=2)

    test_sample_df.to_json(os.path.join(models_dir, 'test_predictions.json'), orient='records')

    print("All models and metadata successfully saved!")

if __name__ == '__main__':
    train_and_evaluate()