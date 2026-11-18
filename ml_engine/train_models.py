"""
========================================================================================
Coursework: Machine Learning (ML) CA1 & CA2 Comprehensive Project
Topic: Real-World Stock Price Predictor & Volatility Regime Segmentation System
Author: Solo Project
Submission: CA1 (Data Collection, EDA & Engineering) & CA2 (Model Training & Evaluation)
========================================================================================
Algorithms Implemented (Min. 3 required, non-trivial algorithms used):
  1. Gradient Boosting Regressor (GBM with Huber & Squared Error Loss)
  2. Random Forest Regressor (Ensemble Bagging with 100 Estimators)
  3. Support Vector Regression (SVR with Non-Linear Radial Basis Function Kernel)
  4. Unsupervised K-Means Clustering (Market Volatility & Risk Regime Segmentation)
========================================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "8")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)

def main():
    print_section("CA1: Data Ingestion & Exploratory Data Analysis (EDA)")
    
    # 1. Load Dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "..", "dataset", "stock_market_dataset.csv")
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset file not found at: {dataset_path}")
        sys.exit(1)
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset successfully loaded from: {dataset_path}")
    print(f"Shape: {df.shape[0]} rows (records), {df.shape[1]} columns (features)")
    print("\nFirst 5 Records:")
    print(df[['symbol', 'sector', 'close_price', 'rsi_14', 'volatility_30d', 'target_next_price']].head())
    
    # Check for missing / null values
    null_counts = df.isnull().sum()
    print(f"\nMissing values check: {null_counts.sum()} total missing values found.")
    
    # Summary Statistics
    print("\nDescriptive Statistics of Key Technical Features:")
    print(df[['close_price', 'rsi_14', 'sma_ratio', 'volatility_30d', 'beta', 'target_next_price']].describe().T)
    
    print_section("CA1: Feature Selection & Preprocessing Pipeline")
    
    feature_cols = [
        'close_price',
        'rsi_14',
        'sma_ratio',
        'volatility_30d',
        'beta',
        'earnings_growth_pct',
        'institutional_flow_idx'
    ]
    target_col = 'target_next_price'
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 80/20 Train/Test Split (Stratified random seed for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train Set size: {X_train.shape[0]} samples (80%)")
    print(f"Test Set size:  {X_test.shape[0]} samples (20%)")
    
    # Feature Standardization (Crucial for SVR & Distance-based K-Means)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("StandardScaler fitted on training distribution and applied to test set.")
    
    print_section("CA2: Model Training - Algorithm 1: Gradient Boosting Regressor (GBM)")
    print("Architecture: 100 Boosting Stages, Learning Rate = 0.1, Max Depth = 4, Loss = 'squared_error'")
    gbm_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        loss='squared_error'
    )
    gbm_model.fit(X_train, y_train)
    y_pred_gbm = gbm_model.predict(X_test)
    
    r2_gbm = r2_score(y_test, y_pred_gbm)
    mae_gbm = mean_absolute_error(y_test, y_pred_gbm)
    rmse_gbm = np.sqrt(mean_squared_error(y_test, y_pred_gbm))
    mape_gbm = mean_absolute_percentage_error(y_test, y_pred_gbm) * 100
    
    print(f"-> GBM R² Score: {r2_gbm * 100:.2f}%")
    print(f"-> GBM MAE:      ₹{mae_gbm:.2f}")
    print(f"-> GBM RMSE:     ₹{rmse_gbm:.2f}")
    print(f"-> GBM MAPE:     {mape_gbm:.2f}%")
    
    print_section("CA2: Model Training - Algorithm 2: Random Forest Regressor (Ensemble Bagging)")
    print("Architecture: 100 Decision Trees, Bootstrap Sampling, Max Features = 'sqrt'")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        max_features='sqrt',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf) * 100
    
    print(f"-> RF R² Score: {r2_rf * 100:.2f}%")
    print(f"-> RF MAE:      ₹{mae_rf:.2f}")
    print(f"-> RF RMSE:     ₹{rmse_rf:.2f}")
    print(f"-> RF MAPE:     {mape_rf:.2f}%")
    
    print_section("CA2: Model Training - Algorithm 3: Support Vector Regression (SVR - RBF Kernel)")
    print("Architecture: Non-linear Radial Basis Function (RBF) Kernel, C = 100.0, Epsilon = 0.1")
    svr_model = SVR(
        kernel='rbf',
        C=100.0,
        epsilon=0.1
    )
    svr_model.fit(X_train_scaled, y_train)
    y_pred_svr = svr_model.predict(X_test_scaled)
    
    r2_svr = r2_score(y_test, y_pred_svr)
    mae_svr = mean_absolute_error(y_test, y_pred_svr)
    rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
    mape_svr = mean_absolute_percentage_error(y_test, y_pred_svr) * 100
    
    print(f"-> SVR R² Score: {r2_svr * 100:.2f}%")
    print(f"-> SVR MAE:      ₹{mae_svr:.2f}")
    print(f"-> SVR RMSE:     ₹{rmse_svr:.2f}")
    print(f"-> SVR MAPE:     {mape_svr:.2f}%")
    
    print_section("CA2: Model Training - Algorithm 4: Unsupervised K-Means Volatility Segmentation")
    print("Architecture: K=3 Clusters (Bullish Accumulation, Consolidation, High Volatility Shock)")
    cluster_features = df[['volatility_30d', 'rsi_14', 'target_return_pct']].values
    cluster_scaler = StandardScaler()
    cluster_features_scaled = cluster_scaler.fit_transform(cluster_features)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_assigned'] = kmeans.fit_predict(cluster_features_scaled)
    inertia = kmeans.inertia_
    
    print(f"-> Within-Cluster Inertia (Sum of Squared Distances): {inertia:.2f}")
    print(f"-> Total Samples Segmented: {len(df)}")
    print("\nCluster Centroids (Standardized Feature Space):")
    for idx, center in enumerate(kmeans.cluster_centers_):
        print(f"   Cluster #{idx}: Volatility={center[0]:.2f}, RSI={center[1]:.2f}, Return={center[2]:.2f}")
        
    print_section("CA2: Comparative Model Performance Summary")
    comparison_df = pd.DataFrame({
        'Algorithm': [
            'Gradient Boosting Regressor (Primary)',
            'Random Forest Regressor (Ensemble)',
            'Support Vector Regressor (SVR RBF)',
            'K-Means Clustering (Unsupervised)'
        ],
        'Type': ['Supervised Boosting', 'Supervised Bagging', 'Supervised Kernel ML', 'Unsupervised Clustering'],
        'R² Score / Inertia': [f"{r2_gbm*100:.2f}%", f"{r2_rf*100:.2f}%", f"{r2_svr*100:.2f}%", f"Inertia: {inertia:.1f}"],
        'MAE': [f"₹{mae_gbm:.2f}", f"₹{mae_rf:.2f}", f"₹{mae_svr:.2f}", "N/A (Clustering)"],
        'RMSE': [f"₹{rmse_gbm:.2f}", f"₹{rmse_rf:.2f}", f"₹{rmse_svr:.2f}", "N/A (Clustering)"],
        'MAPE': [f"{mape_gbm:.2f}%", f"{mape_rf:.2f}%", f"{mape_svr:.2f}%", "N/A (Clustering)"]
    })
    print(comparison_df.to_string(index=False))
    
    print("\nFeature Importances (Gradient Boosting Model):")
    for feat, imp in zip(feature_cols, gbm_model.feature_importances_):
        print(f"  {feat:<25}: {imp * 100:.2f}%")
        
    print_section("Project Pipeline Execution Completed Successfully")
    print("All models trained and ready. Run web_interface/index.html to launch interactive GUI.")

if __name__ == '__main__':
    main()
