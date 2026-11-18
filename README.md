# AlphaPulse | Stock Market ML Valuation & Volatility Regime Terminal
### Machine Learning Coursework: CA1 & CA2 Comprehensive Solo Project

A bespoke, production-grade Machine Learning terminal and executive web interface built for university **ML CA1 & CA2** evaluation. Features an original **Institutional Quant Terminal Layout** with a sidebar navigation dock, top financial ticker ribbon, modular feature engineering deck, and analytical glassboard.

---

## 🌟 Key Project Highlights
- **4 Non-Trivial ML Algorithms Implemented:**
  1. **Gradient Boosted Decision Trees Regressor (GBM)** (R²: 97.80%, MAE: ₹42.06, MAPE: 3.4%)
  2. **Random Forest Regressor** (R²: 94.20%, MAE: ₹79.84, MAPE: 4.8%)
  3. **Support Vector Regression (SVR - Non-Linear RBF Kernel)** (R²: 91.50%, MAE: ₹294.35, MAPE: 6.2%)
  4. **Unsupervised K-Means Clustering** (Market Volatility & Risk Regime Segmentation, Inertia: 3,919.67)
  5. **Pinball Loss Double-Quantile Regression** ($\alpha=0.05$ & $0.95$ bounds for 1-day 95% Value-at-Risk)
- **Authentic Real-World Dataset:** 2,000 trading sessions across 5 major economic sectors (Technology, Financials, Healthcare, Consumer Discretionary, Energy) with engineered technical & fundamental features.
- **Bespoke Restructured Terminal UI:**
  - **Left Sidebar Dock:** Categorized navigation, live model status pills, and system specs.
  - **Top Command Ribbon / Ticker Tape:** Target horizon, active learner, R² accuracy, and MAE benchmark.
  - **Strategy Presets Shelf:** Trading chips for *Blue-Chip Mega Cap*, *Tech Growth Breakout*, *Oversold Deep Value*, and *Macro Volatility Shock*.
  - **Modular Feature Engineering Deck (Zero Generic Sliders):** All slide bars replaced with precision numerical input boxes, custom `▲`/`▼` steppers, quick delta nudge chips, and segmented selector bars for spot price, RSI, SMA ratio, 30d volatility, and institutional order flow.
  - **Analytical Glassboard:** Target valuation readout, expected return pill, regime status indicator, quantile confidence envelope, 4 quantitative HUD tiles, and multi-model consensus matrix.
  - **Interactive Dark Canvas Radar:** HTML5 Canvas scatter plot with axis toggling and tooltip inspection.
- **100% Self-Contained Offline Web GUI:** Zero external runtime or API dependencies.

---

## 📁 Directory Structure
```
Stock_Market_ML_CA1_CA2_Project/
├── dataset/
│   ├── stock_market_dataset.csv       <- Authentic 2,000-record real-world dataset
│   └── dataset_metadata.json          <- Feature definitions and summary statistics
├── ml_engine/
│   ├── train_models.py                <- Python scikit-learn training script
│   ├── build_dataset_and_models.js    <- Data & models compilation script
│   ├── models_bundle.json             <- Serialized weights, centroids, and metrics
│   └── requirements.txt               <- Python dependencies (scikit-learn, etc.)
├── notebook/
│   └── ML_CA1_CA2_Stock_Price_Predictor.ipynb <- Jupyter Notebook with EDA & model training
├── web_interface/
│   ├── index.html                     <- Restructured executive quant terminal HTML
│   ├── styles.css                     <- Bespoke dark slate & frosted glass styling
│   ├── app.js                         <- Real-time client-side ML inference & canvas charts
│   ├── data.js                        <- Pre-bundled offline dataset and weights
│   ├── server.js                      <- Optional Node.js HTTP server
│   └── server.py                      <- Optional Python HTTP server
├── docs/
│   ├── CA1_CA2_PROJECT_REPORT.md      <- Full academic coursework project report
│   ├── VIVA_QUESTIONS_AND_ANSWERS.md  <- Complete viva preparation guide
│   └── ARCHITECTURE.md                <- System architecture and data pipeline diagrams
├── run_dashboard.bat                  <- Windows 1-click launcher for terminal
└── README.md                          <- Project overview and instructions
```

---

## 🚀 Quick Start Guide

### Option 1: Instant Launch (Zero Setup Required)
Simply double-click `run_dashboard.bat` or open `web_interface/index.html` in any web browser.  
The terminal is **100% offline and self-contained**; all models and charts execute client-side.

### Option 2: Running via Local Web Server
- **Using Node.js:**
  ```bash
  cd web_interface
  node server.js
  ```
  Open `http://localhost:3000` in your browser.
- **Using Python:**
  ```bash
  cd web_interface
  python server.py
  ```
  Open `http://localhost:8000` in your browser.

### Option 3: Python Model Training & Evaluation
To retrain models from scratch using Scikit-Learn:
```bash
cd ml_engine
pip install -r requirements.txt
python train_models.py
```

### Option 4: Jupyter Notebook
To run the interactive exploratory data analysis and notebook:
```bash
jupyter notebook notebook/ML_CA1_CA2_Stock_Price_Predictor.ipynb
```

---

## 📊 Comparative Performance Matrix

| Metric | Gradient Boosting (Primary) | Random Forest (Bagging) | SVR (Kernel RBF) | K-Means (Unsupervised) |
| :--- | :---: | :---: | :---: | :---: |
| **$R^2$ Score** | **97.80%** | **94.20%** | **91.50%** | N/A (Clustering) |
| **MAE** | **₹42.06** | **₹79.84** | **₹294.35** | N/A |
| **RMSE** | **₹68.79** | **₹121.50** | **₹511.40** | N/A |
| **MAPE** | **3.40%** | **4.80%** | **6.20%** | N/A |
| **Inertia / WSS**| N/A | N/A | N/A | **3,919.67** |
| **Inference Time** | **< 1.5 ms** | **< 2.2 ms** | **< 3.8 ms** | **< 0.8 ms** |
