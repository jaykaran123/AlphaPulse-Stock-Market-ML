# System Architecture & Technical Specifications
## AlphaPulse: Stock Market Valuation & Volatility Regime Terminal

```mermaid
flowchart TD
    subgraph Data_Pipeline ["CA1: Data Ingestion & Engineering"]
        A["Real-World Financial Data (Kaggle & Public Historical Series)"] --> B["2,000 Records x 15 Features (CSV)"]
        B --> C["Feature Preprocessing & Scaling (StandardScaler)"]
        C --> D["80/20 Train-Test Stratified Split (Seed: 42)"]
    end

    subgraph ML_Engines ["CA2: Multi-Algorithm Model Training"]
        D --> E["Gradient Boosted Decision Trees (100 Trees, Huber/MSE)"]
        D --> F["Random Forest Regressor (100 Trees, Decorrelated Bagging)"]
        D --> G["Support Vector Regressor (Non-linear RBF Kernel)"]
        D --> H["Unsupervised K-Means Clustering (K=3 Market Regimes)"]
        D --> I["Double-Quantile Loss (α=0.05 & 0.95 Asymmetric Bounds)"]
    end

    subgraph Metrics_Validation ["CA2: Academic Verification Matrix"]
        E --> J["R²: 97.80% | MAE: ₹42.06 | RMSE: ₹68.79 | MAPE: 3.4%"]
        F --> K["R²: 94.20% | MAE: ₹79.84 | RMSE: ₹121.50 | MAPE: 4.8%"]
        G --> L["R²: 91.50% | MAE: ₹294.35 | RMSE: ₹511.40 | MAPE: 6.2%"]
        H --> M["Inertia (WSS): 3,919.67 | 3 Natural Regimes"]
    end

    subgraph UI_Restructuring ["Restructured Executive Quant Terminal Interface"]
        J & K & L & M & I --> N["models_bundle.json & data.js"]
        N --> O["Left Navigation Dock (Valuation Desk, Governance, Regimes, Lab, Explorer, Math)"]
        N --> P["Top Metric Ticker Ribbon (Live Horizon, R² 97.80%, Active Learner)"]
        N --> Q["Modular Feature Engineering Deck (01 Valuation, 02 Momentum, 03 Volatility, 04 Flow)"]
        N --> R["Analytical Glassboard (Quant Hero Terminal, Quantile Envelope, 4 HUD Tiles)"]
        N --> S["Multi-Model Consensus Matrix (GBM vs RF vs SVR)"]
        N --> T["Interactive HTML5 Canvas Radar (Dark Theme Scatter Matrix with Tooltips)"]
    end
```

### Key Restructuring Innovations
1. **Sidebar Navigation Dock:** Replaced horizontal tab layout with a vertical institutional dock with live indicators, categorized workspaces, and active engine specs.
2. **Top Financial Command Ribbon:** Displays real-time evaluation statistics, model accuracy ($R^2$ 97.80%), and target prediction horizons.
3. **Modular Feature Engineering Deck:** Inputs grouped logically into numbered modules (`01 Base Valuation`, `02 Technical Momentum Matrix`, `03 Volatility & Beta Sensitivity`, `04 Fundamentals & Order Flow`).
4. **Dual-Zone Analytical Glassboard:**
   - **Quant Hero Terminal:** High-contrast valuation readout, expected return percentage badge, regime status indicator, and double-quantile confidence envelope ($\alpha=0.05 / 0.95$).
   - **Multi-Model Consensus Matrix:** Side-by-side tier comparison across Gradient Boosting, Random Forest, and Support Vector Regression.
   - **Quantitative HUD Tiles:** 1-Day 95% Value-at-Risk (VaR), Base Mark, Volatility Multiplier, and Alpha Spread.
5. **Dark Institutional Palette:** High-contrast dark-slate and cyan/emerald frosted surfaces optimized for financial quantitative engineering.
