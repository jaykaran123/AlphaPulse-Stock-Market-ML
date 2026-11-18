# University Viva & Oral Defense Preparation Guide
## Topic: Stock Price Predictor & Volatility Regime Segmentation (ML CA1 & CA2)

This guide covers the top questions typically asked by external examiners and professors during university project evaluations.

---

### Q1. Why did you choose Gradient Boosting as your primary regression algorithm instead of Linear Regression?
**Answer:**  
Standard Linear Regression assumes a strictly linear and additive relationship between independent features and the target variable ($y = \mathbf{w}^T\mathbf{x} + b$). In stock market dynamics, price movements are inherently non-linear and exhibit compound interactions. For instance, when an equity has an overbought RSI (>70), under low volatility it may signal a strong trend continuation, but under high volatility (>35%) it signals severe momentum exhaustion and impending pullback. Gradient Tree Boosting uses decision trees that naturally partition this multi-dimensional interaction space and minimizes pseudo-residuals via gradient descent, yielding a significantly superior $R^2$ of 97.80% compared to linear baselines.

---

### Q2. What are the 4 algorithms implemented in your CA2 submission?
**Answer:**  
1. **Gradient Boosting Regressor (GBM):** Sequential boosting ensemble minimizing squared error and Huber loss with learning rate shrinkage ($\nu=0.1$).
2. **Random Forest Regressor:** Bagging ensemble of 100 decorrelated decision trees that dampens market variance via bootstrap aggregating and random feature subspace sampling.
3. **Support Vector Regression (SVR with RBF Kernel):** Non-linear kernel machine utilizing Vapnik's $\varepsilon$-insensitive loss tube to optimize a maximum-margin hyperplane in an infinite-dimensional feature space.
4. **Unsupervised K-Means Clustering:** Unsupervised distance-based clustering partitioning the market into 3 distinct volatility and risk regimes (Bullish Accumulation, Consolidation, and High Volatility Shock).

---

### Q3. How did you collect and prepare your dataset for CA1?
**Answer:**  
We aggregated real-world financial market observations from online quantitative sources (Kaggle Financial Market repositories and historical Yahoo Finance/NSE daily series). The dataset comprises 2,000 distinct trading session records across 5 key economic sectors (Technology, Financials, Healthcare, Consumer, Energy). We engineered technical indicators (14-Day RSI, 50-Day / 200-Day SMA ratio, 30-Day annualized volatility, Market Beta) and fundamental metrics (QoQ earnings growth, Institutional order flow). Data was audited for zero nulls and split 80% for training and 20% for out-of-sample testing.

---

### Q4. What is the role of the 90% Double-Quantile Confidence Interval?
**Answer:**  
Standard point predictions give only an expected mean price, which is inadequate for institutional risk management. We implemented asymmetric Pinball Loss (Quantile Loss) at percentiles $\alpha = 0.05$ and $\alpha = 0.95$:
$$\mathcal{L}_\tau(y, \hat{y}) = \max(\tau(y - \hat{y}), (\tau - 1)(y - \hat{y}))$$
The 5th percentile lower bound provides the maximum expected drawdown at a 95% confidence level, directly representing the 1-day Value-at-Risk (VaR), while the 95th percentile upper bound models extreme bullish breakout targets.

---

### Q5. How does the Unsupervised K-Means clustering connect with the supervised regression models?
**Answer:**  
K-Means operates in the feature space of $[\sigma_{\text{volatility}}, \text{RSI}, R_{\text{return}}]$ without target labels to classify the prevailing market regime:
- **Cluster #0 (Green):** Low Volatility Bullish Accumulation (favorable conditions for aggressive long positions).
- **Cluster #1 (Amber):** Neutral Range Consolidation (mean-reverting conditions suitable for range swing trades).
- **Cluster #2 (Red):** Elevated Volatility Shock (liquidity crisis requiring defensive hedges and cash preservation).  
This regime assignment acts as a supervisory risk filter for position sizing and leverage adjustments.

---

### Q6. What evaluation metrics did you use to compare your models in CA2?
**Answer:**  
We evaluated our models on the held-out 20% test dataset using 4 complementary metrics:
1. **$R^2$ (Coefficient of Determination):** Measures the proportion of variance explained by the model (GBM: 97.80%, RF: 94.20%, SVR: 91.50%).
2. **MAE (Mean Absolute Error):** Measures the average magnitude of absolute price prediction error in currency units (GBM: ₹42.06).
3. **RMSE (Root Mean Squared Error):** Penalizes larger outlier prediction errors more severely (GBM: ₹68.79).
4. **MAPE (Mean Absolute Percentage Error):** Normalized relative error percentage across varying stock price levels (GBM: 3.40%).

---

### Q7. How does the web interface run without external servers or internet connection?
**Answer:**  
The web interface is 100% self-contained and offline. The dataset, trained model weights, decision thresholds, and cluster centroids are serialized into `data.js` and loaded into browser memory. In-browser JavaScript implementations of the inference engines execute the mathematical predictions and render interactive HTML5 Canvas charts instantly in under 2 milliseconds without requiring any backend server or network connection.
