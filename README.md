# 📈 Stock Price Prediction Dashboard

An interactive **Stock Price Prediction Dashboard** built using **Streamlit** that enables users to analyze historical stock data, engineer meaningful features, train a machine learning model, evaluate its performance, visualize insights, and generate simplified future price predictions.

This project demonstrates practical application of **data preprocessing, feature engineering, machine learning, and model evaluation** in a real-world financial use case.

---

## 🔍 Project Overview

The dashboard allows users to:
- Fetch real-time historical stock data
- Perform feature engineering on time-series data
- Train and evaluate a **Random Forest Regression** model
- Visualize predictions and model insights
- Generate short-term future price forecasts

The application is fully interactive and designed with usability and clarity in mind.

---

## ✨ Key Features

- **Interactive Web Dashboard**  
  Built using Streamlit for a clean, user-friendly experience.

- **Automated Data Fetching**  
  Retrieves historical stock price data using the `yfinance` API.

- **Feature Engineering**  
  Includes:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Daily Returns
  - Lagged Closing Prices

- **Machine Learning Model**  
  Uses `RandomForestRegressor` for robust non-linear price prediction.

- **Hyperparameter Tuning**  
  Optional **GridSearchCV** to optimize model parameters and reduce error.

- **Performance Evaluation**  
  Displays:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score

- **Visualization**  
  - Actual vs Predicted price comparison
  - Feature importance analysis

- **Future Price Prediction**  
  Predicts stock prices for a user-defined number of future days using an autoregressive approach.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit** – Web application framework
- **yfinance** – Stock market data retrieval
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Scikit-learn** – Machine learning & hyperparameter tuning
- **Matplotlib** – Feature importance visualization

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Dbansal06/Stock-Price-Prediction-Streamlit.git
cd Stock-Price-Prediction-Streamlit
```

### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv .venv
```

### 3️⃣ Activate the Virtual Environment
Windows
```bash
.venv\Scripts\activate
```

macOS / Linux
```bash
source .venv/bin/activate
```

4️⃣ Install Dependencies
```bash
pip install streamlit yfinance pandas scikit-learn matplotlib
```

▶️ Running the Application
```bash
streamlit run app.py
```
The application will launch in your browser at:
```arduino
http://localhost:8501
```

📊 **Dashboard Usage Guide**
🔧 Sidebar Inputs

Stock Ticker (e.g., AAPL, MSFT, GOOG)
Date Range for historical data
Number of Lag Days for feature creation
Random Forest Estimators (model complexity)

📈 **Prediction Dashboard Tab**

- Data fetching & preprocessing status
- Train/Test dataset information
- Model training options:
- Train without tuning
- Train with GridSearchCV
- Evaluation metrics (RMSE, MAE, R²)
- Actual vs Predicted price visualization

🔎 **Model Insights Tab**

- Displays Top 10 Feature Importances
-Helps interpret which features influence predictions most

🔮 **Future Prediction Tab**

-Select number of future days
-Generates predicted prices beyond historical data

Note: Future predictions are based on a simplified autoregressive approach and are intended for short-term analysis.

⚠️ **Disclaimer**

Stock price prediction is inherently uncertain.
This project is developed strictly for educational and demonstration purposes.
Past performance does not guarantee future results.
Do not use this tool for real financial or investment decisions.
Always consult a qualified financial advisor before investing.

👤 **Author**
 Deenu
📌 Aspiring Data Scientist | Machine Learning Enthusiast
🔗 GitHub: https://github.com/Dbansal06
