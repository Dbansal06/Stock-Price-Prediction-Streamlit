📈 Stock Price Prediction Dashboard
This is an interactive web application built with Streamlit that allows users to visualize historical stock prices, train a machine learning model (RandomForestRegressor) for price prediction, evaluate its performance, gain insights into feature importance, and even make simplified future price predictions.

✨ Features
Interactive Dashboard: User-friendly interface built entirely with Python using Streamlit.

Historical Data Fetching: Automatically downloads historical stock data using the yfinance library.

Feature Engineering: Generates essential features like Simple Moving Averages (SMA), Exponential Moving Averages (EMA), Daily Returns, and Lagged Prices.

Machine Learning Model: Utilizes a RandomForestRegressor for robust stock price prediction.

Hyperparameter Tuning: Includes an option to perform GridSearchCV to find optimal model parameters, aiming to minimize prediction errors (RMSE, MAE).

Performance Metrics: Displays Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2) to evaluate model accuracy.

Actual vs. Predicted Chart: Visualizes the model's predictions against actual historical prices.

Model Insights: A dedicated tab to show the importance of different features in the prediction model.

Future Prediction: A "Future Prediction" tab allows users to predict stock prices for a specified number of days into the future based on the trained model.

🚀 Technologies Used
Python 3.x

Streamlit: For building the interactive web application.

yfinance: For fetching historical stock market data.

Pandas: For data manipulation and analysis.

Scikit-learn: For machine learning model (RandomForestRegressor) and hyperparameter tuning (GridSearchCV).

NumPy: For numerical operations.

Matplotlib: For plotting feature importances.

⚙️ Setup and Installation
Follow these steps to get the project up and running on your local machine:

Clone the Repository (or create the files):
If you've already initialized a Git repository and pushed to GitHub, you can clone it:

git clone https://github.com/Dbansal06/Stock-Price-Prediction-Streamlit.git
cd Stock-Price-Prediction-Streamlit

If you haven't pushed to GitHub yet, ensure you have a folder containing the three project files: app.py, config.py, and data_processing.py. Navigate into that folder in your terminal.

Create a Virtual Environment (Recommended):
It's good practice to use a virtual environment to manage project dependencies.

python -m venv .venv

Activate the Virtual Environment:

On Windows:

.venv\Scripts\activate

On macOS/Linux:

source .venv/bin/activate

Install Dependencies:
Install all the required Python libraries using pip:

pip install streamlit yfinance pandas scikit-learn matplotlib

🏃 How to Run the Application
Once the setup is complete and your virtual environment is activated, you can run the Streamlit application:

streamlit run app.py

This command will open a new tab in your default web browser, typically at http://localhost:8501, displaying the Stock Price Prediction Dashboard.

📊 How to Use the Dashboard
Sidebar Parameters:

Stock Ticker: Enter the ticker symbol of the stock you want to analyze (e.g., AAPL, GOOG, MSFT).

Start Date / End Date: Select the date range for historical data.

Number of Lagged Days for Features: This determines how many previous days' closing prices are used as features for prediction.

Random Forest Estimators: Adjust the number of trees in the Random Forest model.

Prediction Dashboard Tab:

Data Acquisition & Feature Engineering: Shows the progress of data fetching and feature creation.

Data Preparation: Displays the sizes of your training and test datasets.

Model Training & Hyperparameter Tuning:

Click "Train Model (No Tuning)" to train the model with default parameters.

Click "Perform Hyperparameter Tuning" to run a Grid Search, which will find the best n_estimators and max_features for your model. This process can take some time but generally leads to better performance.

Prediction and Evaluation: Once the model is trained, it will make predictions on the test set and display RMSE, MAE, and R-squared (R2) metrics.

Actual vs. Predicted Prices: A line chart comparing the actual stock prices with your model's predictions on the test set.

Model Insights Tab:

After training a model, this tab will display a bar chart showing the Top 10 Feature Importances. This helps you understand which historical data points (e.g., recent closing prices, moving averages) the model considers most influential for its predictions.

Future Prediction Tab:

Once a model is trained, use the "Number of Future Days to Predict" slider to select how many days into the future you want to forecast.

Click "Predict Future" to see the predicted stock prices extending beyond your historical data.

Note: The future prediction is a simplified autoregressive process. For highly accurate long-term forecasts, more advanced time series forecasting models and methods for generating future features would be necessary.

⚠️ Disclaimer
Stock price prediction is an inherently challenging task. The models and predictions generated by this dashboard are for educational and demonstration purposes only. Past performance is not indicative of future results. Do not use this tool for actual investment decisions. Investing in the stock market involves significant risk, and you could lose money. Always consult with a qualified financial advisor before making any investment decisions.
