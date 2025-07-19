import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import GridSearchCV 
import matplotlib.pyplot as plt 


from config import (
    DEFAULT_TICKER, DEFAULT_START_DATE, DEFAULT_END_DATE,
    DEFAULT_PREDICTION_DAYS, DEFAULT_N_ESTIMATORS, RANDOM_STATE, TEST_SIZE_RATIO
)
from data_processing import fetch_data, create_features

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Stock Price Prediction Dashboard")
st.markdown("Predict stock prices using historical data and a Machine Learning model.")

st.sidebar.header("Prediction Parameters")

ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL, GOOG)", DEFAULT_TICKER).upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(DEFAULT_START_DATE))
end_date = st.sidebar.date_input("End Date", pd.to_datetime(DEFAULT_END_DATE))
prediction_days = st.sidebar.slider("Number of Lagged Days for Features", 5, 60, DEFAULT_PREDICTION_DAYS)
st.sidebar.write(f"Default Random Forest Estimators: {DEFAULT_N_ESTIMATORS}")


tab1, tab2, tab3 = st.tabs(["📊 Prediction Dashboard", "🧠 Model Insights", "🔮 Future Prediction"])

df = fetch_data(ticker, start_date, end_date)

if df.empty:
    st.error("Cannot proceed without data. Please adjust parameters and try again.")
    st.stop() 

df_features = create_features(df, prediction_days)

if df_features.empty:
    st.error("Not enough data to create features with the selected 'Lagged Days'. Try a longer date range or fewer lagged days.")
    st.stop() 


X = df_features.drop(['Target'], axis=1)
y = df_features['Target']


split_index = int(len(df_features) * (1 - TEST_SIZE_RATIO))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]


with tab1:
    st.header("1. Data Preparation for ML Model")
    st.write(f"Training data size: **{X_train.shape[0]} rows**")
    st.write(f"Test data size: **{X_test.shape[0]} rows**")

    st.header("2. Model Training & Hyperparameter Tuning")
    st.markdown("Train the RandomForestRegressor model. You can also perform hyperparameter tuning to find optimal parameters.")

    if st.button("Train Model (No Tuning)", key="train_no_tuning_button"):
        if X_train.empty or y_train.empty:
            st.error("Not enough data to train the model. Please ensure your date range provides sufficient data for training.")
        else:
            with st.spinner("Training RandomForestRegressor model without tuning..."):
                model = RandomForestRegressor(n_estimators=DEFAULT_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
                model.fit(X_train, y_train)
                st.session_state['model'] = model
                st.session_state['y_test'] = y_test
                st.session_state['X_test'] = X_test
                st.session_state['X_train_cols'] = X_train.columns.tolist()
                st.session_state['best_params'] = {'n_estimators': DEFAULT_N_ESTIMATORS}
            st.success("Model trained without tuning!")

    st.markdown("---")
    st.subheader("Hyperparameter Tuning (Grid Search)")
    st.info("This will search for the best `n_estimators` and `max_features` for the RandomForestRegressor. This can take some time.")
    
    if st.button("Perform Hyperparameter Tuning", key="tune_model_button"):
        if X_train.empty or y_train.empty:
            st.error("Not enough data to perform tuning. Please ensure your date range provides sufficient data for training.")
        else:
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_features': ['sqrt', 'log2', None] 
            }
            with st.spinner("Running GridSearchCV... This might take a while."):
                grid_search = GridSearchCV(
                    estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                    param_grid=param_grid,
                    cv=3, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                st.session_state['model'] = best_model
                st.session_state['y_test'] = y_test
                st.session_state['X_test'] = X_test
                st.session_state['X_train_cols'] = X_train.columns.tolist()
                st.session_state['best_params'] = grid_search.best_params_
            
            st.success(f"Hyperparameter tuning complete! Best parameters: {grid_search.best_params_}")
            st.write(f"Best RMSE from tuning: {-grid_search.best_score_:.2f}")
    if 'best_params' in st.session_state:
        st.write(f"**Current Model Parameters:** {st.session_state['best_params']}")


    st.header("3. Prediction and Evaluation")
    if 'model' in st.session_state:
        model = st.session_state['model']
        y_test = st.session_state['y_test']
        X_test = st.session_state['X_test']

        if X_test.empty or y_test.empty:
            st.warning("No test data available for prediction and evaluation. Adjust date range or lagged days.")
        else:
            with st.spinner("Making predictions on the test set..."):
                predictions = model.predict(X_test)
            st.success("Predictions made!")

            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions) 

            col1, col2, col3 = st.columns(3) 
            with col1:
                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
            with col2:
                st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
            with col3:
                st.metric("R-squared ($R^2$)", f"{r2:.2f}") 
            st.info("R-squared ($R^2$) indicates how well the predictions approximate the real data. A value of 1.0 (or 100%) means a perfect fit.")


            st.header("4. Actual vs. Predicted Prices")

            plot_df = pd.DataFrame({
                'Actual Prices': y_test,
                'Predicted Prices': predictions
            })
            plot_df.index = pd.to_datetime(plot_df.index)

            st.line_chart(plot_df)
    else:
        st.warning("Train the model first to see predictions and evaluation metrics.")

with tab2:
    st.header("Model Insights")
    st.markdown("Understand which features are most important for the prediction.")

    if 'model' in st.session_state and 'X_train_cols' in st.session_state:
        model = st.session_state['model']
        
       
        if hasattr(model, 'feature_importances_') and len(model.feature_importances_) > 0:
            feature_importances = pd.Series(model.feature_importances_, index=st.session_state['X_train_cols'])
            
            st.subheader("Feature Importances")
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_importances.nlargest(10).plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title("Top 10 Feature Importances")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Features")
            ax.invert_yaxis() 
            st.pyplot(fig) 
            
            st.write("These values indicate the relative importance of each feature in predicting the stock price.")
        else:
            st.info("Feature importances are not available for the trained model or are empty.")
    else:
        st.info("Please train the model in the 'Prediction Dashboard' tab to see model insights.")

with tab3:
    st.header("🔮 Predict Future Prices")
    st.markdown("Predict the stock price for a specified number of days into the future based on the trained model.")

    if 'model' not in st.session_state or 'X_test' not in st.session_state:
        st.warning("Please train the model in the 'Prediction Dashboard' tab first to enable future predictions.")
    else:
        model = st.session_state['model']
        
        
        if len(df_features) < prediction_days:
            st.error(f"Not enough historical data ({len(df_features)} rows) to form initial input for future prediction (requires {prediction_days} lagged days).")
        else:
            last_known_data = df_features.iloc[-prediction_days:].drop('Target', axis=1)
            
            num_future_days = st.slider("Number of Future Days to Predict", 1, 30, 5)
            
            if st.button("Predict Future", key="predict_future_button"):
                if last_known_data.empty:
                    st.error("Cannot make future predictions: Last known data is empty. Adjust date range or lagged days.")
                else:
                    future_predictions = []
                    current_input_features = last_known_data.iloc[-1].values.reshape(1, -1) # Start with the last row of features
                    
                    last_date = df_features.index[-1]
                    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, num_future_days + 1)]

                    
                    with st.spinner(f"Predicting {num_future_days} future days..."):
                        for i in range(num_future_days):
                            next_day_price = model.predict(current_input_features)[0]
                            future_predictions.append(next_day_price)

                        
                            new_features = np.roll(current_input_features[0], 1) 
                            
                           
                            
                            last_features_series = last_known_data.iloc[-1].copy()
                            
                            new_features_series = pd.Series(index=last_features_series.index)
                            
                            new_features_series['Close'] = next_day_price
                            
                            for j in range(prediction_days, 1, -1):
                                new_features_series[f'Close_Lag_{j}'] = last_features_series[f'Close_Lag_{j-1}']
                            new_features_series['Close_Lag_1'] = last_features_series['Close'] # The current 'Close' becomes 'Close_Lag_1' for the next step

                   
                            new_features_series['SMA_10'] = last_features_series['SMA_10'] 
                            new_features_series['SMA_30'] = last_features_series['SMA_30'] 
                            new_features_series['EMA_10'] = last_features_series['EMA_10'] 
                            new_features_series['EMA_30'] = last_features_series['EMA_30'] 
                            new_features_series['Daily_Return'] = (next_day_price - last_features_series['Close']) / last_features_series['Close'] # Calculate new daily return
                            
                            current_input_features = new_features_series.values.reshape(1, -1)
                            last_known_data = pd.DataFrame([new_features_series], index=[future_dates[i]]) 

                   
                    combined_plot_df = pd.DataFrame({
                        'Actual Prices': y.iloc[split_index:], 
                        'Predicted Prices': model.predict(X.iloc[split_index:]) 
                    })
                    combined_plot_df.index = pd.to_datetime(combined_plot_df.index)

                   
                    future_df = pd.DataFrame({
                        'Predicted Prices': future_predictions
                    }, index=future_dates)

                   
                    final_plot_df = pd.concat([combined_plot_df, future_df])
                    final_plot_df.index.name = 'Date'

                    st.line_chart(final_plot_df)
                    st.write("Future predicted prices:")
                    st.dataframe(future_df)

st.markdown("---")
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50; /* Green */
    color: white;
    padding: 10px 24px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    font-size: 16px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #45a049;
    transform: scale(1.05);
}
.stMetric {
    background-color: #2d3748;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #4a5568;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    text-align: center; /* Center align metrics */
}
.stMetric label {
    color: #a0aec0;
    font-size: 0.9em;
}
.stMetric .stMetricValue {
    color: #66bb6a; /* Green for RMSE */
    font-size: 2.5em;
    font-weight: bold;
}
.stMetric:nth-child(2) .stMetricValue { /* MAE */
    color: #ffeb3b; /* Yellow for MAE */
}
.stMetric:nth-child(3) .stMetricValue { /* R-squared */
    color: #8884d8; /* Purple for R-squared */
}
</style>
""", unsafe_allow_html=True)
