import yfinance as yf
import pandas as pd
import streamlit as st 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='yfinance')

@st.cache_data 
def fetch_data(ticker_symbol, start, end):
    """
    Fetches historical stock data for a given ticker and date range.
    Uses Streamlit's caching to prevent repeated downloads.

    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL').
        start (pd.Timestamp or str): Start date.
        end (pd.Timestamp or str): End date.

    Returns:
        pd.DataFrame: DataFrame containing historical 'Close' prices.
                      Returns an empty DataFrame if data fetching fails.
    """
    st.info(f"Fetching historical data for {ticker_symbol} from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}...")
    try:
        stock_data = yf.download(ticker_symbol, start=start, end=end)
        if stock_data.empty:
            st.error("No data downloaded. Check ticker symbol or date range.")
            return pd.DataFrame()
        st.success("Data fetched successfully!")
        return stock_data[['Close']].copy()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.warning("Please ensure you have an active internet connection and the ticker symbol is valid.")
        return pd.DataFrame()

def create_features(data_frame, lag_days):
    """
    Generates features for stock price prediction from a DataFrame of 'Close' prices.

    Args:
        data_frame (pd.DataFrame): DataFrame with at least a 'Close' column.
        lag_days (int): Number of past days to use for lagged features.

    Returns:
        pd.DataFrame: DataFrame with engineered features and the 'Target' variable.
    """
    st.info("Generating features...")
    df_processed = data_frame.copy()

    
    df_processed['SMA_10'] = df_processed['Close'].rolling(window=10).mean()
    df_processed['SMA_30'] = df_processed['Close'].rolling(window=30).mean()

  
    df_processed['EMA_10'] = df_processed['Close'].ewm(span=10, adjust=False).mean()
    df_processed['EMA_30'] = df_processed['Close'].ewm(span=30, adjust=False).mean()

   
    df_processed['Daily_Return'] = df_processed['Close'].pct_change()

    
    for i in range(1, lag_days + 1):
        df_processed[f'Close_Lag_{i}'] = df_processed['Close'].shift(i)

   
    df_processed['Target'] = df_processed['Close'].shift(-1)

    
    initial_rows = df_processed.shape[0]
    df_processed.dropna(inplace=True)
    rows_dropped = initial_rows - df_processed.shape[0]

    if rows_dropped > 0:
        st.warning(f"Dropped {rows_dropped} rows with NaN values after feature engineering.")
    st.success(f"Features generated. Data shape after dropping NaNs: {df_processed.shape}")

    return df_processed
