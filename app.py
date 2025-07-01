import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objs as go

# Set page configuration
st.set_page_config(page_title="Bitcoin Price Predictor",
                   page_icon="₿",
                   layout="wide",
                   initial_sidebar_state="expanded")

# --- Custom CSS for a professional look ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #ff4b4b;
        border-radius:10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    h1, h2, h3 {
        color: #1e1e1e;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        text-align: center;
    }
    .metric-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .metric-title {
        font-size: 18px;
        color: #555555;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)


# --- App Title and Description ---
st.title("Bitcoin Price Prediction App ₿")
st.markdown("This app leverages the power of **Facebook Prophet** to forecast the future price of Bitcoin (BTC-USD).")
st.markdown("---")

# --- Data Loading ---
@st.cache_data
def load_data(ticker):
    """
    Loads historical data for the given ticker.
    Adjusted start date to ensure sufficient data is downloaded.
    """
    # Changed start date from "2014-01-01" to "2017-01-01" as reliable data
    # might not be available that far back on Yahoo Finance, which was causing the error.
    # The previous log showed YFPricesMissingError for 2014 data. 
    data = yf.download(ticker, start="2017-01-01")
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
try:
    data = load_data('BTC-USD')
    # Check if data is empty after download
    if data.empty:
        st.error("Could not load Bitcoin data. Please check the ticker symbol or the availability of data for the specified date range.")
        st.stop() # Stop the app if no data is loaded
    data_load_state.text('Loading data... done!')
except Exception as e:
    st.error(f"Error loading data: {e}. Please check your internet connection or the data source.")
    st.stop()


# --- Sidebar ---
st.sidebar.header("Prediction Parameters")
n_years = st.sidebar.slider('Years of prediction:', 1, 4, 1)
period = n_years * 365

# --- Main App Logic ---
if st.sidebar.button('Predict'):

    st.header("1. Raw Bitcoin Price Data")
    st.write("Below is the historical price data for Bitcoin (BTC-USD).")
    st.write(data.tail())

    # --- Plotting Raw Data ---
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    plot_raw_data()

    # --- Forecasting with Prophet ---
    st.header("2. Bitcoin Price Forecast")
    st.write(f"The model will now predict the Bitcoin price for the next {n_years} year(s).")
    
    # Prepare data for Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    
    # Ensure no NaN values in 'ds' or 'y' after renaming
    df_train.dropna(subset=['ds', 'y'], inplace=True)

    # Validate that there are enough rows after dropping NaNs
    if df_train.shape[0] < 2:
        st.error("Insufficient valid data points to train the Prophet model. Please adjust the data loading parameters or check the data source.")
        st.stop() # Stop the app if there's not enough data for Prophet 


    # Initialize and train the model
    m = Prophet(
        changepoint_prior_scale=0.15,
        holidays_prior_scale=0.01,
        seasonality_prior_scale=10.0,
        seasonality_mode='multiplicative'
    )
    # The error "Dataframe has less than 2 non-NaN rows" occurs here if df_train is not sufficient 
    m.fit(df_train)

    # Create future dataframe and make predictions
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Display forecast data
    st.subheader('Forecast Data')
    st.write(forecast.tail())

    # Plot forecast
    st.subheader('Forecast Visualization')
    fig1 = plot_plotly(m, forecast)
    fig1.update_layout(
        title='Bitcoin Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Plot forecast components
    st.subheader("Forecast Components")
    st.write("These charts show the trend and seasonality components of the forecast.")
    fig2 = plot_components_plotly(m, forecast)
    st.plotly_chart(fig2, use_container_width=True)

    # --- Model Evaluation ---
    st.header("3. Model Performance")
    
    # Get predictions for the training data period
    forecast_train = m.predict(df_train)
    
    # Calculate metrics
    r2 = r2_score(df_train['y'], forecast_train['yhat'])
    mae = mean_absolute_error(df_train['y'], forecast_train['yhat'])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">R-Squared (R²)</div>
            <div class="metric-value">{r2:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Absolute Error (MAE)</div>
            <div class="metric-value">${mae:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.info(f"""
    **Interpretation:**
    - **R-squared ({r2:.4f})**: This indicates that approximately **{r2*100:.2f}%** of the variance in Bitcoin's price is explained by the model. A higher value is better.
    - **MAE (${mae:,.2f})**: On average, the model's predictions are off by about **${mae:,.2f}**. A lower value is better.
    """)

else:
    st.info('Click the "Predict" button in the sidebar to start the forecast.')

st.markdown("---")
st.markdown("Created by a world-class investor and app developer.")
