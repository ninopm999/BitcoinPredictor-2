import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objs as go
import datetime
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Advanced Bitcoin Price Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        padding-top: 2rem;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #ff4b4b;
        border-radius:10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        width: 100%;
        margin-top: 20px;
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
        margin-bottom: 20px;
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

# App title and description
st.title("Bitcoin Price Prediction App ₿")
st.markdown("This app leverages the power of **Facebook Prophet** to forecast the future price of Bitcoin (BTC-USD).")
st.markdown("---")

# Data loading function
@st.cache_data(ttl=3600)
def load_data(ticker, start_date_str):
    try:
        data = yf.download(ticker, start=start_date_str)
        if data.empty:
            st.warning(f"No data downloaded for {ticker} from {start_date_str}. Please check the ticker and date range.")
            return pd.DataFrame()
        
        data.reset_index(inplace=True)
        date_col_name = 'Date' if 'Date' in data.columns else 'index'
        if date_col_name != 'Date':
            data.rename(columns={date_col_name: 'Date'}, inplace=True)
            
        if 'Close' not in data.columns:
            st.error(f"Missing 'Close' price column for {ticker}. Cannot proceed with prediction.")
            return pd.DataFrame()

        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data.dropna(subset=['Date', 'Close'], inplace=True)

        if data.empty:
            st.warning(f"No valid data points remaining for {ticker} after cleaning.")
            return pd.DataFrame()

        return data[['Date', 'Close']].copy()
    except Exception as e:
        st.error(f"An error occurred while loading data for {ticker}: {e}")
        return pd.DataFrame()

# Sidebar inputs
st.sidebar.header("Data & Prediction Settings")
selected_ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., BTC-USD)", "BTC-USD").upper()
default_start_date = datetime.date(2017, 1, 1)
start_date = st.sidebar.date_input("Select Start Date for Data", default_start_date)
n_years = st.sidebar.slider('Years to predict:', 1, 4, 1)
prediction_freq = st.sidebar.selectbox('Prediction Frequency:', ('Daily', 'Weekly', 'Monthly'), index=0)
freq_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
prophet_freq = freq_map[prediction_freq]

st.sidebar.header("Prophet Model Parameters")
cp_scale = st.sidebar.slider('Changepoint Prior Scale (flexibility):', 0.01, 0.5, 0.15, 0.01)
season_scale = st.sidebar.slider('Seasonality Prior Scale (strength):', 0.1, 20.0, 10.0, 0.1)

# Prediction button and logic
if st.sidebar.button('Run Prediction'):
    data_load_status = st.empty()
    data_load_status.text('Loading data...')
    data = load_data(selected_ticker, start_date.strftime("%Y-%m-%d"))
    
    if data.empty:
        data_load_status.text('Data loading failed or no data found.')
        st.stop()
    else:
        data_load_status.text(f'Loading data for {selected_ticker} from {start_date} done!')

    st.header(f"1. Raw {selected_ticker} Price Data")
    st.write(f"Displaying the last few rows of historical price data for {selected_ticker}.")
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", line=dict(color='royalblue')))
        fig.layout.update(title_text=f'{selected_ticker} Historical Close Price', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    plot_raw_data()

    st.header(f"2. {selected_ticker} Price Forecast")
    st.write(f"The Prophet model will now predict the price for the next {n_years} year(s) at {prediction_freq} frequency.")
    
    df_train = pd.DataFrame()
    df_train['ds'] = pd.to_datetime(data['Date'])
    df_train['y'] = data['Close']
    df_train.dropna(subset=['ds', 'y'], inplace=True)
    
    if df_train.shape[0] < 2:
        st.error("Insufficient valid data points to train the Prophet model.")
        st.stop()

    st.info("Training the Prophet model...")
    m = Prophet(
        changepoint_prior_scale=cp_scale,
        holidays_prior_scale=0.01,
        seasonality_prior_scale=season_scale,
        seasonality_mode='multiplicative'
    )
    m.fit(df_train)
    st.success("Model training complete!")

    if prophet_freq == 'D':
        period = n_years * 365
    elif prophet_freq == 'W':
        period = n_years * 52
    elif prophet_freq == 'M':
        period = n_years * 12

    future = m.make_future_dataframe(periods=period, freq=prophet_freq)
    forecast = m.predict(future)

    st.subheader('Forecast Data')
    st.write("Predicted prices for the future period, including uncertainty intervals:")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    st.subheader('Forecast Visualization')
    st.write("The forecast plot shows the predicted trend and uncertainty intervals.")
    fig1 = plot_plotly(m, forecast)
    fig1.update_layout(title=f'{selected_ticker} Price Forecast', xaxis_title='Date', yaxis_title='Price (USD)', hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Forecast Components")
    st.write("Breakdown of the forecast into trend and seasonality components:")
    fig2 = plot_components_plotly(m, forecast)
    st.plotly_chart(fig2, use_container_width=True)

    st.header("3. Model Performance")
    st.write("Evaluating the model's accuracy on training data and via cross-validation.")
    
    forecast_train = m.predict(df_train)
    r2 = r2_score(df_train['y'], forecast_train['yhat'])
    mae = mean_absolute_error(df_train['y'], forecast_train['yhat'])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">R-Squared (R²) on Training Data</div>
            <div class="metric-value">{r2:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Absolute Error (MAE) on Training Data</div>
            <div class="metric-value">${mae:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.info(f"""
    **Interpretation:**
    - **R-squared ({r2:.4f})**: {r2*100:.2f}% of price variance explained.
    - **MAE (${mae:,.2f})**: Average prediction error on training data.
    """)

    st.subheader("Cross-Validation Performance")
    st.write("Cross-validation simulates historical forecasts for better accuracy estimation.")

    with st.expander("Run Cross-Validation"):
        total_days = (df_train['ds'].max() - df_train['ds'].min()).days
        initial_days = min(365 * 2, int(total_days * 0.5))
        period_days = min(30, int(total_days * 0.1))
        horizon_days = 90

        initial_cv = st.text_input("Initial training period:", f"{initial_days} days")
        period_cv = st.text_input("Period between cutoffs:", f"{period_days} days")
        horizon_cv = st.text_input("Forecast horizon:", f"{horizon_days} days")

        if st.button("Start Cross-Validation"):
            with st.spinner('Running cross-validation...'):
                try:
                    df_cv = cross_validation(m, initial=initial_cv, period=period_cv, horizon=horizon_cv, parallel="processes")
                    df_p = performance_metrics(df_cv)

                    st.success("Cross-validation complete!")
                    st.write("Performance Metrics:")
                    st.write(df_p.head())

                    fig_cv = go.Figure()
                    fig_cv.add_trace(go.Scatter(x=df_p['horizon'], y=df_p['rmse'], mode='lines+markers', name='RMSE'))
                    fig_cv.add_trace(go.Scatter(x=df_p['horizon'], y=df_p['mae'], mode='lines+markers', name='MAE'))
                    fig_cv.update_layout(title='Cross-Validation Performance', xaxis_title='Horizon (days)', yaxis_title='Error')
                    st.plotly_chart(fig_cv, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during cross-validation: {e}")

else:
    st.info('Adjust the parameters in the sidebar and click "Run Prediction" to start.')

st.markdown("---")
st.markdown("Created by a world-class investor and app developer.")
