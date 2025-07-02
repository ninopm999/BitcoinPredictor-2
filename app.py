import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objs as go
import datetime

st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")
st.title("Bitcoin Price Prediction App ₿")
st.markdown("Forecast future BTC-USD prices using Facebook Prophet.")
st.markdown("---")

@st.cache_data(ttl=3600)
def load_data(ticker, start_date_str):
    try:
        data = yf.download(ticker, start=start_date_str)

        if data.empty:
            st.warning(f"No data downloaded for {ticker} from {start_date_str}. Please check the ticker or try again later.")
            return pd.DataFrame()

        data.reset_index(inplace=True)

        if 'Date' not in data.columns:
            if 'index' in data.columns:
                data.rename(columns={'index': 'Date'}, inplace=True)
            else:
                st.error(f"Date column missing in data for {ticker}.")
                return pd.DataFrame()

        if 'Close' not in data.columns:
            st.error(f"'Close' price column is missing for {ticker}.")
            return pd.DataFrame()

        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data.dropna(subset=['Date', 'Close'], inplace=True)

        if data.empty:
            st.warning(f"No usable data found after cleaning for {ticker}. Try a different date or wait a while.")
            return pd.DataFrame()

        return data[['Date', 'Close']].copy()

    except yf.shared._exceptions.YFRateLimitError:
        st.error("Yahoo Finance has temporarily rate-limited your request. Please wait a few minutes and try again.")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"An unexpected error occurred while processing {ticker}: {e}")
        return pd.DataFrame()

# --- Sidebar Inputs ---
st.sidebar.header("Settings")
selected_ticker = st.sidebar.text_input("Ticker Symbol", "BTC-USD").upper()
def_start = datetime.date(2017, 1, 1)
start_date = st.sidebar.date_input("Start Date", def_start)
n_years = st.sidebar.slider('Years to Predict', 1, 4, 1)
pred_freq = st.sidebar.selectbox('Prediction Frequency', ('Daily', 'Weekly', 'Monthly'))
freq_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
prophet_freq = freq_map[pred_freq]

cp_scale = st.sidebar.slider('Changepoint Prior Scale', 0.01, 0.5, 0.15, 0.01)
season_scale = st.sidebar.slider('Seasonality Prior Scale', 0.1, 20.0, 10.0, 0.1)

if st.sidebar.button('Run Prediction'):
    data_status = st.empty()
    data_status.text('Loading data...')
    data = load_data(selected_ticker, start_date.strftime("%Y-%m-%d"))

    if data.empty:
        data_status.text('Data loading failed.')
        st.stop()
    else:
        data_status.text('Data loaded.')

    st.subheader("Historical Data")
    st.write(data.tail())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
    fig.layout.update(title=f"{selected_ticker} Price", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    df_train = pd.DataFrame()
    df_train['ds'] = pd.to_datetime(data['Date'])
    df_train['y'] = data['Close']
    df_train.dropna(subset=['ds', 'y'], inplace=True)

    if df_train.shape[0] < 2:
        st.error("Not enough data to train model.")
        st.stop()

    m = Prophet(changepoint_prior_scale=cp_scale, seasonality_prior_scale=season_scale, seasonality_mode='multiplicative')
    m.fit(df_train)
    st.success("Model trained.")

    period = n_years * {'D': 365, 'W': 52, 'M': 12}[prophet_freq]
    future = m.make_future_dataframe(periods=period, freq=prophet_freq)
    forecast = m.predict(future)

    st.subheader("Forecast Data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    st.subheader("Forecast Plot")
    fig1 = plot_plotly(m, forecast)
    fig1.update_layout(title=f"{selected_ticker} Forecast", xaxis_title='Date', yaxis_title='Price (USD)')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Forecast Components")
    fig2 = plot_components_plotly(m, forecast)
    st.plotly_chart(fig2, use_container_width=True)
