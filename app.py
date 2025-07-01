import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objs as go

st.set_page_config(page_title="Bitcoin Price Predictor", page_icon="₿", layout="wide")

st.title("Bitcoin Price Prediction App ₿")
st.markdown("This app uses **Facebook Prophet** to forecast Bitcoin (BTC-USD) prices.")

@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start="2017-01-01").reset_index()

data_load_state = st.text('📥 Loading Bitcoin price data...')
try:
    data = load_data("BTC-USD")
    if data.empty:
        st.error("⚠️ Data kosong. Gagal mengunduh data BTC-USD dari Yahoo Finance.")
        st.stop()
    data_load_state.text('✅ Data berhasil dimuat!')
except Exception as e:
    st.error(f"❌ Error saat mengambil data: {e}")
    st.stop()

st.sidebar.header("Prediction Settings")
n_years = st.sidebar.slider("Years of prediction:", 1, 4)
period = n_years * 365

if st.sidebar.button("Predict"):

    st.subheader("1. Historical Bitcoin Price Data")
    st.write(data.tail())

    def plot_raw():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Open"))
        fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close"))
        fig.update_layout(title="Bitcoin Price Time Series", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    plot_raw()

    st.subheader("2. Forecasting with Prophet")

    required_cols = {"Date", "Close"}
    if not required_cols.issubset(data.columns):
        st.error(f"❌ Data tidak memiliki kolom yang dibutuhkan: {required_cols - set(data.columns)}")
        st.stop()

    df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

    if 'ds' not in df_train.columns or 'y' not in df_train.columns:
        st.error("❌ Rename kolom gagal. Tidak ditemukan 'ds' atau 'y'")
        st.stop()

    if df_train[["ds", "y"]].isnull().all().any():
        st.error("❌ Semua nilai 'ds' atau 'y' adalah NaN.")
        st.stop()

    df_train.dropna(subset=["ds", "y"], inplace=True)

    if df_train.shape[0] < 2:
        st.error("❌ Data terlalu sedikit setelah dibersihkan. Prophet butuh minimal 2 baris.")
        st.stop()

    m = Prophet(seasonality_mode='multiplicative')
    m.fit(df_train)

    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader("Forecast Result")
    st.write(forecast.tail())

    fig1 = plot_plotly(m, forecast)
    fig1.update_layout(title="Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Forecast Components")
    fig2 = plot_components_plotly(m, forecast)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("3. Model Performance")
    forecast_train = m.predict(df_train)
    r2 = r2_score(df_train["y"], forecast_train["yhat"])
    mae = mean_absolute_error(df_train["y"], forecast_train["yhat"])
    st.metric("R² (Accuracy)", f"{r2:.4f}")
    st.metric("MAE (Error)", f"${mae:,.2f}")

    st.info(f"R² means ~{r2*100:.2f}% of variance is explained.\nMAE means predictions are off by ${mae:,.2f} on average.")
else:
    st.info("Klik tombol Predict di sidebar untuk memulai peramalan.")

st.markdown("---")
st.markdown("Created by a world-class investor and app developer.")
