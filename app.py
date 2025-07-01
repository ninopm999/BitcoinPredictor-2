import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objs as go

st.set_page_config(page_title="Bitcoin Price Predictor", page_icon="₿", layout="wide")

st.title("🔮 Bitcoin Price Prediction App")
st.markdown("This app uses **Facebook Prophet** to forecast future Bitcoin prices (BTC-USD).")

@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start="2017-01-01", progress=False)

# Load data safely
st.info("📥 Mengambil data historis dari Yahoo Finance...")
data = load_data("BTC-USD")

# Validasi data
if data is None or data.empty:
    st.error("❌ Gagal mengunduh data BTC-USD dari Yahoo Finance. Data kosong.")
    st.stop()

required_cols = {"Date", "Close", "Open"}
if not required_cols.issubset(data.columns):
    st.error(f"❌ Data yang diunduh tidak memiliki kolom: {required_cols - set(data.columns)}")
    st.stop()

data.reset_index(inplace=True)

st.sidebar.header("⏳ Prediction Settings")
n_years = st.sidebar.slider("Berapa tahun ke depan?", 1, 4, 1)
period = n_years * 365

if st.sidebar.button("🔮 Prediksi Sekarang"):

    st.subheader("1. 📈 Data Historis Harga Bitcoin")
    st.write(data.tail())

    d
