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

required_cols = {"Date", "Close"}
if not required_cols.issubset(data.columns):
    st.error(f"❌ Data yang diunduh tidak memiliki kolom {required_cols}")
    st.stop()

data.reset_index(inplace=True)

st.sidebar.header("⏳ Prediction Settings")
n_years = st.sidebar.slider("Berapa tahun ke depan?", 1, 4, 1)
period = n_years * 365

if st.sidebar.button("🔮 Prediksi Sekarang"):

    st.subheader("1. 📈 Data Historis Harga Bitcoin")
    st.write(data.tail())

    def plot_raw():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Open"))
        fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close"))
        fig.update_layout(title="Pergerakan Harga BTC", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    plot_raw()

    st.subheader("2. 🔮 Forecasting dengan Prophet")

    try:
        df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

        if not {"ds", "y"}.issubset(df_train.columns):
            st.error("❌ Gagal membuat kolom 'ds' dan 'y'")
            st.stop()

        df_train.dropna(subset=["ds", "y"], inplace=True)

        if df_train.shape[0] < 2:
            st.error("❌ Prophet membutuhkan minimal 2 baris data valid.")
            st.stop()

        m = Prophet(seasonality_mode='multiplicative')
        m.fit(df_train)

        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader("📊 Data Hasil Prediksi")
        st.write(forecast.tail())

        fig1 = plot_plotly(m, forecast)
        fig1.update_layout(title="Prediksi Harga BTC", xaxis_title="Tanggal", yaxis_title="Harga (USD)")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📉 Komponen Prediksi")
        fig2 = plot_components_plotly(m, forecast)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("3. 📏 Evaluasi Akurasi Model")
        forecast_train = m.predict(df_train)
        r2 = r2_score(df_train["y"], forecast_train["yhat"])
        mae = mean_absolute_error(df_train["y"], forecast_train["yhat"])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² (Akurasi)", f"{r2:.4f}")
        with col2:
            st.metric("MAE (Error)", f"${mae:,.2f}")

        st.info(f"""
📌 **R² = {r2:.2f}** → sekitar {r2*100:.1f}% variasi harga dijelaskan oleh model.  
📌 **MAE = ${mae:,.2f}** → rata-rata kesalahan prediksi harian.
""")

    except Exception as e:
        st.error(f"❌ Gagal melakukan forecasting: {e}")
        st.stop()

else:
    st.info("Silakan klik tombol **🔮 Prediksi Sekarang** di sidebar untuk memulai forecasting.")

st.markdown("---")
st.caption("🚀 Dibuat oleh NPM Marketing | Powered by Facebook Prophet & Streamlit")
