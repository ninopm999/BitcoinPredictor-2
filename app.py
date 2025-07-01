data = yf.download("BTC-USD", start="2017-01-01", progress=False)

# 🛡️ Cek apakah data valid dan mengandung kolom yang dibutuhkan
required_cols = {"Date", "Close"}
if data.empty or not required_cols.issubset(data.columns):
    st.error("❌ Gagal memuat data historis BTC-USD. Ticker mungkin tidak tersedia atau tidak lengkap.")
    st.stop()

# Reset index dan rename hanya jika kolom tersedia
data.reset_index(inplace=True)
df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

# Cek lagi apakah ds & y sudah valid
if not {"ds", "y"}.issubset(df_train.columns):
    st.error("❌ Kolom 'ds' dan 'y' tidak ditemukan setelah proses rename.")
    st.stop()

# Hapus baris kosong
df_train.dropna(subset=["ds", "y"], inplace=True)

# Minimal 2 baris
if df_train.shape[0] < 2:
    st.error("❌ Prophet membutuhkan minimal 2 baris data valid.")
    st.stop()
