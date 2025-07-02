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
st.set_page_config(page_title="Advanced Bitcoin Price Predictor",
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
        padding-top: 2rem;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #ff4b4b;
        border-radius:10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        width: 100%; /* Make button full width in sidebar */
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
        margin-bottom: 20px; /* Space between metric cards */
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
@st.cache_data(ttl=3600) # Cache data for 1 hour to balance freshness and performance
def load_data(ticker, start_date_str):
    """
    Loads historical data for the given ticker from a specified start date.
    Includes robust error handling for yfinance download issues and column identification.
    """
    try:
        data = yf.download(ticker, start=start_date_str)
        if data.empty:
            st.warning(f"No data downloaded for {ticker} from {start_date_str}. Please check the ticker and date range.")
            return pd.DataFrame() # Return empty DataFrame on failure
        
        # Reset index to bring Date from index to a regular column.
        data.reset_index(inplace=True)

        # Identify and ensure 'Date' column is present and correctly named
        # If 'Date' column doesn't exist, try renaming 'index' to 'Date'
        if 'Date' not in data.columns and 'index' in data.columns:
            data.rename(columns={'index': 'Date'}, inplace=True)
        
        # If 'Date' is still not found after attempting to rename 'index', it's a critical error
        if 'Date' not in data.columns:
            st.error(f"Could not find a 'Date' column in the downloaded data for {ticker}. The data structure might be unexpected.")
            return pd.DataFrame()

        # Check for 'Close' column
        if 'Close' not in data.columns:
            st.error(f"Missing 'Close' price column for {ticker}. Cannot proceed with prediction.")
            return pd.DataFrame()

        # Convert 'Date' to datetime. Coerce errors, as invalid dates will become NaT.
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        
        # Select only the necessary columns 'Date' and 'Close'
        # This step ensures that subsequent operations like dropna only deal with these columns
        # and explicitly verifies their existence by potentially raising a KeyError here if they're truly missing.
        df_cleaned = data[['Date', 'Close']].copy()

        # Now drop NaNs from these two columns. This dropna should now be safe.
        df_cleaned.dropna(subset=['Date', 'Close'], inplace=True)

        if df_cleaned.empty:
            st.warning(f"No valid data points remaining for {ticker} after cleaning. Please try a different date range or ticker.")
            return pd.DataFrame()

        return df_cleaned # Return the cleaned DataFrame
    except Exception as e:
        # Generic catch for any other unexpected errors during data downloading or processing
        st.error(f"An unexpected error occurred during data processing for {ticker}: {e}. Please check your internet connection or the ticker symbol.")
        return pd.DataFrame()

# --- Sidebar Inputs ---
st.sidebar.header("Data & Prediction Settings")

selected_ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., BTC-USD)", "BTC-USD").upper()
# Set default start date to a reasonable point where Bitcoin data is widely available
default_start_date = datetime.date(2017, 1, 1) 
start_date = st.sidebar.date_input("Select Start Date for Data", default_start_date)

n_years = st.sidebar.slider('Years to predict:', 1, 4, 1)
# Prediction granularity
prediction_freq = st.sidebar.selectbox(
    'Prediction Frequency:',
    ('Daily', 'Weekly', 'Monthly'),
    index=0 # Default to Daily
)
# Map human-readable frequency to Prophet's 'freq' argument
freq_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
prophet_freq = freq_map[prediction_freq]

st.sidebar.header("Prophet Model Parameters")
# Hyperparameter tuning sliders
cp_scale = st.sidebar.slider('Changepoint Prior Scale (flexibility):', 0.01, 0.5, 0.15, 0.01)
season_scale = st.sidebar.slider('Seasonality Prior Scale (strength):', 0.1, 20.0, 10.0, 0.1)

# Main prediction button
if st.sidebar.button('Run Prediction'):
    data_load_status = st.empty() # Placeholder for loading status messages
    data_load_status.text('Loading data...')
    data = load_data(selected_ticker, start_date.strftime("%Y-%m-%d"))
    
    if data.empty:
        data_load_status.text('Data loading failed or no data found.')
        st.stop() # Stop execution if data is not loaded successfully
    else:
        data_load_status.text(f'Loading data for {selected_ticker} from {start_date} done!')

    st.header(f"1. Raw {selected_ticker} Price Data")
    st.write(f"Displaying the last few rows of historical price data for {selected_ticker}.")
    st.write(data.tail())

    # --- Plotting Raw Data ---
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", line=dict(color='royalblue')))
        fig.layout.update(title_text=f'{selected_ticker} Historical Close Price', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    plot_raw_data()

    # --- Forecasting with Prophet ---
    st.header(f"2. {selected_ticker} Price Forecast")
    st.write(f"The Prophet model will now predict the price for the next {n_years} year(s) at {prediction_freq} frequency.")
    
    # Prepare data for Prophet
    df_train = pd.DataFrame()
    df_train['ds'] = pd.to_datetime(data['Date']) # Ensure 'ds' is datetime type
    df_train['y'] = data['Close']

    # Ensure no NaN values in 'ds' or 'y' which can cause issues for Prophet
    df_train.dropna(subset=['ds', 'y'], inplace=True)
    
    # Validate that there are enough rows after dropping NaNs for Prophet.
    if df_train.shape[0] < 2:
        st.error("Insufficient valid data points to train the Prophet model after cleaning. Please try a different date range or ticker.")
        st.stop() 

    # Initialize and train the model with user-selected hyperparameters
    st.info("Training the Prophet model...")
    m = Prophet(
        changepoint_prior_scale=cp_scale,
        holidays_prior_scale=0.01, # Can be exposed as a slider if needed in future
        seasonality_prior_scale=season_scale,
        seasonality_mode='multiplicative' # Good for crypto where volatility increases with price
    )
    m.fit(df_train)
    st.success("Model training complete!")

    # Create future dataframe and make predictions
    # Calculate rough number of periods based on years and frequency
    if prophet_freq == 'D':
        period = n_years * 365
    elif prophet_freq == 'W':
        period = n_years * 52
    elif prophet_freq == 'M':
        period = n_years * 12

    future = m.make_future_dataframe(periods=period, freq=prophet_freq)
    forecast = m.predict(future)

    # Display forecast data
    st.subheader('Forecast Data')
    st.write("Below are the predicted prices for the future period, including lower and upper bounds of the uncertainty interval.")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Plot forecast
    st.subheader('Forecast Visualization')
    st.write("The main forecast plot shows the predicted trend (blue line) and the uncertainty intervals (light blue shaded area).")
    fig1 = plot_plotly(m, forecast)
    fig1.update_layout(
        title=f'{selected_ticker} Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode="x unified" # Improves tooltip experience
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Plot forecast components
    st.subheader("Forecast Components")
    st.write("These charts break down the forecast into its underlying components: overall trend, yearly seasonality, and weekly seasonality.")
    fig2 = plot_components_plotly(m, forecast)
    st.plotly_chart(fig2, use_container_width=True)

    # --- Model Evaluation ---
    st.header("3. Model Performance")
    st.write("Evaluating the model's accuracy on the training data and through cross-validation.")
    
    # Get predictions for the training data period for R2 and MAE
    forecast_train = m.predict(df_train)
    
    # Calculate metrics
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
    **Interpretation on Training Data:**
    - **R-squared ({r2:.4f})**: This indicates that approximately **{r2*100:.2f}%** of the variance in {selected_ticker}'s price is explained by the model on the data it was trained on. A higher value (closer to 1.0) is better.
    - **MAE (${mae:,.2f})**: On average, the model's predictions on the training data were off by about **${mae:,.2f}**. A lower value indicates higher accuracy.
    """)

    # --- Cross-Validation ---
    st.subheader("Cross-Validation Performance")
    st.write("Cross-validation evaluates the model's performance on unseen data by simulating historical forecasts. This gives a more reliable estimate of accuracy.")

    with st.expander("Run Cross-Validation (May take a while for large datasets)"):
        st.write("You can adjust the initial training period, period between cutoffs, and forecast horizon for cross-validation.")
        
        # Calculate reasonable initial and period values based on total data length
        total_days = (df_train['ds'].max() - df_train['ds'].min()).days
        # Ensure initial training period is at least 2 years or 50% of data, whichever is smaller
        initial_days = min(365 * 2, int(total_days * 0.5)) 
        # Ensure period between cutoffs is at least 1 month or 10% of data, whichever is smaller
        period_days = min(30, int(total_days * 0.1)) 
        horizon_days = 90 # Default 3 months forecast horizon for evaluation

        initial_cv = st.text_input("Initial training period (e.g., '730 days' for 2 years):", f"{initial_days} days")
        period_cv = st.text_input("Period between cutoffs (e.g., '180 days' for 6 months):", f"{period_days} days")
        horizon_cv = st.text_input("Forecast horizon (e.g., '90 days' for 3 months):", f"{horizon_days} days")

        if st.button("Start Cross-Validation"):
            with st.spinner('Running cross-validation... This might take some time.'):
                try:
                    df_cv = cross_validation(m, initial=initial_cv, period=period_cv, horizon=horizon_cv, parallel="processes")
                    df_p = performance_metrics(df_cv)

                    st.success("Cross-validation complete!")
                    st.write("Performance Metrics from Cross-Validation:")
                    st.write(df_p.head())

                    fig_cv = go.Figure()
                    fig_cv.add_trace(go.Scatter(x=df_p['horizon'], y=df_p['rmse'], mode='lines+markers', name='RMSE'))
                    fig_cv.add_trace(go.Scatter(x=df_p['horizon'], y=df_p['mae'], mode='lines+markers', name='MAE'))
                    fig_cv.update_layout(title='Cross-Validation Performance (RMSE & MAE vs. Forecast Horizon)',
                                         xaxis_title='Horizon (days)',
                                         yaxis_title='Error')
                    st.plotly_chart(fig_cv, use_container_width=True)

                    st.info(f"""
                    **Interpretation of Cross-Validation:**
                    These metrics show how the model's errors change as the forecast horizon increases.
                    - **RMSE (Root Mean Squared Error)**: Measures the average magnitude of the errors.
                    - **MAE (Mean Absolute Error)**: Measures the average magnitude of the errors without considering direction.
                    You want both RMSE and MAE to be as low as possible. A consistent increase in errors with horizon is typical.
                    """)

                except Exception as e:
                    st.error(f"Error during cross-validation: {e}. Please check the input periods and ensure enough historical data is available. Ensure your initial training period is long enough to train the model properly.")

else:
    st.info('Adjust the prediction parameters in the sidebar and click "Run Prediction" to start the forecast.')

st.markdown("---")
st.markdown("Created by a world-class investor and app developer.")
