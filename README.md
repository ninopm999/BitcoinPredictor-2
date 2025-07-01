Bitcoin Price Prediction App
This Streamlit application uses Facebook's Prophet model to forecast the price of Bitcoin (BTC-USD).

Features
Fetches real-time Bitcoin data using the yfinance library.

Uses the Prophet model for time series forecasting.

Allows users to select the number of years for the prediction.

Visualizes historical data, the forecast, and model components using Plotly.

Evaluates the model's performance with R-squared and Mean Absolute Error (MAE) metrics.

How to Deploy
Create a GitHub Repository: Create a new repository on GitHub and upload the following files:

app.py

requirements.txt

README.md (optional, but recommended)

Sign up for Streamlit Cloud: If you haven't already, sign up for a free account at share.streamlit.io.

Deploy the App:

Click on "New app" from your Streamlit Cloud dashboard.

Connect your GitHub account and select the repository you just created.

Ensure the "Main file path" is set to app.py.

Click "Deploy!".

Streamlit will then build the application and deploy it to a public URL.
