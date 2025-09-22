import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import streamlit as st


def sarima_forecast(df):
    # Convert 'Trans. Date' to datetime format
    df['Trans. Date'] = pd.to_datetime(df['Trans. Date'], format='%m/%d/%Y')

    # Remove commas from 'Amount' and convert to numeric
    df['Amount'] = df['Amount'].str.replace(',', '').astype(float)

    # Set 'Trans. Date' as index
    df.set_index('Trans. Date', inplace=True)

    # Resample data to monthly frequency and aggregate (sum) the amounts
    df_monthly = df.resample('M').sum()

    # Train-test split
    train = df_monthly[:-12]  # Training on all but the last 12 months
    test = df_monthly[-12:]  # Testing on the last 12 months

    # Define and fit the SARIMA model with simplified parameters
    model = SARIMAX(train['Amount'], order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
    model_fit = model.fit(disp=False)

    # Make predictions on the test set
    predictions = model_fit.get_forecast(steps=12).predicted_mean
    mae = mean_absolute_error(test['Amount'], predictions)

    # Plot the results
    st.write(f'Mean Absolute Error: {mae}')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train.index, train['Amount'], label='Training Data')
    ax.plot(test.index, test['Amount'], label='Actual Data')
    ax.plot(test.index, predictions, label='Predicted Data', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount Spent')
    ax.set_title('SARIMA Model - Actual vs Predicted')
    ax.legend()
    st.pyplot(fig)

    # Forecast for the next 12 months
    future_steps = 12
    forecast = model_fit.get_forecast(steps=future_steps).predicted_mean

    # Create a date range for the forecast
    last_date = df_monthly.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='M')

    # Plot the forecast along with the historical data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_monthly.index, df_monthly['Amount'], label='Historical Data')
    ax.plot(future_dates, forecast, label='Forecasted Data', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount Spent')
    ax.set_title('Monthly Spending Forecast for Next Year')
    ax.legend()
    st.pyplot(fig)

    # Print forecasted values
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Amount': forecast})
    st.write(forecast_df)
