import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Function to forecast demand for the next 15 weeks
def forecast_demand(stock_code, data):
    stock_data = data[data['StockCode'] == stock_code]
    weekly_data = stock_data['Quantity'].resample('W').sum()

    train_size = int(len(weekly_data) * 0.8)
    train, test = weekly_data[:train_size], weekly_data[train_size:]

    # Define and train the SARIMAX model
    seasonal_order = (1, 1, 1, 52)  # Adjust as necessary for your data
    model = SARIMAX(train, order=(5, 1, 0), seasonal_order=seasonal_order)
    model_fit = model.fit()

    # Make predictions for the train and test sets
    train_pred = model_fit.predict(start=0, end=len(train) - 1)
    test_pred = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)

    # Forecasting next 15 weeks
    forecast_steps = 15
    future_pred = model_fit.forecast(steps=forecast_steps)
    future_dates = pd.date_range(start=weekly_data.index[-1] + pd.Timedelta(weeks=1), periods=forecast_steps, freq='W')

    # Plotting the actual vs predicted demand
    plt.figure(figsize=(12, 8))
    plt.plot(train.index, train, label='Train Actual Demand', color='blue', marker='o')
    plt.plot(train.index, train_pred, label='Train Predicted Demand', color='red', linestyle='--', marker='o')
    plt.plot(test.index, test, label='Test Actual Demand', color='orange', marker='o')
    plt.plot(test.index, test_pred, label='Test Predicted Demand', color='green', linestyle='--', marker='o')
    plt.plot(future_dates, future_pred, label='15 Weeks Forecast', color='purple', linestyle='--', marker='o')

    plt.title(f'Demand Overview and Forecast for {stock_code}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Calculate errors for both train and test sets
    train_errors = train - train_pred
    test_errors = test - test_pred

    # Set up the matplotlib figure for error distribution
    plt.figure(figsize=(14, 6))

    # Plotting the training error distribution
    plt.subplot(1, 2, 1)
    sns.histplot(train_errors, bins=30, kde=True, color='blue', stat='density')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Training Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Density')

    # Plotting the testing error distribution
    plt.subplot(1, 2, 2)
    sns.histplot(test_errors, bins=30, kde=True, color='orange', stat='density')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Testing Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Density')

    # Show the plots
    plt.tight_layout()
    st.pyplot(plt)

# Main Streamlit app
def main():
    st.title("Demand Forecasting")

    # Load your data (replace 'cleaned_data.csv' with your actual data file)
    data = pd.read_csv('cleaned_data.csv', parse_dates=['InvoiceDate'], index_col='InvoiceDate')

    data['StockCode'] = data['StockCode'].astype(str)  # Ensure StockCode is a string

    # Get top 10 StockCodes by Quantity
    top_10_stock_codes = data.groupby('StockCode')['Quantity'].sum().nlargest(10).index.tolist()

    # Sidebar for selecting stock code
    st.sidebar.header("Select Stock Code")
    selected_stock = st.sidebar.selectbox("Select a Stock Code:", top_10_stock_codes)

    # Display selected stock code as a heading
    st.header(f"Demand Overview and 15 Weeks Forecast for: {selected_stock}")

    if st.button("Forecast Demand"):
        forecast_demand(selected_stock, data)

if __name__ == "__main__":
    main()
