#from sklearn.metrics import mean_squared_error
#from flask import Flask, request, jsonify, url_for
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error as mae
#from sklearn.metrics import mean_absolute_percentage_error as mape
import csv
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
#from arima_model_selection import *
import dataframe_image as dfi


def predict(fileName, periodicity, forecast_periods=10):
    df = pd.read_csv(fileName)
    df.columns = ["Month", "Sales"]
    df.drop(106, axis=0, inplace=True)
    df.drop(105, axis=0, inplace=True)
    df['Month'] = pd.to_datetime(df['Month'])

    df.set_index('Month', inplace=True)

    periodicity = 'months'

    # Set the number of periods to forecast
    forecast_periods = int(forecast_periods)

    test_result = adfuller(df['Sales'])

    # Ho: It is non stationary
    # H1: It is stationary

    def adfuller_test(sales):
        result = adfuller(sales)
        labels = ['ADF Test Statistic', 'p-value',
                  '#Lags Used', 'Number of Observations Used']

    adfuller_test(df['Sales'])
    df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)
    df['Sales'].shift(1)
    df['Seasonal First Difference'] = df['Sales'] - df['Sales'].shift(12)

    model = ARIMA(df['Sales'], order=(1, 1, 1))
    model_fit = model.fit()
    df['forecast'] = model_fit.predict(start=90, end=103, dynamic=True)

    model = sm.tsa.statespace.SARIMAX(
        df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    df['forecast'] = results.predict(start=90, end=103, dynamic=True)
    from pandas.tseries.offsets import DateOffset
    # future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]

    # Convert the input values to integers
    forecast_periods = int(forecast_periods)
    # Create the future dates list using the DateOffset module
    if periodicity == "days":
        future_dates = [df.index[-1] +
                        DateOffset(days=x) for x in range(0, forecast_periods)]
    elif periodicity == "months":
        future_dates = [df.index[-1] +
                        DateOffset(months=x) for x in range(0, forecast_periods)]
    elif periodicity == "years":
        future_dates = [df.index[-1] +
                        DateOffset(years=x) for x in range(0, forecast_periods)]
    future_datest_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
    future_df = pd.concat([df, future_datest_df])
    future_df['forecast'] = results.predict(start=104, end=120, dynamic=True)
    future_df[['Sales', 'forecast']].plot(figsize=(12, 8))

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.savefig("static/output.png")
    # Output the predicted dataset
    future_df.drop(['Sales', 'Sales First Difference',
                   'Seasonal First Difference'], axis=1, inplace=True)
    future_df.rename(columns={'forecast': 'FORECAST'}, inplace=True)
    predicted_df = future_df.tail()
    print(predicted_df)
    dfi.export(predicted_df, 'static/input.png')
    try:
        plt.clf()
    except Exception:
        pass

    return {"output": "output.png", "data": "input.png", "analyze": analyze_data(fileName)}


def analyze_data(fileName):
    df = pd.read_csv(fileName)
    df.columns = ["Date", "Sales"]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Split the data into training and testing sets
    train = df[:len(df) - 12]
    test = df[len(df) - 12:]

    # Fit an ARIMA model to the training set
    model = ARIMA(train['Sales'], order=(1, 1, 1))
    model_fit = model.fit()

    # Make predictions on the testing set
    predictions = model_fit.predict(
        start=len(train), end=len(train) + len(test) - 1, dynamic=True)

    # Calculate mean squared error and other errors
    mse = mean_squared_error(test['Sales'], predictions)

    return {"mse": mse, "rmse": np.sqrt(mse), "mae": mae(test['Sales'], predictions), "mape": mape(test['Sales'], predictions)}