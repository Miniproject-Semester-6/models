import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
import calendar


def preprocess_data(df):
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.rename(columns={"date": "ds", "expenses": "y"})


def add_seasonality(model, days_in_month):
    model.add_seasonality(name="monthly", period=days_in_month, fourier_order=5)


def plot_forecast(model, forecast):
    fig = model.plot(forecast)
    plt.xlabel("Date")
    plt.ylabel("Expenses")
    plt.title("Forecasted Daily Expenses")
    plt.show()


def plot_components(model, forecast):
    fig = model.plot_components(forecast)
    plt.show()


def filter_current_month_forecast(forecast, current_month, current_year):
    return forecast[
        (forecast["ds"].dt.month == current_month)
        & (forecast["ds"].dt.year == current_year)
    ]


def format_forecasted_data(current_month_forecast):
    forecasted_data = []
    for index, row in current_month_forecast.iterrows():
        date = row["ds"]
        expenses = row["yhat"]
        forecasted_data.append(
            {"date": date.strftime("%Y-%m-%d"), "expenses": expenses}
        )
    return forecasted_data


def forecast(data, api=False):
    current_month = datetime.now().month
    current_year = datetime.now().year
    days_in_current_month = calendar.monthrange(current_year, current_month)[1]

    df = pd.DataFrame(data)
    df = preprocess_data(df)

    model = Prophet()
    add_seasonality(model, days_in_current_month)
    model.fit(df)

    future = model.make_future_dataframe(periods=days_in_current_month)
    forecast = model.predict(future)

    if not api:
        plot_forecast(model, forecast)
        plot_components(model, forecast)

    current_month_forecast = filter_current_month_forecast(
        forecast, current_month, current_year
    )

    forecasted_data = format_forecasted_data(current_month_forecast)
    return forecasted_data
