import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import calendar
import json


# Generate input data for 4 months
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 4, 30)

data = []
current_date = start_date
while current_date <= end_date:
    expense = random.uniform(100, 10000)
    data.append({'date': current_date, 'expenses': expense})
    current_date += timedelta(days=1)

# Convert input data to DataFrame
df = pd.DataFrame(data)

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Extract date part only
df['date'] = df['date'].dt.date

# Renaming columns as expected by Prophet
df = df.rename(columns={'date': 'ds', 'expenses': 'y'})

# Calculate the number of days in the current month dynamically
current_month = datetime.now().month
current_year = datetime.now().year
days_in_current_month = calendar.monthrange(current_year, current_month)[1]

# Initialize Prophet model
model = Prophet()

# Add seasonality - assuming monthly seasonality (period=30) based on your requirement
model.add_seasonality(name='monthly', period=days_in_current_month, fourier_order=5)

# Fit the model to the data
model.fit(df)

# Generate future dates for prediction (current month)
future = model.make_future_dataframe(periods=days_in_current_month)

# Make predictions for the future dates
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.xlabel('Date')
plt.ylabel('Expenses')
plt.title('Forecasted Daily Expenses')
plt.show()

# Optionally, plot components (trend, yearly and weekly seasonality)
fig = model.plot_components(forecast)
plt.show() 

# Filter forecasted data for the current month
current_month_forecast = forecast[(forecast['ds'].dt.month == current_month) & (forecast['ds'].dt.year == current_year)]

# Forecasted data for the current month
forecasted_data = []

for index, row in current_month_forecast.iterrows():
    date = row['ds']
    expenses = row['yhat']
    forecasted_data.append({'date': date.strftime('%Y-%m-%d'), 'expenses': expenses})

# Save forecasted data for the current month to a JSON file
with open('forecasted_data_current_month.json', 'w') as json_file:
    json.dump(forecasted_data, json_file)

print("Forecasted data for the current month saved to 'forecasted_data_current_month.json'")
