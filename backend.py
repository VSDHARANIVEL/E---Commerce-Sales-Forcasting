pip install kagglehub prophet matplotlib pandas numpy
pip install pystan==2.19.1.1 prophet
# --- Import Required Libraries ---
import kagglehub
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# --- 1. Download Dataset from Kaggle ---
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("vijayuv/onlineretail")
print("Path to dataset files:", path)

# --- 2. Load Dataset ---
# Adjust the file name if it's different
file_path = os.path.join(path, "OnlineRetail.csv")
df = pd.read_csv(file_path, encoding="ISO-8859-1")

print("\nOriginal Data Head:")
print(df.head())

# --- 3. Data Preprocessing ---
# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Remove negative or zero quantities (returns or invalid)
df = df[df['Quantity'] > 0]

# Compute TotalSales = Quantity * UnitPrice
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

# Group by date to get daily sales
daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalSales'].sum().reset_index()
daily_sales = daily_sales.rename(columns={'InvoiceDate': 'ds', 'TotalSales': 'y'})

# Convert 'ds' to datetime
daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])

print("\nProcessed Daily Sales Data:")
print(daily_sales.head())

# --- 4. Train Prophet Model ---
print("\nTraining Prophet model...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.fit(daily_sales)

# --- 5. Create Future DataFrame for 90 Days Ahead ---
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# --- 6. Plot Forecast ---
print("\nPlotting forecast...")
model.plot(forecast)
plt.title("E-commerce Sales Forecast (Next 90 Days)")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()

# --- 7. Optional: Show Last Few Forecasted Values ---
print("\nForecasted Values:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
