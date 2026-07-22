# =========================================
# TEMPERATURE ANALYSIS AND FORECASTING PROJECT
# =========================================

# 1 - Import Required Libraries
# pandas -> for data handling
# matplotlib -> for plotting graphs
# statsmodels -> for time series forecasting

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# =========================================
# 2️⃣ Load the Dataset
# =========================================

# Load CSV file
data = pd.read_csv(r"C:\Users\Aryan\Downloads\archive(4)\city_temperature.csv")

# Display first 5 rows
print("First 5 rows of dataset:")
print(data.head())


# =========================================
# 3️⃣ Dataset Overview
# =========================================

# Show dataset information
print("\nDataset Info:")
print(data.info())

# Show statistical summary
print("\nSummary Statistics:")
print(data.describe())


# =========================================
# 4️⃣ Monthly Average Temperature
# =========================================

# Group data by Month and calculate average temperature
monthly_avg = data.groupby("Month")["AvgTemperature"].mean()

print("\nMonthly Average Temperature:")
print(monthly_avg)


# =========================================
# 5️⃣ Plot Seasonal Pattern
# =========================================

plt.figure()

# Plot month vs temperature
plt.plot(monthly_avg.index, monthly_avg.values)

plt.xlabel("Month")
plt.ylabel("Average Temperature")
plt.title("Seasonal Temperature Pattern")

plt.show()


# =========================================
# 6️⃣ Seasonal Pattern for One City
# Example: Mumbai
# =========================================

# Filter data for Mumbai
mumbai = data[data["City"] == "Mumbai"]

# Calculate monthly average for Mumbai
monthly_avg_city = mumbai.groupby("Month")["AvgTemperature"].mean()

plt.figure()

plt.plot(monthly_avg_city.index, monthly_avg_city.values)

plt.xlabel("Month")
plt.ylabel("Temperature")
plt.title("Seasonal Pattern - Mumbai")

plt.show()


# =========================================
# 7️⃣ Monthly Temperature Distribution
# =========================================

plt.figure()

# Boxplot for seasonal distribution
data.boxplot(column="AvgTemperature", by="Month")

plt.xlabel("Month")
plt.ylabel("Temperature")
plt.title("Monthly Temperature Distribution")

plt.suptitle("")  # remove default title

plt.show()


# =========================================
# 8️⃣ Create Date Column
# =========================================

# Combine Year, Month, Day into one Date column
data["Date"] = pd.to_datetime(data[["Year", "Month", "Day"]])

# Sort data by date
data = data.sort_values("Date")


# =========================================
# 9️⃣ Time Series Creation
# =========================================

# Use Mumbai temperature for time series example
ts_data = mumbai["AvgTemperature"]

plt.figure()
plt.plot(ts_data)

plt.title("Temperature Time Series - Mumbai")
plt.xlabel("Time")
plt.ylabel("Temperature")

plt.show()


# =========================================
# 10  ARIMA Forecast Model
# =========================================

# Fit ARIMA model
model = ARIMA(ts_data, order=(1,1,1))

model_fit = model.fit()

print("\nModel Summary:")
print(model_fit.summary())


# =========================================
# 11Forecast Future Temperature
# =========================================

# Predict next 30 days
forecast = model_fit.forecast(steps=30)

print("\nForecasted Temperatures:")
print(forecast)


# Plot forecast
plt.figure()

plt.plot(forecast)

plt.title("Temperature Forecast for Next 30 Days")
plt.xlabel("Days")
plt.ylabel("Predicted Temperature")

plt.show()
