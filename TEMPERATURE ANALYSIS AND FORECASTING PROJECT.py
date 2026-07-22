import pandas as pd

# Read the CSV file
file_path = r"C:\Users\Aryan\Downloads\archive(4)\city_temperature.csv"
city_temp = pd.read_csv(file_path)

# Quick look at the data
print(city_temp.head())
print(city_temp.info())


# Average temperature by country
avg_by_country = city_temp.groupby("Country")["AvgTemperature"].mean().sort_values(ascending=False)
print(avg_by_country)

# Average temperature by city
avg_by_city = city_temp.groupby("City")["AvgTemperature"].mean().sort_values(ascending=False)
print(avg_by_city)

# Average temperature by year
avg_by_year = city_temp.groupby("Year")["AvgTemperature"].mean()
print(avg_by_year)


# Country average (India)
india_avg = city_temp[city_temp["Country"] == "India"]["AvgTemperature"].mean()

# City average (Chennai / Madras)
chennai_avg = city_temp[city_temp["City"].str.contains("Chennai|Madras", case=False)]["AvgTemperature"].mean()

print("India Avg Temperature:", india_avg)
print("Chennai Avg Temperature:", chennai_avg)


import matplotlib.pyplot as plt

# Prepare data for plotting
labels = ["India (Country Avg)", "Chennai (City Avg)"]
values = [india_avg, chennai_avg]

plt.figure(figsize=(8,6))
bars = plt.bar(labels, values, color=["red", "blue"])

# Annotate values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.1f}", ha="center", va="bottom")

plt.title("Average Temperature Comparison: India vs Chennai")
plt.ylabel("Avg Temperature")
plt.grid(axis="y", linestyle="--", alpha=0.7)  # optional: add gridlines
plt.show()


import seaborn as sns

india_monthly = city_temp[city_temp["Country"] == "India"].groupby("Month")["AvgTemperature"].mean()
chennai_monthly = city_temp[city_temp["City"].str.contains("Chennai|Madras", case=False)].groupby("Month")["AvgTemperature"].mean()

monthly_df = pd.DataFrame({
    "India": india_monthly,
    "Chennai": chennai_monthly
})

plt.figure(figsize=(8,6))
sns.heatmap(monthly_df.T, annot=True, cmap="coolwarm", cbar=True)
plt.title("Monthly Avg Temperature: India vs Chennai")
plt.xlabel("Month")
plt.ylabel("Region")
plt.show()


subset = city_temp[(city_temp["Country"] == "India") | 
                   (city_temp["City"].str.contains("Chennai|Madras", case=False))]

plt.figure(figsize=(8,6))
sns.boxplot(data=subset, x="Country", y="AvgTemperature")
sns.boxplot(data=subset, x="City", y="AvgTemperature", color="lightblue")

plt.title("Temperature Distribution: India vs Chennai")
plt.ylabel("Avg Temperature")
plt.show()
