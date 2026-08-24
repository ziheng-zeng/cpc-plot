import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# Load the uploaded CSV file to check its content
file_path = 'C:/Users/zengz/Downloads/MANY_20240405_125242.csv'
df = pd.read_csv(file_path)

# Convert datetime columns to datetime objects for plotting
df['datetime'] = pd.to_datetime(df['datetime'])
df['datetime.1'] = pd.to_datetime(df['datetime.1'])

# Create the plot
plt.figure(figsize=(12, 6))

# Plot for CPC 3025_1
plt.plot(df['datetime'], df['concentration'], label='CPC 3025_1', color='blue')

# Plot for CPC 3010
plt.plot(df['datetime.1'], df['concentration.1'], label='CPC 3010', color='green')

# Formatting the plot with less cluttered x-axis
plt.title('CPC 3010 and CPC 3025_1 Concentration vs. Time')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator())

# Show plot
plt.tight_layout()
plt.show()

# Calculate the ratio of concentrations directly, ensuring no division by zero
df_simple = df.copy()
df_simple['concentration.1'].replace(0, np.nan, inplace=True)  # Replace 0 with NaN to avoid division by zero
df_simple.dropna(subset=['concentration', 'concentration.1'], inplace=True)  # Ensure we have valid data for both concentrations
df_simple['ratio'] = df_simple['concentration'] / df_simple['concentration.1']

# Plot the ratio vs the datetime column
plt.figure(figsize=(12, 6))
plt.plot(df_simple['datetime'], df_simple['ratio'], label='Ratio (CPC 3025_1 / CPC 3010)', color='darkcyan')
plt.title('Ratio of CPC 3025_1 to CPC 3010 Concentration vs. Time')
plt.xlabel('Time')
plt.ylabel('Ratio')
plt.xticks(rotation=45)
plt.legend()
plt.grid(visible=True)
plt.tight_layout()  # Adjust layout to not cut off labels

plt.show()