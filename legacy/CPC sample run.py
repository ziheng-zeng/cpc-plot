import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from matplotlib.ticker import ScalarFormatter

# Function to read all CSV files for a given CPC from a folder
def read_csvs_from_folder(folder_path, cpc_id):
    data_frames = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv') and str(cpc_id) in file_name:
            file_path = os.path.join(folder_path, file_name)
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip')  # Skip problematic lines
                df['time'] = pd.to_datetime(df['time'])  # Convert 'time' to datetime
                data_frames.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return pd.concat(data_frames, ignore_index=True)

# Folder path containing CSV files for both CPCs
folder_path = 'D:/Documents/research spring 24/CPC data/sample run 24-2-16-2-19'  

# Reading and appending the data for each CPC
data1 = read_csvs_from_folder(folder_path, 3025)
data2 = read_csvs_from_folder(folder_path, 3772)

# Plotting
plt.figure(figsize=(12, 6))  # Adjusted for potentially better layout
plt.plot(data1['time'], data1['concentration'], label='CPC 3025')
plt.plot(data2['time'], data2['concentration'], label='CPC 3772')

# Formatting the plot
ax = plt.gca()  # Get current axis

# Set major locator to day and minor locator to hour, adjust as necessary
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))  # Adjust interval as needed

# Set major formatter to a more concise date format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
ax.tick_params(axis='x', which='major', pad=15)

# Set the formatter for the y-axis to use scientific notation
y_formatter = ScalarFormatter(useMathText=True)
y_formatter.set_scientific(True)
y_formatter.set_powerlimits((-3, 3))  # Adjust the range for which scientific notation is used
ax.yaxis.set_major_formatter(y_formatter)

# Increase font size for readability, adjust size as needed
plt.xticks(fontsize=10)  # Rotate to 45 degrees for better legibility

plt.xlabel('Time')
plt.ylabel('Concentration (ct/cc)')
plt.title('Concentration vs. Time for CPC 3025 and 3772')
plt.legend()
plt.tight_layout()

plt.show()