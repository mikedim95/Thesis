import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
# Get the directory of the currently executing Python script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Specify file paths relative to the script directory
folder_path = os.path.join(script_directory, "../datasets")
results_file_path = os.path.join(
    script_directory, "destributionResults.out")
# List all files in the folder
files = os.listdir(folder_path)
for file in files:
    if file.endswith(".txt"):
        file_path = os.path.join(folder_path, file)
        title = os.path.splitext(file)[0]
        parts = title.split('_')
        numbers = [int(num) for num in parts[-3:]]
        print("Last three numbers as integers:", numbers)
        # usefull data from title
        train_until = numbers[0]
        anomaly_from = numbers[1]
        anomaly_to = numbers[2]
      # Read the dataset from the file
        with open(file_path, 'r') as f:
            data = np.loadtxt(f)
        timestamps = pd.date_range(
            start='2020-01-01', periods=len(data), freq='ms')
        ts = pd.Series(
            data, index=timestamps)

        decomposition = seasonal_decompose(ts, model='additive', period=209)
        plt.figure(figsize=(10, 8))
        plt.subplot(8, 1, 1)
        plt.plot(ts, label='Original')
        plt.legend()

        window_size = 3740  # Adjust window size as needed
        smoothed_trend = decomposition.trend.rolling(
            window=window_size, min_periods=1).mean()
        plt.subplot(8, 1, 2)
        plt.plot(smoothed_trend, label='Trend')
        plt.legend()

        smoothed_season = decomposition.seasonal.rolling(
            window=window_size, min_periods=1).mean()
        plt.subplot(8, 1, 3)
        plt.plot(decomposition.seasonal, label='Seasonal')
        plt.legend()

        plt.subplot(8, 1, 4)
        plt.plot(decomposition.resid, label='Residual')
        plt.legend()

        """ plt.subplot(8, 1, 6)
        plot_acf(ts, lags=500)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation Function (ACF)') """
        """ stl = STL(ts, seasonal=1001)
        result = stl.fit()

        plt.subplot(8, 1, 5)
        plt.plot(ts, label='Original stl')
        plt.legend()

        window_size = 500  # Adjust window size as needed
        smoothed_trend = result.trend.rolling(
            window=window_size, min_periods=1).mean()

        plt.subplot(8, 1, 6)
        plt.plot(smoothed_trend, label='trend stl')
        plt.legend()

        plt.subplot(8, 1, 7)
        plt.plot(result.seasonal, label='seasonal stl')
        plt.legend()

        plt.subplot(8, 1, 8)
        plt.plot(result.resid, label='resid stl')
        plt.legend() """

        plt.tight_layout()
        plt.show()
