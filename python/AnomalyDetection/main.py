import pandas as pd
import numpy as np
from SAND.slidingWindows import find_length, plotFig
from SAND.feature import Window
from SAND.sand import SAND
from sklearn.preprocessing import MinMaxScaler
import os
""" df = pd.read_csv("datasets/MBA_ECG805_data.out",
                 header=None).dropna().to_numpy()

max_length = 79795

data = df[:max_length, 0].astype(float)
label = df[:max_length, 1].astype(int)

slidingWindow = find_length(data)
X_data = Window(window=slidingWindow).convert(data).to_numpy()

# Prepare data for semisupervised method.
# Here, the training ratio = 0.1

data_train = data[:35000]
data_test = data[35001:]

X_train = Window(window=slidingWindow).convert(data_train).to_numpy()
X_test = Window(window=slidingWindow).convert(data_test).to_numpy()

print("Estimated Subsequence length: ", slidingWindow)
print("Time series length: ", len(data))
print("Number of abnormal points: ", list(label).count(1))

modelName = 'SAND (offline)'
clf = SAND(pattern_length=slidingWindow, subsequence_length=4*(slidingWindow))
x = data
clf.fit(x, overlaping_rate=int(1.5*slidingWindow))
score = clf.decision_scores_
score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
    score.reshape(-1, 1)).ravel()
plotFig(data, label, score, slidingWindow, fileName='name',
        modelName=modelName)  # , plotRange=[1775,2200]
 """

# Get the directory of the currently executing Python script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Specify file paths relative to the script directory
folder_path = os.path.join(script_directory, "datasets/reformed")
results_file_path = os.path.join(
    script_directory, "destributionResults.txt")
# List all files in the folder
files = os.listdir(folder_path)
for file in files:
    if file.endswith(".txt"):
        file_path = os.path.join(folder_path, file)
        title = os.path.splitext(file)[0]
        parts = title.split('_')
        numbers = [num for num in parts[-3:]]
        print("Last three numbers as integers:", numbers)
        # usefull data from title
        train_until = int(numbers[0])
        anomaly_from = numbers[1]
        anomaly_to = numbers[2]
      # Read the dataset from the file
        with open(file_path, 'r') as f:
            df = pd.read_csv(f, header=None).dropna().to_numpy()
            data = df[:len(df), 0].astype(float)
            label = df[:len(df), 1].astype(int)
            slidingWindow = find_length(data)
            X_data = Window(window=slidingWindow).convert(data).to_numpy()

            # Prepare data for semisupervised method.

            data_train = data[:train_until]
            data_test = data[train_until+1:]

            X_train = Window(window=slidingWindow).convert(
                data_train).to_numpy()
            X_test = Window(window=slidingWindow).convert(data_test).to_numpy()

            print("Estimated Subsequence length: ", slidingWindow)
            print("Time series length: ", len(data))
            print("Number of abnormal points: ", list(label).count(1))

            modelName = 'SAND (offline)'
            clf = SAND(pattern_length=slidingWindow,
                       subsequence_length=4*(slidingWindow))
            x = data
            clf.fit(x, overlaping_rate=int(1.5*slidingWindow))
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                score.reshape(-1, 1)).ravel()
            plotFig(data, label, score, slidingWindow, fileName='name',
                    modelName=modelName)  # , plotRange=[1775,2200]
