import pandas as pd
import numpy as np
from SAND.slidingWindows import find_length, plotFig
from SAND.feature import Window
from SAND.sand import SAND
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("datasets/MBA_ECG805_data.out",
                 header=None).dropna().to_numpy()

max_length = 79795

data = df[:max_length, 0].astype(float)
label = df[:max_length, 1].astype(int)

slidingWindow = find_length(data)
X_data = Window(window=slidingWindow).convert(data).to_numpy()

# Prepare data for semisupervised method.
# Here, the training ratio = 0.1

data_train = data[:int(0.1 * len(data))]
data_test = data

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
plotFig(data, label, score, slidingWindow, fileName='MBA_ECG805_data',
        modelName=modelName)  # , plotRange=[1775,2200]
