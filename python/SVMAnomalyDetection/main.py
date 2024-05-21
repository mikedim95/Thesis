import time
import os
from Utils.slidingWindows import find_length, plotFig
from Utils.feature import Window
from Utils.ocsvm import OCSVM
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Get the directory of the currently executing Python script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Specify file paths relative to the script directory
datasets_directory = os.path.join(script_directory, "../Datasets/reformed")
results_file_path = os.path.join(script_directory, "../Results/SVMResults.txt")

# List all files in the folder
files = os.listdir(datasets_directory)

with open(results_file_path, 'a') as results_file:
    # Write a new line with results
    results_file.write(
        f"-------> THIS IS THE SVM METHOD REPORT <----------\n\n")

for file in files:
    if file.endswith(".txt"):
        start_time = time.time()
        file_path = os.path.join(datasets_directory, file)
        title = os.path.splitext(file)[0]
        parts = title.split('_')
        numbers = [num for num in parts[-3:]]
        print("Last three numbers as integers:", numbers)

        # Useful data from title
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

            # Prepare data for semisupervised method
            data_train = data[:train_until]
            data_test = data[train_until:]

            X_train = Window(window=slidingWindow).convert(
                data_train).to_numpy()
            X_test = Window(window=slidingWindow).convert(data_test).to_numpy()

            print("Estimated Subsequence length: ", slidingWindow)
            print("Time series length: ", len(data))
            print("Number of abnormal points: ", list(label).count(1))

            modelName = 'OCSVM'
            X_train_ = MinMaxScaler(feature_range=(
                0, 1)).fit_transform(X_train.T).T
            X_test_ = MinMaxScaler(feature_range=(
                0, 1)).fit_transform(X_test.T).T

            clf = OCSVM()
            clf.fit(X_train_, X_test_)

            score = clf.decision_scores_
            score = np.array([score[0]] * math.ceil((slidingWindow - 1) / 2) +
                             list(score) + [score[-1]] * ((slidingWindow - 1) // 2))
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                score.reshape(-1, 1)).ravel()

            # plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName)
            AUC, R_AUC, Precision, Recall, F, ExistenceReward, OverlapReward, AP, R_AP, Precisionk, Rprecision, Rrecall, Rf, tn_count, fn_count, fp_count, tp_count = plotFig(
                data, label, score, slidingWindow, fileName=title, modelName=modelName)

            end_time = time.time()
            elapsed_time = end_time - start_time

            with open(results_file_path, 'a') as results_file:
                # Write a new line with results
                results_file.write(f"fileName:{title}, AUC:{AUC}, R_AUC:{R_AUC}, Precision:{Precision}, Recall:{Recall}, F:{F}, ExistenceReward:{ExistenceReward}, OverlapReward:{OverlapReward}, AP:{AP}, R_AP:{R_AP}, Precisionk:{
                                   Precisionk}, R_precision:{Rprecision}, R_recall:{Rrecall}, R_f:{Rf}, tn_count:{tn_count}, fn_count:{fn_count}, fp_count:{fp_count}, tp_count:{tp_count}, elapsed_time:{elapsed_time} seconds\n")
