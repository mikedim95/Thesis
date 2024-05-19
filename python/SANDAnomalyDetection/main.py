import pandas as pd
import numpy as np
from SAND.slidingWindows import find_length, plotFig
from SAND.feature import Window
from SAND.sand import SAND
from sklearn.preprocessing import MinMaxScaler
import os
import time

# Get the directory of the currently executing Python script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Specify file paths relative to the script directory
datasets_directory = os.path.join(script_directory, "../Datasets/reformed")
results_file_path = os.path.join(
    script_directory, "../Results/SANDResults.txt")
# List all files in the folder
files = os.listdir(datasets_directory)
with open(results_file_path, 'a') as results_file:
    # Write a new line with results
    results_file.write(
        f"-------> THIS IS THE SAND METHOD REPORT <----------\n\n")
for file in files:
    if file.endswith(".txt"):
        start_time = time.time()
        file_path = os.path.join(datasets_directory, file)
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

            modelName = 'SAND (online)'
            clf = SAND(pattern_length=slidingWindow,
                       subsequence_length=4*(slidingWindow))
            x = data

            """clf.fit(x, online=True, alpha=0.5, init_length=5000, batch_size=2000,
                    verbose=True, overlaping_rate=int(4*slidingWindow))"""

            clf.overlaping_rate = int(4*slidingWindow)
            clf.ts = list(x)
            clf.decision_scores_ = []
            clf.alpha = 0.5
            clf.init_length = 5000
            clf.batch_size = 2000
            print(clf.current_time, end='-->')
            clf._initialize()
            clf._set_normal_model()
            clf.decision_scores_ = clf._run(
                clf.ts[:min(len(clf.ts), clf.current_time)])
            while clf.current_time < len(clf.ts)-clf.subsequence_length:
                print(clf.current_time, end='-->')
                clf._run_next_batch()
                clf._set_normal_model()
                if clf.current_time < len(clf.ts)-clf.subsequence_length:
                    clf.decision_scores_ += clf._run(
                        clf.ts[clf.current_time-clf.batch_size:min(len(clf.ts), clf.current_time)])
                else:
                    clf.decision_scores_ += clf._run(
                        clf.ts[clf.current_time-clf.batch_size:])

            print("[STOP]: score length {}".format(
                len(clf.decision_scores_)))
            clf.decision_scores_ = np.array(clf.decision_scores_)

            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                score.reshape(-1, 1)).ravel()
            AUC, R_AUC, Precision, Recall, F, ExistenceReward, OverlapReward, AP, R_AP, Precisionk, Rprecision, Rrecall, Rf, tn_count, fn_count, fp_count, tp_count = plotFig(
                data, label, score, slidingWindow, fileName=title, modelName=modelName)

            end_time = time.time()
            elapsed_time = end_time - start_time
            with open(results_file_path, 'a') as results_file:
                # Write a new line with results
                results_file.write(f"fileName:{title}, AUC:{AUC}, R_AUC:{R_AUC}, Precision:{Precision}, Recall:{Recall}, F:{F}, ExistenceReward:{ExistenceReward}, OverlapReward:{OverlapReward}, AP:{AP}, R_AP:{R_AP}, Precisionk:{Precisionk}, R_precision:{
                                   Rprecision}, R_recall:{Rrecall}, R_f:{Rf}, tn_count:{tn_count}, fn_count:{fn_count}, fp_count:{fp_count}, tp_count:{tp_count}, elapsed_time:{elapsed_time} seconds\n")
