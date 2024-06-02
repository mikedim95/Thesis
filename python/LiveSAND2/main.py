import pandas as pd
import numpy as np
from Utils.slidingWindows import find_length, plotFig
from Utils.feature import Window
from Utils.sand import SAND
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
target_file_name = "target_file.txt"
slidingWindow = None
train_until = 4992
alpha = 0.5
batch_size = 2000


def get_all_data():
    global slidingWindow
    if target_file_name in files:
        # Construct the full file path
        target_file_path = os.path.join(datasets_directory, target_file_name)

        # Open the target file
        with open(target_file_path, 'r') as file:
            df = pd.read_csv(file, header=None).dropna().to_numpy()
            all_data = df[:, 0].astype(float)
        slidingWindow = find_length(all_data)
        print(len(all_data))
        return all_data
    else:
        print(f"{target_file_name} not found in {datasets_directory}")
        return "error"


def get_all_labels():
    global slidingWindow
    if target_file_name in files:
        # Construct the full file path
        target_file_path = os.path.join(datasets_directory, target_file_name)

        # Open the target file
        with open(target_file_path, 'r') as file:
            df = pd.read_csv(file, header=None).dropna().to_numpy()
            all_labels = df[:, 1].astype(int)
        slidingWindow = find_length(all_labels)
        print(len(all_labels))
        return all_labels
    else:
        print(f"{target_file_name} not found in {datasets_directory}")
        return "error"


start_time = time.time()
# Read the dataset from the file

data = get_all_data()
print(data[0:10])
label = get_all_labels()
print(label[0:10])
slidingWindow = find_length(data)

# Prepare data for semisupervised method.

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
clf.init_length = 35000
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
    data, label, score, slidingWindow, fileName='title', modelName=modelName)

end_time = time.time()
elapsed_time = end_time - start_time
with open(results_file_path, 'a') as results_file:
    # Write a new line with results
    results_file.write(f"fileName:{title}, AUC:{AUC}, R_AUC:{R_AUC}, Precision:{Precision}, Recall:{Recall}, F:{F}, ExistenceReward:{ExistenceReward}, OverlapReward:{OverlapReward}, AP:{AP}, R_AP:{R_AP}, Precisionk:{Precisionk}, R_precision:{
        Rprecision}, R_recall:{Rrecall}, R_f:{Rf}, tn_count:{tn_count}, fn_count:{fn_count}, fp_count:{fp_count}, tp_count:{tp_count}, elapsed_time:{elapsed_time} seconds\n")
