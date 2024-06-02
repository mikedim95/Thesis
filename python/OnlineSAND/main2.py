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


def get_training_data(train_until):
    global slidingWindow
    if target_file_name in files:
        # Construct the full file path
        target_file_path = os.path.join(datasets_directory, target_file_name)

        # Open the target file
        with open(target_file_path, 'r') as file:
            df = pd.read_csv(file, header=None).dropna().to_numpy()
            trainning_data = df[:train_until, 0].astype(float)
        slidingWindow = find_length(trainning_data)
        return trainning_data
    else:
        print(f"{target_file_name} not found in {datasets_directory}")
        return "error"


def initialise_clf_trainning(training_data):
    global clf
    clf.overlaping_rate = int(4*slidingWindow)
    clf.decision_scores_ = []
    clf.alpha = 0.5
    clf.init_length = len(training_data)
    clf.batch_size = 2000
    begin_initial_training(training_data)


def begin_initial_training(training_data):
    global clf
    clf._initialize(training_data)
    """ clf._set_normal_model()
    print(clf.current_time, end='-->')
    clf.decision_scores_ = clf._run(
        training_data[:]) """


def bring_next_batch(i):
    global clf
    if target_file_name in files:
        # Construct the full file path
        target_file_path = os.path.join(datasets_directory, target_file_name)

        # Open the target file
        with open(target_file_path, 'r') as file:
            df = pd.read_csv(file, header=None).dropna().to_numpy()
            next_batch = df[train_until+i*batch_size:train_until +
                            (i+1)*batch_size, 0].astype(float)
        return next_batch
    else:
        print(f"{target_file_name} not found in {datasets_directory}")
        return "error"


def detect_anomalies_in_next_batch(next_batch):
    global clf
    print(next_batch, 'with length', len(next_batch))


training_data = get_training_data(train_until)
clf = SAND(pattern_length=slidingWindow,
           subsequence_length=4*(slidingWindow))
initialise_clf_trainning(training_data)
for i in range(5):
    next_batch = bring_next_batch(i)
    detect_anomalies_in_next_batch(next_batch)
