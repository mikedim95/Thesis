import pandas as pd
import numpy as np
from Utils.slidingWindows import find_length, plotFig
from Utils.feature import Window
from Utils.sand import SAND
from Utils.HTTPReqs import report_to_system
from sklearn.preprocessing import MinMaxScaler
import os
import time
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Global variable to store the model
model_state = {
    "clf": None,
    "slidingWindow": None
}


def save_model_state(clf, filename='model_checkpoint.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(clf, file)
    print(f"Model state saved as {filename}")


def load_model_state(filename='model_checkpoint.pkl'):
    try:
        with open(filename, 'rb') as file:
            clf = pickle.load(file)

        return clf
    except FileNotFoundError:
        print(f"No checkpoint file found at {filename}. Starting fresh.")
        return None


def TrainTheModelInitially(trainData, label):
    start_time = time.time()
    print('Training model from scratch with ',
          len(trainData), ' data points...')
    slidingWindow = find_length(trainData)

    clf = SAND(pattern_length=slidingWindow,
               subsequence_length=4*(slidingWindow))
    clf.overlaping_rate = int(4*slidingWindow)
    clf.ts = trainData
    clf.decision_scores_ = []
    clf.alpha = 0.5
    clf.init_length = len(trainData)
    clf.batch_size = 2000
    clf._initialize()
    clf._set_normal_model()
    clf.decision_scores_ = clf._run(clf.ts)
    print("Done analysing data...")
    clf.decision_scores_ = np.array(clf.decision_scores_)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        score.reshape(-1, 1)).ravel()
    AUC, R_AUC, Precision, Recall, F, ExistenceReward, OverlapReward, AP, R_AP, Precisionk, Rprecision, Rrecall, Rf, tn_count, fn_count, fp_count, tp_count = plotFig(
        trainData, label, score, slidingWindow, fileName='title', modelName='modelName')

    end_time = time.time()
    elapsedTime = end_time - start_time
    return clf


def update_model_with_newBatch(clf, newBatch, label):
    start_time = time.time()
    clf.ts = newBatch
    clf.batch_size = len(newBatch)
    clf.current_time = 0
    while clf.current_time < len(clf.ts) - clf.subsequence_length:
        print('Analysing data...')
        clf._run_next_batch()
        clf._set_normal_model()
        if clf.current_time < len(clf.ts) - clf.subsequence_length:
            clf.decision_scores_ += clf._run(
                clf.ts[clf.current_time -
                       clf.batch_size:min(len(clf.ts), clf.current_time)]
            )
        else:
            clf.decision_scores_ += clf._run(
                clf.ts[clf.current_time - clf.batch_size:]
            )
    print("Done analysing data...")
    clf.decision_scores_ = np.array(clf.decision_scores_)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        score.reshape(-1, 1)).ravel()

    AUC, R_AUC, Precision, Recall, F, ExistenceReward, OverlapReward, AP, R_AP, Precisionk, Rprecision, Rrecall, Rf, tn_count, fn_count, fp_count, tp_count = plotFig(
        newBatch, label, score, clf.slidingWindow, fileName='title', modelName='modelName')

    save_model_state(clf)  # Save model state after processing the batch

    end_time = time.time()
    elapsedTime = end_time - start_time
    anomalyId = report_to_system(newBatch, score, elapsedTime)
    return anomalyId


@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "No JSON received"}), 400

    # Extract specific key-value pairs from the received data
    trainData = data.get('trainData')
    label = data.get('label')
    trainData = np.array(trainData)
    label = np.array(label)

    clf = TrainTheModelInitially(trainData, label)
    model_state['clf'] = clf
    model_state['slidingWindow'] = find_length(trainData)
    save_model_state(clf)

    clf = load_model_state()  # Load model state if exists
    if clf:
        return jsonify({"successfullyTrained": True})
    else:
        return jsonify({"successfullyTrained": False})


@app.route('/evaluateBatch', methods=['POST'])
def evaluateBatch():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "No JSON received"}), 400

    # Extract specific key-value pairs from the received data
    newBatch = data.get('newBatch')
    label = data.get('label')
    newBatch = np.array(newBatch)
    label = np.array(label)

    clf = model_state.get('clf')
    if clf is None:
        clf = load_model_state()
        if clf is None:
            return jsonify({"error": "Model not initialized"}), 400
        print(f"Model state loaded from checkpoint")
    anomalyId = update_model_with_newBatch(clf, newBatch, label)
    model_state['clf'] = clf

    return jsonify({"decision_scores": len(clf.decision_scores_), "anomalyId": anomalyId})


if __name__ == '__main__':
    clf = load_model_state()  # Load model state if exists
    if clf:
        print(f"Model state loaded from checkpoint")
        model_state['clf'] = clf
        # Assuming this is correct
        model_state['slidingWindow'] = clf.pattern_length
    app.run(host='0.0.0.0', port=5000)
