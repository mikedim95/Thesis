import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model_state = {}


def load_model_state():
    try:
        with open('/app/model_data/model_checkpoint.pkl', 'rb') as file:
            clf = pickle.load(file)
            print("Model loaded successfully.")
            return clf
    except FileNotFoundError:
        print("No checkpoint file found at model_checkpoint.pkl. Starting fresh.")
        return None


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

    clf = model_state.get('clf')
    if clf is None:
        clf = SAND(pattern_length=4, subsequence_length=4)
        model_state['clf'] = clf

    # Training logic
    clf.fit(trainData, label)

    # Save the model state
    with open('/app/model_data/model_checkpoint.pkl', 'wb') as file:
        pickle.dump(clf, file)
    print("Model state saved to model_checkpoint.pkl")

    return jsonify({"tn_count": len(trainData) - np.sum(label)})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


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

    decision_scores = update_model_with_newBatch(clf, newBatch, label)
    model_state['clf'] = clf

    # Ensure decision_scores is a list or a NumPy array
    if isinstance(decision_scores, dict):
        decision_scores = np.array(list(decision_scores.values()))

    return jsonify({"decision_scores": decision_scores.tolist()})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    clf = load_model_state()  # Load model state if exists
    if clf:
        model_state['clf'] = clf
        model_state['slidingWindow'] = clf.pattern_length
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
