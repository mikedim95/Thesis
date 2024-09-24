import requests
import json
import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify
import time

app = Flask(__name__)

# Replace with your Flask endpoint URL
url = 'http://main:5000/train'


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


# Get the current directory
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, 'data.txt')

# Example data (replace with your actual data)
with open(file_path, 'r') as f:
    df = pd.read_csv(f, header=None).dropna().to_numpy()
    trainData = df[:5000, 0].astype(float)
    label = df[:5000, 1].astype(int)

# Convert numpy arrays to Python lists
trainData = trainData.tolist()
label = label.tolist()

# Create JSON payload
payload = {
    'trainData': trainData,
    'label': label
}

# Wait for main service to be ready
while True:
    try:
        response = requests.get('http://main:5000/health')
        if response.status_code == 200:
            print('Main service is healthy.')
            break
    except requests.exceptions.RequestException as e:
        print('Waiting for main service to be healthy:', e)
    time.sleep(5)

# Send POST request with JSON payload
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors
    print('POST request successful.', response.json())
except requests.exceptions.RequestException as e:
    print('POST request failed:', e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
