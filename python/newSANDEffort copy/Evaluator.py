import requests
import json
import os
import pandas as pd
import numpy as np
import time

# Replace with your Flask endpoint URL
url = 'http://main:5000/evaluateBatch'
main_health_url = 'http://main:5000/health'
trainer_health_url = 'http://trainer:5001/health'

# Wait for both main and trainer services to be ready
while True:
    try:
        main_response = requests.get(main_health_url)
        trainer_response = requests.get(trainer_health_url)
        if main_response.status_code == 200 and trainer_response.status_code == 200:
            print('Main and Trainer services are healthy.')
            break
    except requests.exceptions.RequestException as e:
        print('Waiting for main and trainer services to be healthy:', e)
    time.sleep(5)

# Get the current directory
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, 'data.txt')

# Example data (replace with your actual data)
with open(file_path, 'r') as f:
    df = pd.read_csv(f, header=None).dropna().to_numpy()
    newBatch = df[15000:20000, 0].astype(float)
    label = df[15000:20000, 1].astype(int)

# Convert numpy arrays to Python lists
newBatch = newBatch.tolist()
label = label.tolist()

# Create JSON payload
payload = {
    'newBatch': newBatch,
    'label': label
}

# Send POST request with JSON payload
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors
    print('POST request successful.', response.json())
except requests.exceptions.RequestException as e:
    print('POST request failed:', e)
