import requests
import json
import os
import pandas as pd
import numpy as np

# Replace with your Flask endpoint URL
url = 'http://localhost:5000/train'

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

# Send POST request with JSON payload
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors
    print('POST request successful.', response.json())
except requests.exceptions.RequestException as e:
    print('POST request failed:', e)
