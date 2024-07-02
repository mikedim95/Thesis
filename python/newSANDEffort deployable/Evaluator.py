import time
import os
import requests
import pandas as pd
import numpy as np


def wait_for_training_complete(signal_path, timeout=60):
    """Wait for the training to be completed."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(signal_path):
            print("Training complete signal found!")
            os.remove(signal_path)  # Remove the signal file after reading it
            return True
        else:
            print("Waiting for training to complete...")
            time.sleep(5)
    print("Timeout reached. Training did not complete in time.")
    return False


def wait_for_main_service(url, timeout=60):
    """Wait for the main service to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Main service is up and running!")
                return True
        except requests.ConnectionError:
            print("Main service is not available yet, retrying...")
            time.sleep(5)
    print("Timeout reached. Main service is still not available.")
    return False


# Replace with your Flask endpoint URL
url = 'http://main:5000/evaluateBatch'
health_url = 'http://main:5000/health'

# Path to the signal file
signal_path = '/app/shared/training_complete.txt'

# Wait for the training to be complete
if wait_for_training_complete(signal_path):
    # Wait for the main service to be ready
    if wait_for_main_service(health_url):
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
    else:
        print("Main service did not start in time.")
else:
    print("Training did not complete in time.")
