import time
import requests
import os
import pandas as pd
import numpy as np


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
url = 'http://main:5000/train'
health_url = 'http://main:5000/health'

# Wait for the main service to be ready
if wait_for_main_service(health_url):
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

        # Write signal file
        signal_path = '/app/shared/training_complete.txt'
        with open(signal_path, 'w') as signal_file:
            signal_file.write('Training complete')
        print('Training complete signal written.')

    except requests.exceptions.RequestException as e:
        print('POST request failed:', e)
else:
    print("Main service did not start in time.")
