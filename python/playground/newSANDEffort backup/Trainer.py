import requests
import json
import os
import pandas as pd
import numpy as np
import time
import sys
# Replace with your Flask endpoint URL
url = 'http://localhost:5000/train'

# Get the current directory
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, 'data.txt')


def send_training_data():
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
        print('successfullyTrained: ',
              response.json().get('successfullyTrained'))
        print("Goodbye!")
        sys.exit(0)
        return True
    except requests.exceptions.RequestException as e:
        print('POST request failed:', e)
        return False

# Repeat mechanism


def main_loop(interval=60, iterations=10):
    print("Warming up...")
    time.sleep(5)  # Pauses the program for 5 seconds
    print("Sending training data for training.")
    for i in range(iterations):
        print(f"{i + 1} try of total {iterations} tries")
        if send_training_data():
            break
        # Wait for the specified interval before next iteration
        time.sleep(interval)
    else:
        print("Failed to send training data after multiple attempts.")


# Start the main loop
if __name__ == "__main__":
    # Adjust interval (seconds) and iterations as needed
    main_loop(interval=10, iterations=10)
