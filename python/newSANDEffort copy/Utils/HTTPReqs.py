import json
import numpy as np
import requests
url = 'http://localhost:3000/api/reportAnomaly'


def report_to_system(newBatch, score, elapsedTime):
    threshold = np.mean(score) + 3 * np.std(score)
    print("threshold:", threshold)

    # Create binary indicator array
    binary_indicators = (score > threshold).astype(int)
    print("Binary indicators:", binary_indicators)

    # Create JSON object
    json_output = {
        "values": newBatch.tolist(),
        "anomalyScores": score.tolist(),
        "indicators": binary_indicators.tolist(),
        "elapsedTime": elapsedTime,
        "threshold": threshold
    }
    try:
        response = requests.post(url, json=json_output)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print('POST request successful.', response.json())

    except requests.exceptions.RequestException as e:
        print('POST request failed:', e)
