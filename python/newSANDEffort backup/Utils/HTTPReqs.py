import json
import numpy as np
import requests
url = 'https://thesis-gray.vercel.app/api/reportAnomaly/'


def report_to_system(newBatch, score, elapsedTime):
    threshold = np.mean(score) + 3 * np.std(score)
    print("threshold:", threshold)

    # Create binary indicator array
    binary_indicators = (score > threshold).astype(int)

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
        anomaly_id = response.json().get('anomalyId')
        if anomaly_id:
            return anomaly_id

        else:
            print('Anomaly ID not found in the response.')

    except requests.exceptions.RequestException as e:
        print('POST request failed:', e)
