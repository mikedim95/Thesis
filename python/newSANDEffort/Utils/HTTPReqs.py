import json
import numpy as np
import requests
url = 'http://localhost:3000/endpoints'


def report_to_system(new_batch, score, elapsed_time):
    threshold = np.mean(score) + 3 * np.std(score)
    print("threshold:", threshold)

    # Create binary indicator array
    binary_indicators = (score > threshold).astype(int)
    print("Binary indicators:", binary_indicators)

    # Create JSON object
    json_output = {
        "values": new_batch.tolist(),
        "isAnomaly": binary_indicators.tolist()
    }
    json_string = json.dumps(json_output)
    print("JSON output:", json_string)

    payload = {
        'newBatch': newBatch,
        'label': label
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print('POST request successful.', response.json())

    except requests.exceptions.RequestException as e:
        print('POST request failed:', e)
