from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Endpoint for Service B


@app.route('/response', methods=['POST'])
def response():
    data = request.get_json()
    # Extract specific key-value pairs from the received data
    custom_key_1 = data.get('key')

    print(f"Received custom_key_1: {custom_key_1}")

    # Return the response along with the extracted data
    return jsonify({
        "received_data": {
            "custom_key_1": custom_key_1,
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
