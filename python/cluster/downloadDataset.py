import requests

# URL of the dataset
url = "https://cs.joensuu.fi/sipu/datasets/s1.txt"

# Send an HTTP request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Save the content to a file
    with open("s1.txt", "wb") as f:
        f.write(response.content)
    print("Dataset downloaded successfully.")
else:
    print("Failed to download the dataset.")
