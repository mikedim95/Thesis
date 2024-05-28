import os
import numpy as np
import calculateThreshold
import z_scoreEngine


def begin():

    # Get the directory of the currently executing Python script
    script_directory = os.path.dirname(os.path.abspath(__file__))
# Specify file paths relative to the script directory
    folder_path = os.path.join(script_directory, "../datasets")
    results_file_path = os.path.join(
        script_directory, "destributionResults.txt")
# List all files in the folder
    files = os.listdir(folder_path)
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            title = os.path.splitext(file)[0]
            parts = title.split('_')
            numbers = [int(num) for num in parts[-3:]]
            print("Last three numbers as integers:", numbers)
            # usefull data from title
            train_until = numbers[0]
            anomaly_from = numbers[1]
            anomaly_to = numbers[2]
            # Read the dataset from the file
            with open(file_path, 'r') as f:
                data = np.loadtxt(f)
            margin = calculateThreshold.calculate_threshold(data[:train_until])
            print(margin)
            threshold = margin[0]*margin[1]
            std_dev = margin[1]
            mean = margin[2]
            anomalies = z_scoreEngine.z_scoreEngine(
                threshold, std_dev, mean, data[train_until:])
            with open(results_file_path, 'a') as results_file:
                # Write a new line with results
                results_file.write(f"{title}: Threshold:{threshold}")
                results_file.write(f" No of anomalies found: {anomalies}\n")


begin()
