from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# Read the dataset


def read_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print("Error reading the dataset:", str(e))
        return None

# Detect anomalies using LOF


def detect_anomalies(train_data, test_data, contamination=0.1):
    lof = LocalOutlierFactor(contamination=contamination)
    lof.fit(train_data)
    anomalies = lof.fit_predict(test_data)
    return anomalies

# Main function


def main():
    # Path to the dataset
    file_path = "../datasets/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"

    # Read the dataset
    dataset = read_dataset(file_path)

    if dataset is not None:
        # Extract the column of interest
        # Assuming the dataset has a single column
        column_name = dataset.columns[0]
        data = dataset[[column_name]]

        # Split the dataset into train and test sets
        train_data = data[:35000]
        test_data = data[35000:]
        # Detect anomalies using LOF
        anomalies = detect_anomalies(train_data, test_data)

        # Print the indices of anomalies in the test data
        print("Indices of anomalies detected in test data:")
        for i, anomaly in enumerate(anomalies):
            if anomaly == -1:  # Anomaly detected
                print(i)


if __name__ == "__main__":
    main()
