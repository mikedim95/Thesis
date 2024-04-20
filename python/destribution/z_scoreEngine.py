def z_scoreEngine(threshold, std_dev, mean, testing_data):
    anomalies = 0

    for i, data_point in enumerate(testing_data):

        if i % 100 == 0:
            print("Processing data point index:", i)
            z_score = (data_point - mean) / std_dev
        if abs(z_score) > threshold:
            # anomalies.append(X + i)  # Add the index of the anomaly to the list
            print("FOUND ANOMALY!")
            anomalies += 1

    return anomalies
