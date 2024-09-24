import os

# Path to the directory containing dataset files
dataset_directory = "C:/Users/Mike/Documents/Dev/Thesis/python/datasets/reformed"
# Path to the result file
result_file_path = "C:/Users/Mike/Documents/Dev/Thesis/python/Results/SUBSEQUENCE_ANOMALY_DETECTION_method.txt"


# Function to count the number of datapoints in a dataset file


def count_datapoints(filename):
    file_path = os.path.join(dataset_directory, filename)
    try:
        with open(file_path, 'r') as file:
            # Count the number of lines
            return len(file.readlines())
    except FileNotFoundError:
        return 0  # Return 0 if file is not found


# Read the result file
with open(result_file_path, 'r') as result_file:
    result_lines = result_file.readlines()

# Process the result file and add datapoint counts
with open(result_file_path, 'w') as result_file:
    for line in result_lines:
        # Extract the filename from the result line
        if "fileName" in line:
            # Extract the dataset file name (after 'fileName:' and before ',')
            filename = line.split(",")[0].split(":")[1].strip() + ".txt"

            # Count the number of datapoints from the corresponding dataset file
            datapoint_count = count_datapoints(filename)

            # Append the datapoint count to the line
            line = line.strip() + f", datapoint_count:{datapoint_count}\n"

        # Write the modified line back to the result file
        result_file.write(line)

print("Result file updated successfully.")
