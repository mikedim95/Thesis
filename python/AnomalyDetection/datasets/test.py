import os


def process_files(input_folder, output_folder, range_start=None, range_end=None):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each file in the input folder
    for file in os.listdir(input_folder):
        if file.endswith(".txt"):

            input_file_path = os.path.join(input_folder, file)
            title = os.path.splitext(file)[0]
            parts = title.split('_')
            numbers = [int(num) for num in parts[-3:]]
            print("Last three numbers as integers:", numbers)
            # usefull data from title
            train_until = numbers[0]
            anomaly_from = numbers[1]
            anomaly_to = numbers[2]

            output_file_path = os.path.join(
                output_folder, file.replace(".txt", ".txt"))
            process_values(input_file_path, output_file_path,
                           anomaly_from, anomaly_to)


def process_values(input_file_path, output_file_path, anomaly_from=None, anomaly_to=None):
    with open(input_file_path, 'r') as f_in, open(output_file_path, 'w') as f_out:
        index = 0
        for line in f_in:
            index += 1
            # Strip any leading/trailing whitespace and split values
            values = line.strip().split()

            for i, value in enumerate(values, start=1):
                # Write the value and a comma
                f_out.write(value + ',')

                # Check if a range is specified and if the current index is within the range
                if anomaly_from is not None and anomaly_to is not None:
                    if anomaly_from <= index <= anomaly_to:
                        # Write 1 if the index is within the range
                        f_out.write('1\n')
                    else:
                        f_out.write('0\n')  # Write 0 otherwise
                else:
                    f_out.write('0\n')  # Write 0 if no range is specified


# Example usage:
input_folder = './'
output_folder = './reformed'

process_files(input_folder, output_folder)
