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

            # usefull data from title
            train_until = numbers[0]
            anomaly_from = numbers[1]
            anomaly_to = numbers[2]

            output_file_path = os.path.join(
                output_folder, file.replace(".txt", ".txt"))
            process_values(input_file_path, output_file_path,
                           anomaly_from, anomaly_to, title)


def process_values(input_file_path, output_file_path, anomaly_from=None, anomaly_to=None, title=None):
    count = 0
    index = 0
    with open(input_file_path, 'r') as f_in, open(output_file_path, 'w') as f_out:

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
                        count += 1
                    else:
                        f_out.write('0\n')  # Write 0 otherwise
                else:
                    f_out.write('0\n')  # Write 0 if no range is specified
    if count == 0:
        print(title)
        print(f"Processed until {index} index.")


# Example usage:
input_folder = './virgin'
output_folder = './reformed'

process_files(input_folder, output_folder)
