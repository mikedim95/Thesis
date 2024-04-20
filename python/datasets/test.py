def process_values(input_file, output_file, range_start=None, range_end=None):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        index = 0
        for line in f_in:
            index += 1
            print(index)
            # Strip any leading/trailing whitespace and split values
            values = line.strip().split()

            for i, value in enumerate(values, start=1):
                # Write the value and a comma
                f_out.write(value + ',')

                # Check if a range is specified and if the current index is within the range
                if range_start is not None and range_end is not None:
                    if range_start <= index <= range_end:
                        # Write 1 if the index is within the range
                        f_out.write('1\n')
                    else:
                        f_out.write('0\n')  # Write 0 otherwise
                else:
                    f_out.write('0\n')  # Write 0 if no range is specified


# Example usage:
input_file = './001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt'
output_file = 'output.out'
range_start = 52000  # Specify the start index of the range
range_end = 52620   # Specify the end index of the range

process_values(input_file, output_file, range_start, range_end)
