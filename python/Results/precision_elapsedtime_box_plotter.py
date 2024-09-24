import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


import re


def parse_metrics(file_path):
    metrics = {
        'Precision': [],
        'tn_count': [],
        'fn_count': [],
        'fp_count': [],
        'tp_count': [],
        'elapsed_time_per_1000_dataponts': [],
        'datapoint_count': [],
        'total_count': []
    }
    with open(file_path, 'r') as file:
        for line in file:

            match = re.search(
                r'AUC:([\d\.]+), R_AUC:[\d\.]+, Precision:([\d\.]+), Recall:([\d\.]+), F:([\d\.]+), '
                r'ExistenceReward:[\d\.]+, OverlapReward:[\d\.]+, AP:([\d\.]+), R_AP:[\d\.]+, '
                r'Precisionk:[\d\.]+, R_precision:[\d\.]+, R_recall:[\d\.]+, R_f:[\d\.]+, '
                r'tn_count:(\d+), fn_count:(\d+), fp_count:(\d+), tp_count:(\d+), '
                r'elapsed_time:([\d\.]+) seconds, datapoint_count:(\d+)', line
            )
            if match:

                # Correct group indexing
                metrics['Precision'].append(float(match.group(2)))
                metrics['tn_count'].append(int(match.group(6)))
                metrics['fn_count'].append(int(match.group(7)))
                metrics['fp_count'].append(int(match.group(8)))
                metrics['tp_count'].append(int(match.group(9)))
                time = round(float(match.group(10)), 2)
                datapoints = int(match.group(11))
                metrics['elapsed_time_per_1000_dataponts'].append(
                    time*1000/datapoints)
                metrics['datapoint_count'].append(int(match.group(11)))
        metrics['total_count'].append(sum(metrics['tn_count']))
        metrics['total_count'].append(sum(metrics['fn_count']))
        metrics['total_count'].append(sum(metrics['fp_count']))
        metrics['total_count'].append(sum(metrics['tp_count']))

    return metrics


def plot_precision_box(all_metrics):
    precision_data = []
    algo_names = []

    # Collecting precision values and algorithm names
    for algo_name, metrics in all_metrics.items():
        if 'Precision' in metrics:
            # Add all precision values
            precision_data.extend(metrics['Precision'])
            # Repeat the algorithm name
            algo_names.extend([algo_name] * len(metrics['Precision']))

    # Create a DataFrame for easier plotting
    data = {'Algorithm': algo_names, 'Precision': precision_data}

    # Create the boxplot using seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Algorithm', y='Precision', data=data)

    # Add title and labels
    plt.title('Precision Distribution by Algorithm')
    plt.xlabel('Algorithms')
    plt.ylabel('Precision')

    # Show the plot
    plt.show()


def plot_time_box(all_metrics):
    precision_data = []
    algo_names = []

    # Collecting precision values and algorithm names
    for algo_name, metrics in all_metrics.items():
        if 'elapsed_time_per_1000_dataponts' in metrics:
            # Add all elapsed_time values
            precision_data.extend(metrics['elapsed_time_per_1000_dataponts'])
            # Repeat the algorithm name
            algo_names.extend(
                [algo_name] * len(metrics['elapsed_time_per_1000_dataponts']))

    # Create a DataFrame for easier plotting
    data = {'Algorithm': algo_names,
            'elapsed_time_per_1000_dataponts': precision_data}

    # Create the boxplot using seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Algorithm', y='elapsed_time_per_1000_dataponts', data=data)

    # Set y-axis to log scale
    plt.yscale('log')

    # Add title and labels
    plt.title('Elapsed Time per 1000 Datapoints Distribution by Algorithm')
    plt.xlabel('Algorithms')
    plt.ylabel('Elapsed Time (Log Scale)')

    # Show the plot
    plt.show()


def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(script_directory)

    all_metrics = {}
    for file in files:
        if file.endswith('_method.txt'):
            algo_name = file.split('_method')[0].upper()
            file_path = os.path.join(script_directory, file)

            all_metrics[algo_name] = parse_metrics(file_path)

    if not all_metrics:
        print("No valid result files found.")
        return

    metrics_to_plot = ['Precision',
                       'tn_count', 'fn_count', 'fp_count',  'tp_count', 'ElapsedTime', 'total_count']
    num_metrics = len(metrics_to_plot)
    """ print(all_metrics['ISOLATION_FOREST']['elapsed_time_per_1000_dataponts']) """
    plot_time_box(all_metrics)
    plot_precision_box(all_metrics)


if __name__ == "__main__":
    main()
