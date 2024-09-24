import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


def parse_metrics(file_path):
    metrics = {
        'Precision': []
    }
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(
                r'AUC:([\d\.]+), R_AUC:[\d\.]+, Precision:([\d\.]+), Recall:([\d\.]+), F:([\d\.]+), '
                r'ExistenceReward:[\d\.]+, OverlapReward:[\d\.]+, AP:([\d\.]+), R_AP:[\d\.]+, '
                r'Precisionk:[\d\.]+, R_precision:[\d\.]+, R_recall:[\d\.]+, R_f:[\d\.]+, '
                r'tn_count:[\d\.]+, fn_count:[\d\.]+, fp_count:[\d\.]+, tp_count:[\d\.]+, '
                r'elapsed_time:([\d\.]+)', line)
            if match:
                metrics['Precision'].append(float(match.group(2)))
    return metrics


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

    metrics_to_plot = ['Precision']
    num_metrics = len(metrics_to_plot)

    # Handle the case where num_metrics is 1
    fig, axes = plt.subplots(num_metrics, 1, figsize=(16, 8 * num_metrics))
    if num_metrics == 1:
        axes = [axes]  # Convert single Axes to list for uniformity

    for i, metric in enumerate(metrics_to_plot):
        # Histogram with density plot for metric distribution
        for algo, metrics in all_metrics.items():
            sns.histplot(metrics[metric], kde=True,
                         ax=axes[i], label=algo, element="step")

        axes[i].set_title(f'Distribution of {metric}')
        axes[i].set_xlabel(metric)
        axes[i].set_ylabel('Frequency')
        axes[i].legend(title='Algorithm', loc='upper left',
                       bbox_to_anchor=(1, 1))

        # Set y-axis to logarithmic scale
        axes[i].set_yscale('log')

    # Adjust layout and add padding
    plt.tight_layout(pad=4.0)
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.3)

    plt.show()


if __name__ == "__main__":
    main()
