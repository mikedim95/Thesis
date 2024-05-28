import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


def parse_metrics(file_path):
    metrics = {
        'AUC': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'AP': [],
        'ElapsedTime': []
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
                metrics['AUC'].append(float(match.group(1)))
                metrics['Precision'].append(float(match.group(2)))
                metrics['Recall'].append(float(match.group(3)))
                metrics['F1'].append(float(match.group(4)))
                metrics['AP'].append(float(match.group(5)))
                metrics['ElapsedTime'].append(float(match.group(6)))
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

    metrics_to_plot = ['AUC', 'Precision', 'Recall', 'F1', 'AP', 'ElapsedTime']
    num_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(num_metrics, 2, figsize=(16, 8 * num_metrics))

    for i, metric in enumerate(metrics_to_plot):
        # Donut chart
        total_values = {algo: sum(metrics[metric])
                        for algo, metrics in all_metrics.items()}
        labels = total_values.keys()
        sizes = total_values.values()
        explode = [0.1 if i == 0 else 0 for i in range(
            len(labels))]  # explode the first slice

        axes[i, 0].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                       shadow=True, startangle=140, wedgeprops=dict(width=0.3))
        axes[i, 0].set_title(f'Total {metric} for Each Algorithm')

        # Histogram with density plot for metric distribution
        for algo, metrics in all_metrics.items():
            sns.histplot(metrics[metric], kde=True,
                         ax=axes[i, 1], label=algo, element="step")

        axes[i, 1].set_title(f'Distribution of {metric}')
        axes[i, 1].set_xlabel(metric)
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 1].legend(title='Algorithm', loc='upper left',
                          bbox_to_anchor=(1, 1))

    # Adjust layout and add padding
    plt.tight_layout(pad=4.0)
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.3)

    plt.show()


if __name__ == "__main__":
    main()
