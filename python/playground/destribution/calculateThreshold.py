import numpy as np
import matplotlib.pyplot as plt


def calculate_threshold(training_data):

    mean = np.mean(training_data)
    std_dev = np.std(training_data)
    max_value = np.max(training_data)
    min_value = np.min(training_data)
    # print("min: ", min_value, " max ", max_value,
    #      " mean: ", mean, " std_dev: ", std_dev)
    mean_to_greatest = max_value - mean
    mean_to_least = -min_value + mean
    most_distant_value = max(
        mean_to_greatest, mean_to_least)
    # print("mean_to_greatest: ", mean_to_greatest,
    #      " mean_to_least: ", mean_to_least)

    # print("most_distant_value: ", most_distant_value)
    threshold_coefficient = most_distant_value/std_dev

    threshold = threshold_coefficient*std_dev
    # print("threshold_coefficient:", threshold_coefficient)
    # print("most distand value should be: ",
    #      threshold_coefficient*std_dev, " away from mean")
    # plt.figure(figsize=(8, 6))
    # plt.hist(training_data, bins=600, density=True, alpha=0.7, color='blue')
    # plt.title('Histogram of Training Data Distribution')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.grid(True)

    marginal_point = min_value if most_distant_value == mean_to_least else max_value

    # plt.axvline(x=mean, color='red', linestyle='--', label='Mean')
    # plt.axvline(x=mean+most_distant_value, color='green', linestyle='--',
    #            label='upper Limit (threshold)')
    # plt.axvline(x=mean-most_distant_value, color='blue', linestyle='--',
    #            label='low Limit (threshold)')
    # plt.plot(marginal_point, 0, color='red', marker='o',
    #         markersize=10, label=f'Largest Value ({most_distant_value:.2f})')
    # plt.legend()

    # plt.show()

    # print("Found threshold:", threshold)

    calibration = [threshold_coefficient, std_dev, mean]
    return calibration
