from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np

from .metrics import metricor
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# determine sliding window (period) based on ACF


def find_length(data):
    if len(data.shape) > 1:
        return 0
    data = data[:min(20000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]

    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125


def plotFig(data, label, score, slidingWindow, fileName, modelName, plotRange=None):
    grader = metricor()

    R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(
        labels=label, score=score, window=slidingWindow, plot_ROC=True)

    if np.sum(label) == 0:
        print('No anomalies in the labels. Generating figures with zero metrics.')
        fig3 = plt.figure(figsize=(12, 10), constrained_layout=True)
        gs = fig3.add_gridspec(3, 4)

        f3_ax1 = fig3.add_subplot(gs[0, :-1])
        plt.tick_params(labelbottom=False)
        plt.plot(data, 'k')
        plt.xlim([0, len(data)])

        f3_ax2 = fig3.add_subplot(gs[1, :-1])
        plt.plot(score)
        plt.hlines(np.mean(score) + 3 * np.std(score), 0,
                   len(data), linestyles='--', color='red')
        plt.ylabel('score')
        plt.xlim([0, len(data)])

        f3_ax4 = fig3.add_subplot(gs[0, -1])
        plt.xlabel('FPR')
        plt.ylabel('TPR')

        plot_file = f"{fileName}_{modelName}.png"
        plt.savefig(plot_file)
        plt.close(fig3)

        return 0.0, R_AUC, 0.0, 0.0, 0.0, 0.0, 0.0, R_AP, R_AP, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0

    L, fpr, tpr = grader.metric_new(label, score, plot_ROC=True)
    precision, recall, AP = grader.metric_PR(label, score)

    range_anomaly = grader.range_convers_new(label)
    max_length = len(score)

    if plotRange is None:
        plotRange = [0, max_length]

    fig3 = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig3.add_gridspec(3, 4)

    f3_ax1 = fig3.add_subplot(gs[0, :-1])
    plt.tick_params(labelbottom=False)
    plt.plot(data[:max_length], 'k')
    for r in range_anomaly:
        if r[0] == r[1]:
            plt.plot(r[0], data[r[0]], 'r.')
        else:
            plt.plot(range(r[0], r[1] + 1), data[range(r[0], r[1] + 1)], 'r')
    plt.xlim(plotRange)

    f3_ax2 = fig3.add_subplot(gs[1, :-1])
    L1 = ['%.2f' % elem for elem in L]
    plt.plot(score[:max_length])
    plt.hlines(np.mean(score) + 3 * np.std(score), 0,
               max_length, linestyles='--', color='red')
    plt.ylabel('score')
    plt.xlim(plotRange)

    f3_ax3 = fig3.add_subplot(gs[2, :-1])
    index = (label + 2 * (score > (np.mean(score) + 3 * np.std(score))))
    def cf(x): return 'k' if x == 0 else (
        'r' if x == 1 else ('g' if x == 2 else 'b'))
    cf = np.vectorize(cf)
    color = cf(index[:max_length])

    tn_count = np.sum(index[:max_length] == 0)
    fn_count = np.sum(index[:max_length] == 1)
    fp_count = np.sum(index[:max_length] == 2)
    tp_count = np.sum(index[:max_length] == 3)

    plt.scatter(np.arange(max_length), data[:max_length], c=color, marker='.')
    plt.xlim(plotRange)

    f3_ax4 = fig3.add_subplot(gs[0, -1])
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    plot_file = f"{fileName}_{modelName}.png"
    plt.savefig(plot_file)
    plt.close(fig3)

    return (L1[0], str(round(R_AUC, 2)), L1[1], L1[2], L1[3], L1[5], L1[6], str(round(AP, 2)), str(round(R_AP, 2)), L1[9], L1[7], L1[4], L1[8], tn_count, fn_count, fp_count, tp_count)


def printResult(data, label, score, slidingWindow, fileName, modelName):
    grader = metricor()
    R_AUC = grader.RangeAUC(labels=label, score=score,
                            window=slidingWindow, plot_ROC=False)
    L = grader.metric_new(label, score, plot_ROC=False)
    L.append(R_AUC)
    return L
