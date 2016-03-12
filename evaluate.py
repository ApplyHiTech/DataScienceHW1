import os
import time
import numpy as np
import matplotlib.pyplot as plt
import config

from sklearn.metrics import roc_curve, roc_auc_score

config.maybe_make_path(config.RESULTS_PATH)
PLOT_FILE_TYPE = "svg"


def save_plot(model_name):
    filename = "%s-%s.%s" % (int(time.time()), model_name, PLOT_FILE_TYPE)
    plt.savefig(os.path.join(config.RESULTS_PATH, filename))


def plot_roc(fpr, tpr, roc_auc, model_name=""):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic %s' % model_name)
    plt.legend(loc="lower right")

    save_plot(model_name)


def labels_array(df):
    return np.array(df.map(lambda r: r["label"]).collect())


def scores_array(df):
    return np.array(
        df.map(lambda r: r["probability"])
        .map(lambda r: r.values[1])
        .collect())


def roc_auc(labels, scores):
    return roc_auc_score(labels, scores)


def roc(labels, scores, positive_label=1.0):
    return roc_curve(labels, scores, pos_label=positive_label)
