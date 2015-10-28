from __future__ import print_function
from itertools import combinations
import seaborn as sb
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import recall_score, precision_score, f1_score

def accuracy(prediction_scores, ground_truth_scores):
    acc = 0.0
    for gt_score, pred_score in zip(ground_truth_scores, prediction_scores):
        if (pred_score >= 0.5) == (gt_score >= 0.5):
            acc += 1.0
    return acc / float(len(prediction_scores))

def auc(prediction_scores, ground_truth_scores):
    return roc_auc_score(ground_truth_scores, prediction_scores)

def c_at_1(prediction_scores, ground_truth_scores):
    n = float(len(prediction_scores))
    nc, nu = 0, 0
    for gt_score, pred_score in zip(ground_truth_scores, prediction_scores):
        if pred_score == 0.5:
            nu += 1
        elif (pred_score >= 0.5) == (gt_score >= 0.5):
            nc += 1.0
    return (nc + (nu * nc / n)) / n

def pan_metrics(prediction_scores, ground_truth_scores):
    return accuracy(prediction_scores, ground_truth_scores), \
           auc(prediction_scores, ground_truth_scores), \
           c_at_1(prediction_scores, ground_truth_scores)

def SADPs_DADPs(dm, authors, trim_DADPs=True, random_state=1066):
    SADPs, DADPs = [], []
    idxs = range(len(authors))
    for idx1, idx2 in combinations(idxs, 2):
        author1, author2 = authors[idx1], authors[idx2]
        if author1 == author2:
            SADPs.append(dm[idx1][idx2])
        else:
            DADPs.append(dm[idx1][idx2])
    if trim_DADPs:
        np.random.RandomState(random_state).shuffle(DADPs)
        DADPs = DADPs[:len(SADPs)]
    return np.asarray(SADPs), np.asarray(DADPs)

def evaluate(SADPs, DADPs, beta=2):
    y_true = np.asarray([0 for dp in DADPs]+[1 for dp in SADPs])
    scores = np.concatenate((DADPs, SADPs))
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    precisions, recalls = precisions[:-1], recalls[:-1] # cut off 1
    f_scores = beta * (precisions * recalls) / (precisions + recalls)
    
    # add prec rec curve of imposter approach for test results:
    sb.plt.clf()
    sb.plt.plot(recalls, precisions, label="dev")
    sb.plt.legend(loc="best")
    sb.plt.gca().set_aspect('equal', adjustable='box')
    sb.plt.savefig("../output/test_prec_rec.pdf")
    return f_scores, precisions, recalls, thresholds

def evaluate_with_threshold(SADPs, DADPs, t, beta=2):
    y_true = np.asarray([0 for dp in DADPs]+[1 for dp in SADPs])
    scores = np.concatenate((DADPs, SADPs))
    scores = 1 - scores # highest similarity
    preds = scores >= t 
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f_score = beta * (precision * recall) / (precision + recall)
    f_scores, precisions, recalls, thresholds = evaluate(SADPs, DADPs)
    sb.plt.clf()
    fig = sb.plt.figure()
    sb.set_style("darkgrid")
    c1, c2, c3, c4 = sb.color_palette("Set1")[:4]
    sb.plt.plot(thresholds, f_scores, label="F1 score", c=c1)
    sb.plt.plot(thresholds, precisions, label="Precision", c=c2)
    sb.plt.plot(thresholds, recalls, label="Recall", c=c3)
    #sb.plt.xlim(0, 1.005)
    #sb.plt.ylim(0.4, 1.005)
    ax = sb.plt.gca()
    an1 = ax.annotate("F-score: "+format(f_score*100, '.1f'),
                  xy=(t, 0.45), xycoords="data",
                  va="center", ha="left", fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.3", fc="w"))
    sb.plt.axvline(x=t, linewidth=1, c=c4)
    sb.plt.legend(loc="best")
    sb.plt.xlabel('Threshold', fontsize=10)
    sb.plt.ylabel('Score', fontsize=10)
    sb.plt.savefig("../output/thresholds.pdf")
    return f_score, precision, recall

def dm_dendrogram(dm, authors, titles, metric, filename='distributions.pdf'):
    sb.plt.clf()
    fig = sb.plt.figure()
    Z = linkage(dm, method='ward', metric='euclidean')
    sb.plt.title('Hierarchical Clustering Dendrogram')
    sb.plt.xlabel('sample index')
    sb.plt.ylabel('distance')
    ax = fig.add_subplot(111, axisbg='white')
    dendrogram(Z, labels=titles, leaf_font_size=6,
                             leaf_rotation=180)
    sb.plt.rcParams['font.family'] = 'arial'
    sb.plt.rcParams['font.size'] = 6
    sb.plt.rcParams['lines.linewidth'] = 0.55
    sb.plt.rcParams['lines.color'] = 'darkgrey'
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_rotation('vertical')
        label.set_fontname('Arial')
        label.set_fontsize(6)
    ax.get_yaxis().set_ticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    sb.plt.xticks(rotation=90)
    sb.plt.tick_params(axis='x', which='both', bottom='off', top='off')
    sb.plt.tick_params(axis='y', which='both', bottom='off', top='off')
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    sb.plt.rcParams["figure.facecolor"] = "white"
    sb.plt.rcParams["axes.facecolor"] = "white"
    sb.plt.rcParams["savefig.facecolor"] = "white"
    sb.plt.subplots_adjust(bottom=0.15)
    sb.plt.savefig('../output/'+filename)





