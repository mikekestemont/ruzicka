from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from distance_metrics import pairwise_minmax

def tree(X, labels, outputfile='../output/tree.pdf', fontsize=5):
    fig = sns.plt.figure()
    ax = fig.add_subplot(111, axisbg='white')
    plt.rcParams['font.family'] = 'arial'
    plt.rcParams['font.size'] = 6
    plt.rcParams['lines.linewidth'] = 0.75
    set_link_color_palette([rgb2hex(rgb) for rgb in sns.color_palette("Set2", 10)])
    linkage_obj = linkage(X, method='ward')
    d = dendrogram(Z=linkage_obj, 
                         labels=labels,
                         leaf_font_size=fontsize,
                         leaf_rotation=180,
                         above_threshold_color='#AAAAAA')
    ax = sns.plt.gca()
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_rotation('vertical')
        label.set_fontname('Arial')
        label.set_fontsize(fontsize)
    ax.get_yaxis().set_ticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    sns.plt.xticks(rotation=90)
    sns.plt.tick_params(axis='x', which='both', bottom='off', top='off')
    sns.plt.tick_params(axis='y', which='both', bottom='off', top='off')
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    sns.plt.rcParams["figure.facecolor"] = "white"
    sns.plt.rcParams["axes.facecolor"] = "white"
    sns.plt.rcParams["savefig.facecolor"] = "white"
    sns.plt.subplots_adjust(bottom=0.15)
    fig.savefig(outputfile)

def clustermap(X, labels, outputfile='../output/clustermap.pdf', fontsize=5):
    plt.clf()
    # convert to pandas dataframe:
    df = pd.DataFrame(data=X, columns=labels)
    df = df.applymap(lambda x:int(x*10000)).corr()

    # clustermap plotting:
    cm = sns.clustermap(df)
    ax = cm.ax_heatmap

    # xlabels:
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_rotation('vertical')
        label.set_fontname('Arial')
        label.set_fontsize(fontsize)

    # ylabels:
    for idx, label in enumerate(ax.get_yticklabels()):
        label.set_rotation('horizontal')
        label.set_fontname('Arial')
        label.set_fontsize(fontsize)

    cm.savefig(outputfile)
