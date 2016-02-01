#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    """
    Builds a conventional horizontal dendrogram of the data
    represented in `X`, where `labels` is used to annotate
    the tree's leaves. We use scipy's default settings for
    running the clustering algorithm (e.g. Ward's linkage).

    Parameters
    ----------
    X : array-like [n_docs, n_other]
        The 2D-data which will be clustered

    labels : list of strs
        The labels associated with the 1st dim of the `X` tensor

    outputfile :  str, default='../output/tree.pdf'
        The path where the clustermap will be saved

    fontsize :  int, default=5
        The fontsize used to plot `labels`

    """

    fig = sns.plt.figure()

    # aesthetic interventions:
    ax = fig.add_subplot(111, axisbg='white')
    plt.rcParams['font.family'] = 'arial'
    plt.rcParams['font.size'] = 6
    plt.rcParams['lines.linewidth'] = 0.75

    # get a more pleasing color palette:
    set_link_color_palette([rgb2hex(rgb) for rgb in sns.color_palette("Set2", 10)])

    # run the clustering:
    linkage_obj = linkage(X, method='ward')

    # visualize the dendrogram
    d = dendrogram(Z=linkage_obj, 
                         labels=labels,
                         leaf_font_size=fontsize,
                         leaf_rotation=180,
                         above_threshold_color='#AAAAAA')

    # some more aesthetic interventions:
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
    """
    Builds a `seaborn` clustermap of the data represented in `X`.

    Parameters
    ----------
    X : array-like [n_docs, n_other]
        The 2D-data which will be plotted

    labels : list of strs
        The labels associated with the 1st dim of the `X` tensor

    outputfile :  str, default='../output/clustermap.pdf'
        The path where the clustermap will be saved

    fontsize :  int, default=5
        The fontsize used to plot `labels`

    Notes
    ----------
    See these websites for more info on `seaborn`'s clustermap:
      - http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.clustermap.html
      - https://stanford.edu/~mwaskom/software/seaborn/examples/structured_heatmap.html

    """

    # make sure that we have a clean plotting area:
    plt.clf()

    # convert X to a pandas dataframe:
    df = pd.DataFrame(data=X, columns=labels)

    # extract correlations for a scaled version of X:
    df = df.applymap(lambda x:int(x*100000)).corr()

    # clustermap plotting:
    cm = sns.clustermap(df)
    ax = cm.ax_heatmap

    # some aesthetic interventions:
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
