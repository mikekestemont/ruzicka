from __future__ import print_function

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def pca(X, labels, ints, filename='../output/pca.pdf', feature_names=None):
    prin_comp = PCA(n_components=2)
    pca_matrix = prin_comp.fit_transform(X)
    pca_loadings = prin_comp.components_.transpose()
    sns.set_style('dark')
    sns.plt.rcParams['axes.linewidth'] = 0.4
    fig, ax1 = sns.plt.subplots()  
    x1, x2 = pca_matrix[:,0], pca_matrix[:,1]
    ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
    for x, y, label, l_int in zip(x1, x2, labels, ints):
        ax1.text(x, y, label, ha='center', va="center",
                     color=plt.cm.spectral(int(l_int) / 10.),
                     fontdict={'family': 'Arial', 'size': 10})
    if feature_names:
        ax2 = ax1.twinx().twiny()
        l1, l2 = pca_loadings[:,0], pca_loadings[:,1]
        ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');
        for x, y, l in zip(l1, l2, feature_names):
            ax2.text(x, y, l ,ha='center', va="center", size=8, color="darkgrey",
                fontdict={'family': 'Arial', 'size': 9})
        ax2.set_xticklabels([])
        ax2.set_xticks([])
        ax2.set_yticklabels([])
        ax2.set_yticks([])
    # control aesthetics:
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    sns.plt.savefig(filename, bbox_inches=0)

def tree(X, labels, outputfile='../output/tree.pdf', fontsize=5):
    fig = sns.plt.figure()
    ax = fig.add_subplot(111, axisbg='white')
    plt.rcParams['font.family'] = 'arial'
    plt.rcParams['font.size'] = 6
    plt.rcParams['lines.linewidth'] = 0.75
    from distance_metrics import pairwise_minmax
    linkage_obj = linkage(X, method='ward')
    d = dendrogram(Z=linkage_obj, labels=labels,
                         leaf_font_size=fontsize,
                         leaf_rotation=180)
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
    #cmap = sns.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
    #cm = sns.clustermap(df, cmap=cmap)
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

def tsne(X, labels, ints, filename='../output/tsne.pdf', fontsize=5):
    tsne = TSNE(n_components=2)
    tsne_matrix = tsne.fit_transform(X) # unsparsify
    sns.set_style('dark')
    sns.plt.rcParams['axes.linewidth'] = 0.4
    fig, ax1 = sns.plt.subplots()  
    x1, x2 = tsne_matrix[:,0], tsne_matrix[:,1]
    ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
    for x, y, label, l_int in zip(x1, x2, labels, ints):
        ax1.text(x, y, label, ha='center', va="center",
                     color=plt.cm.spectral(int(l_int) / 10.),
                     fontdict={'family': 'Arial', 'size': 10})
    # control aesthetics:
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    sns.plt.savefig(filename, bbox_inches=0)