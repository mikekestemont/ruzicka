"""
For a given data set, this script will compare and visualize
the distance scores between same-author and different-author
document pairs (SADPs and DADPs), for different vector space
and metric combinations. We compare `n` SADPs to an equal
numver randomly sampled DADPs. The resulting
distributions are statiscally compared using the
Kolmogorov-Smirnov statistic. Restricted to training data.
The visualisation uses seaborn's `kdeplot()`.
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sb
from utilities import load_pan_dataset
from vectorization import Vectorizer
from evaluation import pan_metrics
from distance_metrics import pairwise_minmax
from evaluation import SADPs_DADPs
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

# set hyperparameters:
corpus_dir = '../data/2014/gr_articles/'
mfi = sys.maxint
ngram_type = 'char'
ngram_size = 4
vsms = ('tf', 'tf_std', 'tf_idf', 'bin')
dms = ('minmax', 'euclidean', 'cityblock')


def SADPs_DADPs(dm, authors, trim_DADPs=True, random_state=1066):
    """
    Given the distance matrix `dm`, returns two tuples with the
    distances between same-author and different-author documents
    pairs. Scales both between the min and max of all distances.
    """
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

    DPs = np.asarray(list(SADPs)+list(DADPs))
    # scale the distances to 0-1:
    min_dist, max_dist = np.min(DPs), np.max(DPs)
    SADPs = (SADPs-min_dist) / (max_dist - min_dist)
    DADPs = (DADPs-min_dist) / (max_dist - min_dist)
    return SADPs, DADPs

# load the train data
data, _ = load_pan_dataset(corpus_dir+'train')
labels, documents = zip(*data)

# set visual parameters:
fig = sb.plt.figure(figsize=(len(vsms), len(dms)))
cnt = 0
outer_grid = gridspec.GridSpec(len(vsms), len(dms), wspace=0.1, hspace=0.1)
c1, c2 = sb.color_palette('Set1')[:2]

for vsm_cnt, vsm in enumerate(vsms):
    # vectorize:
    vectorizer = Vectorizer(mfi=mfi,
                            vector_space=vsm,
                            ngram_type=ngram_type,
                            ngram_size=ngram_size)
    X = vectorizer.fit_transform(documents).toarray()

    for dm_cnt, dm in enumerate(dms):
        if dm != 'minmax':
            distance_matrix = squareform(pdist(X, dm))
        else:
            distance_matrix = squareform(pdist(X, pairwise_minmax))

        # extract scores for same-author and different-author DPs:
        SADPs, DADPs = SADPs_DADPs(distance_matrix, labels,
                           trim_DADPs=True, random_state=1072015)

        # plot distributions using kdeplot:
        sb.set_style("dark")
        ax = sb.plt.Subplot(fig, outer_grid[cnt])
        ax.set_xlim([0, 1])
        sb.kdeplot(DADPs, shade=True, legend=False, c=c1, ax=ax, lw=0.5)
        sb.kdeplot(SADPs, shade=True, legend=False, c=c2, ax=ax, lw=0.5)

        # add high-level labels:
        if dm_cnt == 0:
            ax.set_ylabel(vsm.replace('_', '-'), fontsize=5)
        if vsm_cnt == 0:
            ax.set_title(dm, fontsize=5)

        ax.xaxis.set_major_formatter(sb.plt.NullFormatter())
        ax.yaxis.set_major_formatter(sb.plt.NullFormatter())

        # test for signifiance via Kolmogorov-Smirnov:
        D, p = ks_2samp(SADPs, DADPs)
        print("\t\t- KS: D = %s (p = %s)" %(D, p))

        # annotate plots:
        at = AnchoredText("KS: "+str(format(D, '.3f')), prop=dict(size=3), frameon=True, loc=2)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        sb.axes_style()
        fig.add_subplot(ax)
        cnt+=1

sb.plt.savefig('../output/distributions.pdf')

