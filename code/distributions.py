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


corpus_dir = '../data/2014/gr_articles/'
mfi = 10000
ngram_type = 'char'
ngram_size = 4

data, _ = load_pan_dataset(corpus_dir+'train')
labels, documents = zip(*data)

vsms = ('tf', 'tf_std', 'tf_idf', 'bin')
dms = ('minmax', 'euclidean', 'cityblock', 'cosine')
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
        SADPs, DADPs = SADPs_DADPs(distance_matrix, labels,
                           trim_DADPs=True, random_state=1072015)
        DPs = np.asarray(list(SADPs)+list(DADPs))
        # small the distances to 0-1:
        min_dist, max_dist = np.min(DPs), np.max(DPs)
        SADPs = (SADPs-min_dist) / (max_dist - min_dist)
        DADPs = (DADPs-min_dist) / (max_dist - min_dist)

        D, p = ks_2samp(SADPs, DADPs)
        print("\t\t- KS: D = %s (p = %s)" %(D, p))
        sb.set_style("dark")
        ax = sb.plt.Subplot(fig, outer_grid[cnt])
        ax.set_xlim([0, 1])
        sb.kdeplot(DADPs, shade=True, legend=False, c=c1, ax=ax, lw=0.5)
        sb.kdeplot(SADPs, shade=True, legend=False, c=c2, ax=ax, lw=0.5)
        if dm_cnt == 0:
            ax.set_ylabel(vsm.replace('_', '-'), fontsize=5)
        if vsm_cnt == 0:
            ax.set_title(dm, fontsize=5)

        ax.xaxis.set_major_formatter(sb.plt.NullFormatter())
        ax.yaxis.set_major_formatter(sb.plt.NullFormatter())
        at = AnchoredText("KS: "+str(format(D, '.3f')), prop=dict(size=3), frameon=True, loc=2)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        sb.axes_style()
        fig.add_subplot(ax)
        cnt+=1

sb.plt.savefig('../output/distributions.pdf')

