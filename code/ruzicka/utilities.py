#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Offers various utility functions, especially in the context of
manipulating authorship datasets in the PAN framework.
"""

import codecs
import glob
import os
import sys
from collections import Counter

import numpy as np
from sklearn.cross_validation import train_test_split

from vectorization import Vectorizer

def load_pan_dataset(directory, ext='txt', encoding='utf8'):
    """
    Loads the data from `directory`, which should hold subdirs
    for each "problem"/author in a dataset. As with the official
    PAN datasets, all `unknown` instances for authors are included
    as test data; the rest is used as training data.

    Parameters
    ----------
    directory: str, default=None
        Path the directory from which the `problems` are loaded.
    ext: str, default='txt'
        Only loads files with this extension,
        useful to filter out e.g. OS-files.
    encoding: str, default='utf8'
        Sets the encoding of the files, passed to `codecs.open()`

    Returns
    ----------
    train_data:
        list of (author, document) tuples (training/development)
    test_data:
        list of (author, document) tuples (test problems)

    Notes
    ----------
    - See this webpage for more information on the data structure:
      http://www.uni-weimar.de/medien/webis/events/pan-15/pan15-web/author-identification.html
    - The author labels in the `test_data` returned are NOT NECESSARILY
      the actual, correct authors of the `unknown.txt`, but only the
      authorship against which the authorship of the associated test 
      document should be verified.

    """

    train_data, test_data = [], []

    for author in sorted(os.listdir(directory)):
        path = os.sep.join((directory, author))
        if os.path.isdir(path):
            for filepath in sorted(glob.glob(path+'/*.'+ext)):
                text = codecs.open(filepath, mode='r').read()
                name = os.path.splitext(os.path.basename(filepath))[0]
                if name == 'unknown':
                    test_data.append((author, text))
                else:
                    train_data.append((author, text))
    return train_data, test_data


def get_vocab_size(corpus_dir,
                  ngram_type,
                  ngram_size,
                  min_df=0.0,
                  phase='train'):
    """
    Convenience function: fits a vectorizer with the specified
    parameters on the data under `corpus_dir/phase`, i.e. excluding
    the 'unknown' texts. Returns the maximem number 
    of features available, which is useful to test different
    settings for the vocabulary truncation.

    Parameters
    ----------
    corpus_dir : str, default=None
        The main corpus directory (e.g. `du_essays`)

    ngram_type : str, default=None
        See the docs in `vectorization.py`

    ngram_size : int, default=None
        See the docs in `vectorization.py`

    ngram_size : int or float, default=0.0
        See the docs in `vectorization.py`

    phase : str, default='train'
        Should be 'train' or 'test': specifies whether
        to get the max vocabulary size from the train
        or test problems under `corpus_dir`.

    Returns
    ----------
    max_vocab_size = int
        The maximum number of vocabulary items available.

    """

    # preprocess:
    train_data, _ = load_pan_dataset(corpus_dir+'/'+phase)
    train_labels, train_documents = zip(*train_data)
    
    # vectorize with maximum nb of features:
    vectorizer = Vectorizer(mfi = sys.maxint,
                            ngram_type = ngram_type,
                            ngram_size = ngram_size,
                            min_df = min_df)
    vectorizer.fit(train_documents)

    # returns max nb of features:
    return len(vectorizer.feature_names)


def load_ground_truth(filepath, labels):
    """
    Loads the ground truth labels for a given dataset
    (typically `truth.txt`).
      - Needs `labels` as a parameter, to ensure that
        the problem labels are returned in the correct
        order.
      - Expects the file at `filepath` to use the PAN
        2014 syntax.

    Parameters
    ----------
    filepath : str, default=None
        The path to the ground truth file

    labels : list of strs, default=None
        The author labels of problems in a particular order

    Returns
    ----------
    ground_truth_scores = list of floats
        The ground truth scores (0.0 or 1.0) for the test
        problems, returned in the order specified by `labels`.

    Notes
    ----------
    - See this webpage for more information on the 2014 data structure:
      http://www.uni-weimar.de/medien/webis/events/pan-14/pan14-web/author-identification.html

    """

    ground_truth = {}

    for line in open(filepath).readlines():
        problem_id, outcome = line.strip().split()
        if outcome == 'Y':
            outcome = 1.0
        elif outcome == 'N':
            outcome = 0.0
        ground_truth[problem_id] = outcome

    return [ground_truth[l] for l in labels]


def train_dev_split(train_X, train_y, random_state=1027):
    """
    Creates a random 50-50 split of the problems represented by
    `train_X`and `train_y`, into an artificially created set of
    development problems and test problems. The number of same-
    author and different-author problems returned is balanced:
    for each true document-author pair, we also include a false
    document-author, whereby we randomly assign a different author
    to the test document in question. There is no document-level
    overlap between the development and test problems created.

    Parameters
    ----------
    train_X : array-like [nb_problems, nb_features], default=None
        The 2D matrix representing a set of documents.

    train_y : list of ints, default=None
        A list of int-encoded author labels (has to be the correct
        author for each document).
    
    random_state : int, default=1027
        Used a seed for the random splitting.


    Returns
    ----------
    Returns in the following order:

    X_dev : array-like [nb_problems, nb_features]
        The 2D matrix representing a set of development documents.

    y_dev : list of ints, default=None
        A list of int-encoded author labels (actual, correct
        author for each document).
    
    X_test : array-like [nb_problems, nb_features]
        The 2D matrix representing a set of test problems.

    y_test : list of ints, default=None
        A list of int-encoded author labels: not necessarily the
        correct author, but rather representing a concrete
        verification problem.

    test_gt_scores : list of floats
        A list of ground truth scores (0.0 or 1.0) for the test
        problems returned (X_test, y_test). Each test_gt_scores[i]
        indicates whether (1.0) or not (0.0) whether y_test[i] is
        the author of X_test[i].

    """

    # split the original data:
    X_dev, X_test, y_dev, y_test = train_test_split(train_X, train_y,
                                        test_size=.5,
                                        random_state=random_state,
                                        stratify=train_y)
    test_gt_scores = []

    # randomly select 1/2 of the idxs:
    np.random.seed(random_state)
    author_options = set(train_y)
    rnd_idxs = np.random.choice(len(y_test), int(len(y_test)/2))

    for idx, y in enumerate(y_test):
        if idx in rnd_idxs:
            # pick random other author to create a diff-author pair:
            real_author = y_test[idx]
            other_authors = [a for a in author_options if a != real_author]
            fake_author = np.random.choice(other_authors, 1)[0]
            y_test[idx] = fake_author
            # indicate it's a negative example:
            test_gt_scores.append(0.0)
        else:
            # indicate it's a positive example:
            test_gt_scores.append(1.0)

    return X_dev, y_dev,\
           X_test, y_test, test_gt_scores


def binarize(scores):
    """

    Takes a list of scores and binarizes it into strings.
    This is useful for running the ART scripts for testing
    the statistical difference between different outputs.
    0.5 is used a the threshold for the separation of 
    attribution (< 0.5 : 'N') and non-attribution cases
    (> 0.5 : 'Y'). All values equal 0.5 are assigned a
    separate 'X' label.

    Parameters
    ----------
    scores : list of floats
        The original scores (between 0.0 and 1.0 to be binarized)

    Returns
    ----------

    binarized_scores : list of strs
        A list of class labels ('Y', 'N' or 'X')

    """

    scs = []
    for sc in scores:
        if sc == 0.5:
            scs.append('X')
        elif sc < 0.5:
            scs.append('N')
        elif sc > 0.5:
            scs.append('Y')
    return scs

                         
    