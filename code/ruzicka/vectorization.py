#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys

import scipy.sparse as sp
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, Normalizer


def identity(y):
    """
    Simple identity function.
    """
    return y.split()


class StdDevScaler(BaseEstimator):
    """
    Scales a matrix through dividing cells
    by the column-wise standard deviations.
    This scaling method will give more weight
    to features with a lower standard deviation.

    The 'std' vector space model is particularly useful
    to calculate Burrows's Delta in the simplified
    formulation proposed in: S. Argamon, 'Interpreting
    Burrows's Delta: Geometric and Probabilistic
    Foundations', LLC 23:3 (2008).

    """

    def fit(self, X, y=None):
        """
        Compute the column-wise standard deviation of `X`
        to be used for later scaling.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the column-wise
            standard deviation.

        """
        self.weights_ = StandardScaler(with_mean=False).fit(X).std_
        return self

    def transform(self, X):
        """
        Scale a matrix according to the previously
        computed std weights, using a sparse implementation.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
            Supports sparse input.

        Attributes
        ----------
        weight_ : ndarray, shape (n_features,)
            Column-wise standard deviations.

        Returns
        ----------
        X = array-like, shape [n_samples, n_features]
            The scaled input data in sparse format.

        """
        if not sp.isspmatrix_csr(X):  # convert to sparse format if needed:
            X = sp.csr_matrix(X, dtype=np.float64)
        for i in range(X.shape[0]):
            start, end = X.indptr[i], X.indptr[i + 1]
            X.data[start:end] /= self.weights_[X.indices[start:end]]
        return X

    def fit_transform(self, X, y=None):
        """
        Fit `StdDevScaler` to X, then transform X.
        Equivalent to self.fit(X).transform(X),
        but more convenient.

        """
        self.fit(X)
        return self.transform(X)


class Vectorizer:
    """
    Vectorize texts into a sparse, two-dimensional
    matrix.

    """

    def __init__(
        self,
        mfi=100,
        ngram_type="word",
        ngram_size=1,
        vocabulary=None,
        vector_space="tf",
        lowercase=True,
        min_df=0.0,
        max_df=1.0,
        ignore=[],
    ):
        """
        Initialize the vectorizer by setting up a
        vectorization pipeline via sklearn as
        `self.transformer`

        Parameters
        ----------
        mfi: int, default=100
            The nb of most frequent items (words or
            ngrams) to extract.
        ngram_type: str, default='word'
            Set the type of features to be used
            for the ngram extraction:
            - 'word': individual tokens
            - 'char': character ngrams
            - 'char_wb': character ngrams (but not
                across word boundaries)
        ngram_size: int, default=1
            The length of the ngrams to be extracted.
        vocabulary: list, default=None
            Vectorize using an existing vocabulary.
        vector_space: str, default: 'tf'
            Which vector space to use (see below).
            Must be one of: 'tf', 'tf_scaled',
            'tf_std', 'tf_idf', 'bin'.
        lowercase: boolean, default=True
            Whether or not to lowercase the input texts.
        min_df: float, default=0.0
            Proportion of documents in which a feature
            should minimally occur.
            Useful to ignore low-frequency features.
        max_df: float, default=0.0
            Proportion of documents in which a feature
            should maximally occur.
            Useful for 'culling' and ignoring features
            which don't appear in enough texts.
        ignore: list(str), default=[]
            List of features to be ignored.
            Useful to manually remove e.g. stopwords or other
            unwanted items.

        Notes
        -----------
        The following vector space models are supported:
        - 'tf': simple relative term frequency model
        - 'tf_scaled': tf-model, but normalized using a MinMaxScaler
        - 'tf_std': tf-model, but normalized using a StdDevScaler
        - 'tf_idf': traditional tf-idf model
        - 'bin': binary model, only captures presence of features
        """

        if vector_space not in ("tf", "tf_scaled", "tf_std", "tf_idf", "bin"):
            raise ValueError("Unsupported vector space model: %s" % (vector_space))

        self.params = {
            "max_features": mfi,
            "max_df": max_df,
            "min_df": min_df,
            "preprocessor": None,
            "ngram_range": (ngram_size, ngram_size),
            "lowercase": False,
            "decode_error": "ignore",
            "stop_words": ignore,
        }

        if ngram_type == "word":
            self.params["tokenizer"] = identity
        elif ngram_type in ("char", "char_wb"):
            self.params["analyzer"] = ngram_type

        n = Normalizer(norm="l2", copy=False)

        if vector_space == "tf":
            self.params["use_idf"] = False
            v = TfidfVectorizer(**self.params)
            self.transformer = Pipeline([("s1", v), ("s2", n)])

        elif vector_space == "tf_std":
            self.params["use_idf"] = False
            v = TfidfVectorizer(**self.params)
            scaler = StdDevScaler()
            self.transformer = Pipeline([("s1", v), ("s2", scaler), ("s3", n)])

        elif vector_space == "tf_idf":
            self.params["use_idf"] = True
            v = TfidfVectorizer(**self.params)
            self.transformer = Pipeline([("s1", v), ("s2", n)])

        elif vector_space == "bin":
            self.params["binary"] = True
            v = CountVectorizer(**self.params)
            self.transformer = Pipeline([("s1", v), ("s2", n)])

    def fit(self, texts):
        """
        Vectorize input texts and store them in
        sparse format as `self.X`.

        Parameters
        ----------
        texts: 2D-list of strings
            The texts to be vectorized.
            Assumed untokenized input in the case of
            `ngram_type`='word', else expects
            continguous strings.

        Returns
        ----------
        X: array-like, [n_texts, n_features]
            Vectorized texts in sparse format.
        """
        self.transformer.fit(texts)
        self.feature_names = self.transformer.named_steps["s1"].get_feature_names_out()
        return self

    def transform(self, texts):
        return self.transformer.transform(texts)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
