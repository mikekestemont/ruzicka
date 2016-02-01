# Ružička: Authorship Verification in Python

## Introduction
<img align="right" src="https://cloud.githubusercontent.com/assets/4376879/11402489/8703f80a-9398-11e5-8091-2b1ed5b2bb97.png" 
alt="IMAGE ALT TEXT HERE" height="240" border="10"/>
The code in this repository offers an implementation of a number of routines in authorship studies, with a focus on authorship verification. It is named after the inventor of the "minmax" measure (M. Ružička). The repository offers a generic implementation of two commonly used verification systems. The first system is an intrinsic verifier, depending on a first-order metric (O1), close to the one described in:

```
Potha, N. and E. Stamatatos. A Profile-based Method for Authorship Verification
In Proc. of the 8th Hellenic Conference on Artificial Intelligence
(SETN), LNCS, 8445, pp. 313-326, 2014.
```

The second system is an extrinsic verifier with second-order metrics (O2), based the General Imposters framework as described in:

```
M. Koppel and Y. Winter (2014), Determining if Two Documents are by the Same
Author, JASIST, 65(1): 178-187.
```

The package additionally offers a number of useful implementations of common vector space models and evaluation metrics. The code in this repository was used to produce the results in a paper which is currently under submission.


## Quickstart

<img align="right" src="https://cloud.githubusercontent.com/assets/4376879/11402488/87041952-9398-11e5-82f9-cf3abcbe5f53.png" 
alt="IMAGE ALT TEXT HERE" height="240" border="10" style="float: right;" />
While the code in this repository was tailored towards our needs for a specific paper, the `code` folder includes an IPython notebook, which will guide you through some of the main functionality offered. In the code itself, we try to offer comprehensive documentation in the form of docstrings. All experiments in our paper can be repeated using the following scripts under `code`:
+ 01pan_experiments.py
+ 02latin_dev_o1.py
+ 03latin_dev_o2.py
+ 04latin_test_o2.py
+ 05latin_testviz.py

## Data sets
This repository includes 6 multilingual benchmark datasets for authorship verification (under `data/`), which were used as the official competition data in the 2014 track on authorship verification of the annual [PAN evaluation lab](http://www.uni-weimar.de/medien/webis/events/pan-14/pan14-web/) on uncovering plagiarism, authorship, and social software misuse. The [survey paper](http://www.uni-weimar.de/medien/webis/events/pan-14/pan14-papers-final/pan14-authorship-verification/stamatatos14-overview.pdf) by Stamatatos et al. provides detailed information on the provenance, structure and nature of these corpora (together with baselines figures etc.). The competition data for this competition covered the following text varieties:
* Dutch essays
* Dutch reviews
* English essays
* English novels
* Spanish articles
* Greek articles

Additionally, this repository includes a novel benchmark dataset for Latin authors from Antiquity (under `data/latin/`), which were mainly selected from the [Latin Library](http://www.thelatinlibrary.com/). This data set has a similar structure as the PAN corpora. 

## Dependencies
This code requires Python 2.7+ (Python 3 has not been tested). The repository is dependent on a number of well-known third-party Python libraries, including:
+ numpy
+ scipy
+ scikit-learn
+ matplotlib
+ seaborn
+ numba

and preferably (for GPU acceleration and/or JIT-compilation):
+ theano
+ numbapro

We recommend installing Continuum's excellent [Anaconda Python framework](https://www.continuum.io/downloads), which comes bundled with most of these dependencies. Additionally, this code integrates a number of [scripts by Vincent van Asch](http://www.cnts.ua.ac.be/~vincent/software.html) to statistically compare the output of different classifiers, using Approximate Randomization Testing (under `ruzicka/`: `art.py`, `combinations.py` and `confusionmatrix.py`).



