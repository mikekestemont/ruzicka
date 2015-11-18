#!/usr/bin/env python
# -*- coding: utf8 -*-
r'''A script to compute evaluation statistics from files

Example usage inside a script:
    
The Timbl commando "Timbl -f dimin.train -t dimin.test +v cs+cm" gave an outputfile called:
dimin.test.IB1.O.gr.k1.out
    >>> cm = getConfusionMatrixFromFile('dimin.test.IB1.O.gr.k1.out', fsep=',')
    >>> print cm
                                                                predicted            
                           E          J          K          P          T    |     SUM
                --------------------------------------------------------------------------
    g         E |         87          4          8          1          0    |     100
    o         J |          5        346          0          0          1    |     352
    l         K |          7          0          9          0          0    |      16
    d         P |          3          0          0         24          0    |      27
              T |          0          2          0          0        453    |     455
          --------------------------------------------------------------------------------
            SUM |        102        352         17         25        454    |     950

    >>> print cm.averaged(level=MACRO, score=FSCORE)
    0.861914364513

Or, per instance:
   >>> data = [ (0,1), (1,1), (1,0)]
   >>> cm = Confusionmatrix() 
   >>> for gold, pred in data: cm.single_add(gold, pred)

Or, for multi-label:
   >>> data = [ ([0],[1]), ([1,2],[1]), ([1],[0,2])]
   >>> cm = Confusionmatrix() 
   >>> for gold, pred in data: cm.add(gold, pred)
   >>> print cm
                                         predicted            
                           0          1          2    |     SUM
                ----------------------------------------------------
    g         0 |          0          1          0    |       1
    o         1 |      1\0.5          1      1\0.5    |       2
    l         2 |          0        0\1          0    |       1
          ----------------------------------------------------------
            SUM |          1          2          1    |     0\4

When using for multi-label you may be interested in the influence of the compute_none argument when contructing a cm.

For more info:
    >>> help(cm)

Based on:
- Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze, Introduction to Information Retrieval, Cambridge University Press. 2008. http://nlp.stanford.edu/IR-book/
- Grigorios Tsoumakas, Ioannis Katakis, and Ioannis Vlahavas, Mining Multi-label Data, Data Mining and Knowledge Discovery Handbook, Part 6, O. Maimon, L. Rokach (Ed.), Springer, 2nd edition, pp. 667-685, 2010.
- Daelemans, W., Zavrel, J., van der Sloot, K., & van den Bosch, A. (2010). Timbl: Tilburg memory-based learner, version 6.3. Tech. Rep. ILK 10-01, Tilburg University
- van Rijsbergen, C. J. (1975). Information Retrieval . London, UK: Butterworths.

For more informations about the computation of the averaged scores:
http://www.clips.ua.ac.be/~vincent/pdf/microaverage.pdf


# License: GNU General Public License, see http://www.clips.ua.ac.be/~vincent/scripts/LICENSE.txt
'''
__date__ = 'September 2013'
__version__ = '2.4.0'
__author__ = 'Vincent Van Asch'


import os, sys, getopt, re, copy
from math import pow

# CONSTANTS ##############################################################################

MICRO=0
MACRO=1
MICROt=2
UTT = re.compile('<utt>')
encoding = 'utf-8'

FSCORE='fscore'
PRECISION='precision'
RECALL='recall'
AUC='auc'

COL='column'
ROW= 'row'
COUNT='count'
TP = 'true positives'
FP = 'false positives'
FN = 'false negatives'
TN = 'true negatives'

# CODE ###################################################################################

class LightError(AttributeError): pass

def kwad(x): return pow(x, 2)

def precision(tp, fp, tn, fn, beta=None):
    if tp == 0: return 0
    return float(tp)/(tp + fp)
def recall(tp, fp, tn, fn, beta=None):
    if tp == 0: return 0
    return float(tp)/(tp + fn)
def fpr(tp, fp, tn, fn, beta=None):
    if fp ==0: return 0
    return float(fp)/(fp + tn)
    
def fscore(tp, fp, tn, fn, beta=1):
    if not isinstance(beta, (int, float)): raise TypeError('beta must be a float or integer')
    p = precision(tp, fp, tn, fn)
    r = recall(tp, fp, tn, fn)
    beta = float(beta)
    if p == 0 or r == 0: return 0
    return ((1.0+kwad(beta))*p*r)/(kwad(beta)*p+r) 
    
def auc(tp, fp, tn, fn, beta=1):
    t = recall(tp, fp, tn, fn)
    f = fpr(tp, fp, tn, fn)
    return 0.5*(t - f + 1)


class Count(dict):
    def __init__(self, row=0, col=0):
        dict.__init__(self, {COL:col, ROW:row})
    @property
    def col(self):
        v = self[COL]
        if v%1:
            return v
        return int(v)
        
    @property
    def row(self):
        v = self[ROW]
        if v%1:
            return v
        return int(v)
        
    def __str__(self):
        if self.row == self.col: 
            return str(self.row)
        else:
            r = '%d' %self.row
            if isinstance(self.row, float): r = '%.1f' %self.row
            c = '%d' %self.col
            if isinstance(self.col, float): c = '%.1f' %self.col
                
            return '%s\\%s' %(c, r)
        
    def __repr__(self):
        return str(self)
        
    def __add__(self, x):
        '''Add x to row AND col'''
        if not isinstance(x, (float, int)): raise TypeError('Can only add float or int')
        self[COL]+=x
        self[ROW]+=x
        
        return Count(row=self.row, col=self.col)
        

class ConfusionMatrix(dict):
    '''A confusion matrix that can handle multilabels'''
    def __init__(self, freq={}, encoding=encoding, compute_none=False, strict=False, cached=True, light=False):
        '''freq        : the counts of the labels in the training corpus as a dictionary
                             key: label
                             value: count
           encoding    : encoding to print the labels. Note that the input labels should
                         be unicode.
           compute_none: if True, the None label is taken into account for the label frequences.
                         This has an influence on the averaged scores.
           strict      : if True, only use labels from the gold-standard (and not from predicted)
           cached      : if set to True, scores are cached. This means that after computing a score, using the add(),
                         single_add(), or add_training() method may introduce errors.
          light        : if True, trade in functionality for speed. Unavailable: true negatives, printing of confusion matrix, ...
                         Unavailable methods raise a LightError. Normally, works twice as fast.
        '''
        dict.__init__(self, {})
        
        # To store information about the training corpus
        self._freq=copy.deepcopy(freq)
        
        # For the accuracy
        self.total=0
        self.correct=0
        
        self.encoding = encoding
        self.compute_none=compute_none
        self.strict=strict
        self.light=light
        
        self._ng=[]
        self._np=[]
        self._instancecache=[]
        
        self.cached={}
        if cached:  self.cached={1:1}
        
        # Set up adding methods
        if self.light:
            self.add = self._light_add
            self.single_add = self._light_single_add
        else:
            self.add = self._full_add
            self.single_add = self._full_single_add
        
    def __str__(self):
        if self.light: raise LightError('Attribute unavailable: reason: ConfusionMatrix in light mode')
        labelpool = list(set(self.freq_training.keys() + self._np + self._ng))
        labelpool.sort()
        if None in labelpool:
            labelpool.remove(None)
            labelpool.append(None)
        N = len(labelpool)
        
        # A None safe version of the labels
        labels=[]
        for l in labelpool:
            if l is None:
                labels.append('*')
            else:
                labels.append(l)
        labels.append('SUM')
        N+=1

        # The width of the columns
        s = max([max([0]+[len(l) for l in labels if isinstance(l, (str, unicode))]+[len(str(x)) for x in self.columntotals.values()]), 7])
        format = '    '.join(['%%%ds' %(s+2)]+['%%%ds' %(s) for i in range(N-1)]+['| %%%ds' %(s)])
        format = '%s   '+format
        # The first lines
        lines=[(format %tuple(['' for i in range(N)]+['predicted', ' '])).replace('|', '')]
        lines.append(format %tuple([' ', ' ']+labels))
        lines.append('            '+'-'*(len(lines[-1])-7))
        sumline = '      '+'-'*(len(lines[-1])-6)
        
        if len(labels) > 13:
            caption=list('gold-standard')
        else:
            caption = list('gold')
        
        # Adding the data
        for glabel in labelpool:
            if glabel is None:
                line=['* |']
            else:
                line = [unicode(glabel)+' |']
            for plabel in labelpool:
                if glabel in self.keys():
                    if plabel in self[glabel].keys():
                        v = str(self[glabel][plabel])
                    else:
                        v = '0'
                else:
                    v='0'

                line.append(v)
                      
            try:
                letter = caption.pop(0)
            except IndexError:
                letter=' '
            
            # The row totals
            format2 = '%%%d.0f' %(len(str(sum(self.columntotals.values()))))
            
            rowstring = format2 %self.rowtotals.get(glabel, 0)
            line.append(rowstring)
                            
            lines.append(format %tuple([letter]+line))
            
        # The column totals
        lines.append(sumline)
        coltots=[]
        total_col = 0
        for l in labelpool:
            v = '%.0f' %self.columntotals.get(l, 0)
            coltots.append(v)
            total_col = total_col + self.columntotals.get(l, 0)

        total_row = 0
        for l in labelpool:
            total_row = total_row + self.tp(l) + self.fn(l)
    
        if total_row == total_col:
            total =  str(total_row)
        else:
            total = '%.0f\\%s' %(total_col, format2 %total_row)
        
        lines.append(format %(tuple([' ', 'SUM |'] +coltots+[total])))
            
        return ('\n'.join(lines)).encode(self.encoding)
    
    def _full_single_add(self, g, p):
        '''Takes two labels: a gold and a predicted label and
        adds them to the confusionmatrix.
        
        Example:
            self.single_add('a', 'b')
        
        Each instance in the test corpus should be added with this method.
        Cannot handle multi-labels or None labels. See add().
        
        You can use add() with single labels. But single_add is faster, so
        if multi-label functionality is not needed, use single_add().
        '''
        if isinstance(g, str): g = g.decode(self.encoding)
        if isinstance(p, str): p = p.decode(self.encoding)
        
        if g == p: self.correct+=1
        self.total+=1
        
        # Fill
        try:
            self[g][p] += 1
        except KeyError:
            try:
                d = self[g]
            except KeyError:
                self[g] = {p:Count(1,1)}
            else:
                self[g][p] = Count(1,1)
         
        self._ng.append(g)
        self._np.append(p)
        self._instancecache.append(set([g,p]))
        
        
    def _light_single_add(self, g, p):
        '''Takes two labels: a gold and a predicted label and
        adds them to the confusionmatrix.
        
        Example:
            self.single_add('a', 'b')
        
        Each instance in the test corpus should be added with this method.
        Cannot handle multi-labels or None labels. See add().
        
        You can use add() with single labels. But single_add is faster, so
        if multi-label functionality is not needed, use single_add().
        '''
        if isinstance(g, str): g = g.decode(self.encoding)
        if isinstance(p, str): p = p.decode(self.encoding)
        
        if g == p: self.correct+=1
        self.total+=1
        
        try:
            self.cached['observed_labels']
        except KeyError:
            self.cached['observed_labels'] = []
            self.cached['predlabels'] = []
            self.cached['goldlabels'] = []
        
        # Fill
        if g == p:
            cache_key = u'tp_'+unicode(p)
            self.cached[cache_key] = self.cached.get(cache_key, 0) + 1
        else:
            cache_key = u'fp_'+unicode(p)
            self.cached[cache_key] = self.cached.get(cache_key, 0) + 1
            cache_key = u'fn_'+unicode(g)
            self.cached[cache_key] = self.cached.get(cache_key, 0) + 1
         
        if p not in self.cached['observed_labels']:
            self.cached['observed_labels'].append(p)
        if g not in self.cached['observed_labels']:
            self.cached['observed_labels'].append(g)
        if p not in self.cached['predlabels']:
            self.cached['predlabels'].append(p)
        if g not in self.cached['goldlabels']:
            self.cached['goldlabels'].append(g)
        
        
    def _light_add(self, g, p):
        '''Takes a list with gold labels and one with predicted labels
        and adds them to the matrix.
        
        Example:
            self.add(['a', 'b', 'c'], ['a', 'b'])
        
        Each instance in the test corpus should be added with this method.
        '''
        if not isinstance (g, (list, tuple)): g = [g]
        if not isinstance (p, (list, tuple)): p = [p] 
        
        gold = list(g)
        if not g: gold = [None]
        pred = list(p)
        if not p: pred = [None]
        
        # For the accuracy
        if set(gold) == set(pred): self.correct+=1
        self.total+=1
        
        try:
            self.cached['observed_labels']
        except KeyError:
            self.cached['observed_labels'] = []
            self.cached['predlabels'] = []
            self.cached['goldlabels'] = []
        
        # True pos
        for p in pred:
            if p in gold:
                cache_key = u'tp_'+unicode(p)
                self.cached[cache_key] = self.cached.get(cache_key, 0) + 1
                gold.remove(p)
                if p not in self.cached['goldlabels']: self.cached['goldlabels'].append(p)
            else:
                cache_key = u'fp_'+unicode(p)
                self.cached[cache_key] = self.cached.get(cache_key, 0) + 1
                
            if p not in self.cached['observed_labels']:
                self.cached['observed_labels'].append(p)
            if p not in self.cached['predlabels']:
                self.cached['predlabels'].append(p)
        
        # False neg
        for g in gold:
            cache_key = u'fn_'+unicode(g)
            self.cached[cache_key] = self.cached.get(cache_key, 0) + 1
            
            if g not in self.cached['observed_labels']:
                self.cached['observed_labels'].append(g)
            if g not in self.cached['goldlabels']:
                self.cached['goldlabels'].append(g)
            
        

    def _full_add(self, g, p):
        '''Takes a list with gold labels and one with predicted labels
        and adds them to the matrix.
        
        Example:
            self.add(['a', 'b', 'c'], ['a', 'b'])
        
        Each instance in the test corpus should be added with this method.
        '''
        if not isinstance (g, (list, tuple)): g = [g]
        if not isinstance (p, (list, tuple)): p = [p] 
        
        # Add empties
        if not g:
            gold = [None]
            self._ng.append(None)
        else:
            gold=[]
            for e in g:
                if isinstance(e, str):
                    e = e.decode(self.encoding)
                elif isinstance(e, (unicode, int, float)) or e is None:
                    pass
                else:
                    raise TypeError('labels should be unicode objects, None, float or integers: %s (%s)' %(str(e), type(e)))
                self._ng.append(e)
                gold.append(e)
        if not p:
            pred = [None]
            self._np.append(None)
        else:
            pred=[]
            for e in p:
                if isinstance(e, str):
                    e = e.decode(self.encoding)
                elif isinstance(e, (unicode, int, float)) or e is None:
                    pass
                else:
                    raise TypeError('labels should be unicode objects, None, float or integers')
                self._np.append(e)
                pred.append(e)
        
        gref = gold[:]
        pref = pred[:]
        
        if len(gref) > 1 and inside(None, gref): raise ValueError('None cannot be combined with any other label: %s' %gref)
        if len(pref) > 1 and inside(None, pref): raise ValueError('None cannot be combined with any other label: %s' %pref)
        
        # For the true negatives
        self._instancecache.append( set(gold+pred) )
        
        # For the accuracy
        if set(gold) == set(pred): self.correct+=1
        self.total+=1
        
        # The correctly predicted ones (TP)
        cache=[]
        for g in gold:
            if inside(g, pred):
                if not inside(g, self.keys()):
                    self[g]={g:Count(1,1)}
                else:
                    self[g][g] = self[g].get(g, Count()) + 1
                
                pred.remove(g)
            else:
                cache.append(g)
        gold = cache[:]
        
        # There is only one left, so we know which
        # FP is coupled with which FN
        if len(pred) == 1 == len(gold):
            g = gold[0]
            p = pred[0]
            
            if not inside(g, self.keys()): self[g] = {}
            if inside(p, self[g].keys()):
                self[g][p][ROW] += 1
                self[g][p][COL] += 1
            else:
                self[g][p] = Count(1,1)
            return
        
        # We know these golds are not in pred (FNs)
        if not pref:
            pred2 = [None]
        else:
            pred2 = pref
        point = 1.0/len(pred2)
        if not point%1: point = int(point)
        for g in gold:
            if not inside(g, self.keys()): self[g] = {}
            for p in pred2:
                if inside(p, self[g].keys()):
                    self[g][p][ROW] += point
                else:
                    self[g][p] = Count(row=point)
        
        # The FPs
        if not gref:
            gold=[None]
        else:
            gold = gref
        point = 1.0/len(gold)
        if not point%1: point = int(point)
        for p in pred:
            for g in gold:
                if inside(g, self.keys()):
                    if inside(p, self[g].keys()):
                        self[g][p][COL] += point
                    else:
                        self[g][p] = Count(col=point)
                else:
                    self[g] = {p:Count(col=point)}
        
        
    def add_training(self, g):
        '''Adds a list of labels from the training corpus.
        This is used when computing averaged scores with level MICROt.
        
        Example, for an instance with label 'a':
            self.add_training(['a'])
        
        Each instance from the training corpus should be added with this method.
        '''
        for l in g:
            if isinstance(l, str):
                l = l.decode(self.encoding)
            elif isinstance(l, (unicode, int, float)):
                pass
            else:
              raise TypeError('labels should be unicode objects')

            self._freq[l] = self._freq.get(l, 0) + 1

    @property
    def labels(self):
        '''Returns all the labels as used during computation of counts or averaged scores
        
        Depends on the strict setting and whether training counts are set.
        '''
        if self.freq_training:
            l = self.freq_training.keys()
            l.sort()
            if None in l:
                l.remove(None)
                l.append(None)
            return l
        elif self.strict:
            return self.g
        else:
            return self.observed_labels

    @property
    def observed_labels(self):
        '''Returns all labels that are observed in the test corpus, 
        irrespective of the fact if it comes from gold-standard or prediction'''
        if 'observed_labels' not in self.cached.keys():
            out=set(self.g + self.p)
            out = list(out)
            out.sort()
            
            # Put None in last position
            if None in out:
                out.remove(None)
                out.append(None)

            if self.cached:
                self.cached['observed_labels']=out
            
        if 'observed_labels' in self.cached.keys(): return self.cached['observed_labels']
        return out
            
    @property
    def freq_training(self):
        '''The relative frequencies of label in the training corpus.
        '''
        if 'freq_training' not in self.cached.keys():
            # The total number of labels
            if self.compute_none:
                total = sum(self._freq.values())
            else:
                total=sum([v for k,v in self._freq.items() if k is not None])
            
            # The relative frequencies
            out={}
            for l, v in self._freq.items():
                if not self.compute_none:
                    if l is None: continue
                out[l] = float(v)/total
                
            if self.cached: self.cached['freq_training']=out
                
        if 'freq_training' in self.cached.keys(): return self.cached['freq_training']
        return out
            
    @property
    def freq_test(self):
        '''The relative frequencies of labels in the test corpus.
        '''
        if 'freq_test' not in self.cached.keys():
            out={}
                
            total = 0
            for l in self.labels:
                if not self.compute_none:
                    if l is None: continue
                out[l] = float(self.tp(l)+self.fn(l))
                total += (self.tp(l)+self.fn(l))
            for k, v in out.items():
                out[k] = v/total
                
            if self.cached: self.cached['freq_test'] = out

        if 'freq_test' in self.cached.keys(): return self.cached['freq_test']
        return out
            
    @property
    def rowtotals(self):
        '''Returns the rowtotal of the confusionmatrix as a dict.
        This equals the number of times a label occurs in de test reference.'''
        if self.light: raise LightError('Attribute unavailable: reason: ConfusionMatrix in light mode')
        key = 'rowtotals'
        if key not in self.cached.keys():
            out={}
            
            labels = self.observed_labels[:]
            if (None in self._ng or None in self._np) and None not in labels: labels.append(None)
            
            for l in labels:
                try:
                    value = 0
                    for k,count in self[l].items():
                        value = value + count.row
                    out[l] = value
                except KeyError:
                    out[l] = 0
            
            if self.cached: self.cached[key] = out
        
        if key in self.cached.keys(): return self.cached[key]
        return out
        
    @property
    def columntotals(self):
        '''Returns the columntotal of the confusionmatrix as a dict.
        This equals the number of times a label occurs in de test prediction.'''
        if self.light: raise LightError('Attribute unavailable: reason: ConfusionMatrix in light mode')
        key = 'coltotals'
        if key not in self.cached.keys():
            out={}
            
            labels = self.observed_labels[:]
            if (None in self._ng or None in self._np) and None not in labels: labels.append(None)

            for l in labels:
                count = 0
                for l2 in labels:
                    try:
                        count += self[l2][l].col
                    except KeyError:
                        pass
                out[l] = count
            
            if self.cached: self.cached[key] = out
        
        if key in self.cached.keys(): return self.cached[key]
        return out
            
            
    @property
    def g(self):
        '''All labels in gold-standard'''
        key = 'goldlabels'
        if key not in self.cached.keys():
            # Nicer
            out = list(set(self._ng))
            out.sort()
            if None in out:
                out.remove(None)
                if self.compute_none: out.append(None)
            
            if self.cached: self.cached[key] = out
            
        if key in self.cached.keys(): return self.cached[key]
        return out
                
    @property
    def p(self):
        '''All labels in predicted'''
        '''All labels in gold-standard'''
        key = 'predlabels'
        if key not in self.cached.keys():
            out = list(set(self._np))
            out.sort()
            if None in out:
                out.remove(None)
                if self.compute_none: out.append(None)
            
            if self.cached: self.cached[key] = out
            
        if key in self.cached.keys(): return self.cached[key]
        return out

    @property
    def ng(self):
        '''Returns the number of different labels in the gold-standard of test'''
        return len(self.g)
        
    @property
    def np(self):
        '''Returns the number of different labels in the predictions of test'''
        return len(self.p)
        
    @property
    def nG(self):
        '''Returns the number of different labels in the training corpus'''
        return len(self.freq_training.keys())
        
            
    def tp(self, label):
        '''Returns the true positive of a given label'''
        if isinstance(label, str): label = label.decode(self.encoding)
        
        cache_key = u'tp_'+unicode(label)
        if cache_key not in self.cached.keys():
            tp = 0
            if label in self.keys():
                if label in self[label].keys():
                    tp = min([self[label][label].col, self[label][label].row])
                    
            if self.cached: self.cached[cache_key] = tp
        
        try:
            return self.cached[cache_key]
        except KeyError:
            return tp
            
    def fp(self, label):
        '''Returns the false positives of a label'''
        if isinstance(label, str): label = label.decode(self.encoding)
        
        cache_key = u'fp_'+unicode(label)
        if cache_key not in self.cached.keys():
            fp = 0
            for l,scores in self.items():
                try:
                    if l == label and type(l) == type(label):
                        # this line is needed if you get predictions like:
                        # GOLD: K   PRED: K_K_K
                        extra = scores[label].col - scores[label].row
                        if extra > 0:
                            fp+=extra
                    else:
                        fp += scores[label].col
                except KeyError:
                    pass
            if not fp%1: fp = int(fp)
            if self.cached: self.cached[cache_key] = fp
            
        try:
            return self.cached[cache_key]
        except KeyError:
            return fp
                 
    def fn(self, label):
        '''Returns the false negatives of a label'''
        if isinstance(label, str): label = label.decode(self.encoding)
        
        cache_key = u'fn_'+unicode(label)
        if cache_key not in self.cached.keys():
            fn = 0
            if inside(label, self.keys()):
                for l,c in self[label].items():
                    if l == label and type(l) == type(label):
                        # this line is needed if you get predictions like:
                        # GOLD: K_K_K   PRED: K
                        extra = c.col - c.row
                        if extra < 0:
                            fn+=abs(extra)
                    else:
                        fn += c.row 
                
                if not fn%1: fn = int(fn)
                if self.cached: self.cached[cache_key] = fn

        try:
            return self.cached[cache_key]
        except KeyError:
            return fn


    def tn(self, label):
        '''Returns the true negatives of a label.
        
        These are all occurrences of any other label than label that are not predicted as label.
        '''
        if self.light: raise LightError('Attribute unavailable: reason: ConfusionMatrix in light mode')
        if isinstance(label, str): label = label.decode(self.encoding)
        
        cache_key = u'tns'
        if cache_key not in self.cached.keys():
            # Compute ALL true negatives
            cache={}
            for l in self.observed_labels:
                tn=0
                for i in self._instancecache:
                    if l not in i:
                        tn += 1
                cache[l] = tn
                
            if self.cached:
                self.cached[cache_key] = cache
                self._instancecache = {}
            
        try:
            c = self.cached[cache_key]
        except KeyError:
            try:
                return cache[label]
            except KeyError:
                return self.total
        else:
            try:
                return c[label]
            except KeyError:
                return self.total

    def _sum_count(self, func):
        cache_key = func.upper()
        if cache_key not in self.cached.keys():
            count = 0
            
            labels = self.labels[:]
            if not self.compute_none and None in labels: labels.remove(None)
            for l in labels:
                count += getattr(self, func)(l)
            if self.cached: self.cached[cache_key] = count
        
        try:
            return self.cached[cache_key]
        except KeyError:
            return count
            
    @property
    def TP(self):
        '''Sum of all true positives of all labels that are taken into account'''
        return self._sum_count('tp')
    @property
    def FP(self):
        '''Sum of all false positives of all labels that are taken into account'''
        return self._sum_count('fp')
    @property
    def TN(self):
        '''Sum of all true negatives of all labels that are taken into account'''
        return self._sum_count('tn')
    @property
    def FN(self):
        '''Sum of all false negatives of all labels that are taken into account'''
        return self._sum_count('fn')

        
    def precision(self, label):
        '''Returns the precision of a label'''
        return precision(self.tp(label), self.fp(label), 0, 0)
        
    def recall(self, label):
        '''Returns the recall of a label
        
        This is also called true positive rate (TPR)
        '''
        return recall(self.tp(label), 0, 0, self.fn(label))
        
    def fpr(self, label):
        '''Returns the falso postive rate (FPR) of a label
        
        This is also called fall-out
        '''
        if self.light: raise LightError('Attribute unavailable: reason: ConfusionMatrix in light mode')
        return fpr(0, self.fp(label), self.tn(label), 0)
        
    def fscore(self, label, beta=1):
        '''Returns the F-score for a label'''
        return fscore(self.tp(label), self.fp(label), 0, self.fn(label), beta=beta)
        
        
    def auc(self, label):
        '''Returns the area under the ROC (AUC) for a label'''
        if self.light: raise LightError('Attribute unavailable: reason: ConfusionMatrix in light mode')
        return auc(self.tp(label), self.fp(label), self.tn(label), self.fn(label))
        
    def accuracy(self):
        '''The accuracy
        In the case of multilabels: correct means the whole label the same'''
        return float(self.correct)/self.total
        
    def averaged(self, level=MICRO, score=FSCORE, training=False, beta=1):
        '''Returns the averaged version of the score
        
        Possible levels:
            MICRO: micro-averaging: score( sum_q[tp_i], sum_q[fp_i], sum_q[fn_i]) 
            MACRO: macro-averaging: 1/q * sum_q[ score(tp_i, fp_i, fn_i) ]
            MICROt: micro-averaging: sum_q[ f_i*score(tp_i, fp_i, fn_i) ]
        
            with:
                score()          : a given evaluation score
                tp_i, fp_i, fn_i : count for label i
                f_i              : relative frequency of label in in training corpus
                q                : the number of different labels
                sum_q            : summing over all labels
                
        score: one of the following values: PRECISION, RECALL, FSCORE, AUC
        
        beta: the beta value if fscore is computed
        
        training: does nothing, for backwards compatibility. If training counts are present
                  they are used. 
        '''
        # The labels in training
        labels = self.labels[:]

        if not self.compute_none:
            if None in labels: labels.remove(None)

        if score == FSCORE:
            func = fscore
        elif score == PRECISION:
            func = precision
        elif score == RECALL:
            func = recall
        elif score == AUC:
            func = auc
        else:
            raise ValueError('Unknown score: %s' %score)

        if self.light:
            if level == MACRO:
                return sum([func( self.tp(l), self.fp(l), -1, self.fn(l), beta ) for l in labels]) / float(len(labels)) 
            elif level == MICRO:
                return func(sum([self.tp(l) for l in labels]), sum([self.fp(l) for l in labels]), sum([-1 for l in labels]), sum([self.fn(l) for l in labels]), beta=beta)
            elif level == MICROt:
                if self.freq_training:
                    # Use the frequencies from the training corpus
                    return sum([self.freq_training.get(l, 0)*func( self.tp(l), self.fp(l), -1, self.fn(l), beta=beta ) for l in labels])
                else:
                    # Use the frequencies from the test corpus
                    return sum([self.freq_test.get(l, 0)*func( self.tp(l), self.fp(l), -1, self.fn(l), beta=beta ) for l in labels])
            else:
                raise ValueError('Unknown level: %s' %str(level))
        else:
            if level == MACRO:
                return sum([func( self.tp(l), self.fp(l), self.tn(l), self.fn(l), beta ) for l in labels]) / float(len(labels)) 
            elif level == MICRO:
                return func(sum([self.tp(l) for l in labels]), sum([self.fp(l) for l in labels]), sum([self.tn(l) for l in labels]), sum([self.fn(l) for l in labels]), beta=beta)
            elif level == MICROt:
                if self.freq_training:
                    # Use the frequencies from the training corpus
                    return sum([self.freq_training.get(l, 0)*func( self.tp(l), self.fp(l), self.tn(l), self.fn(l), beta=beta ) for l in labels])
                else:
                    # Use the frequencies from the test corpus
                    return sum([self.freq_test.get(l, 0)*func( self.tp(l), self.fp(l), self.tn(l), self.fn(l), beta=beta ) for l in labels])
            else:
                raise ValueError('Unknown level: %s' %str(level))

        

def inside(x, l):
    if isinstance(x, str):
        if not x:
            x = u''
        #else:
        #    x = x.decode('utf8')
    return bool([k for k in l if k == x and type(k) == type(x)])

distribution = re.compile('{[^}]+}')
def fread(fname, encoding=encoding, delete_distr=True):
    '''delete_distr: if True remove {...} at the end of the line'''
    with open(os.path.abspath(os.path.expanduser(fname)), 'rU') as f:
        for l in f:
            if delete_distr: l = distribution.sub('', l)
            line = l.strip()
            if line: yield line.decode(encoding)    

def get_translation(fname, encoding=encoding):
    out={}
    for line in fread(fname, encoding=encoding):
        k, v = line.split('>')
        out[k.strip()] = v.strip() 
    return out

def get_plain_iterator(fname, gold_index, pred_index=None, fsep=None, lsep='_', encoding=encoding, multi=False, ignore=[], none='NONE'):
    '''Yields all gold_labels, pred_lables tuples
     If pred_index is None: it only looks for gold labels (training files, maxent files, ...)
     
    gold_index: the index of the gold standard label
    pred_index: the index of the predicted label
     
    fsep: the field separator
    lsep: in the case of multilabels, the label separator
    none: the empty prediction, None
    
    multi: if True, split labels using lsep
    
    encoding: encoding used to decode the files
    '''
    if isinstance(fsep, str): fsep=fsep.decode(encoding)
    if isinstance(lsep, str): lsep=lsep.decode(encoding)
    
    for line in fread(fname, encoding=encoding):
        # Don't keep lines that match
        if ignore:
            for pattern in ignore:
                quit=False
                if pattern.search(line):
                    quit=True
                    break
            if quit: continue

        # The fields
        fields = []
        for v in line.split(fsep):
            if v == none:
                fields.append(None)
            else:
                fields.append(v)
        
        # GOLD
        try:
            gold = fields[gold_index]
        except IndexError:
            raise IndexError('Maybe the field separator is not set correctly, see -f') 
        if multi:
            if gold is None:
                pass
            else:
                gold = gold.split(lsep)
        else:
            gold = [gold]
            
        if pred_index is not None:
            pred = fields[pred_index]
            if multi:
                if pred is None:
                    pass
                else:
                    pred = pred.split(lsep)
            else:
                pred = [pred]
                
        if pred_index is None:
            yield gold
        else:
            yield gold, pred
            
        
def get_combination_iterator(test_file, prediction_file, gold_index, fsep=None, lsep='_', encoding=encoding, multi=False, cutoff=None, ignore=[], pos_label='+1', neg_label='-1', none='NONE'):
    test = fread(test_file, encoding=encoding)
    pred = fread(prediction_file, encoding=encoding)

    for line, pred in zip(test, pred):
        # Don't keep lines that match
        if ignore:
            for pattern in ignore:
                quit=False
                if pattern.search(line):
                    quit=True
                    break
            if quit: continue
    
        gold = line.split(fsep)[gold_index]
        
        if cutoff is None:
            # Maxent style
            if gold == none: gold = None
            if pred == none: pred = None
            if multi:
                if gold is None:
                    pass
                else:
                    gold = gold.split(lsep)
                if pred is None:
                    pass
                else:
                    pred = pred.split(lsep)
            else:
                gold = [gold]
                pred = [pred]
        else:
            # SVM style
            if multi: raise ValueError('Do not know how to handle multi labels together with cutoff')
            
            gold=[gold]
            
            if float(pred) >= cutoff:
                pred = [pos_label]
            else:
                pred = [neg_label]
            
        yield gold, pred
        
def getConfusionMatrixFromIterator(iterator, training_iterator=None , encoding=encoding, translation=None, compute_none=False, strict=False, cached=True, light=False):
    '''Construct a confusion matrix from an iterator
    
    multi: set to True if the labels are multilabels and should be analyzed at the single label level
    training_iterator: produce labels in train
    
    translation: a dict. All labels are translated key => value.
    compute_none: if True, the None label is taken into account for the label frequences. This has an influence on
                  the macro-averaged scores and the label-freq-based micro-averaged scores.
    cached      : if set to True, scores are cached. This means that after computing a score, using the add()
                  or add_training() method may introduce errors.
    '''
    cm = ConfusionMatrix(encoding=encoding, compute_none=compute_none, strict=strict, cached=cached, light=light)
                
    if training_iterator is not None:
        for labels in training_iterator:
            if translation:
                try:
                    labels = [translation[l] for l in labels]
                except KeyError, e:
                    raise KeyError('"%s" is not present in translation' %(e.args[0].encode(encoding)))
        
            cm.add_training(labels)
        
    for gold_labels, pred_labels in iterator:
        if translation:
            try:
                gold_labels = [translation[l] for l in gold_labels]
                pred_labels = [translation[l] for l in pred_labels]
            except KeyError, e:
                raise KeyError('"%s" is not present in translation' %(e.args[0].encode(encoding)))
        cm.add(gold_labels, pred_labels)
        
    return cm
    
def getConfusionMatrixFromFile(fname, gold_index=-2, pred_index=-1, fsep=None, lsep='_', none='NONE', ignore=[], multi=False, training_file=None, train_gold_index=-1, encoding=encoding, strict=False, compute_none=False, cached=True):
    '''Construct a confusion matrix from a TiMBL-style file
    
    gold_index      : the index of the gold standard label in test
    pred_index      : the index of the predicted label in test
    train_gold_index: the index of the gold standard label in the optional training file
     
    fsep: the field separator
    lsep: in the case of multilabels, the label separator
    
    ignore: a list of Pattern objects. Lines that match any of these are omitted. 
    multi: set to True if the labels are multilabels and should be analyzed at the single label level
    training_file: path to the training file used to produce the predictions
    
    compute_none: if True, the None label is taken into account for the label frequences.
                  This has an influence on the macro-averaged scores and the
                  label-freq-based micro-averaged scores.
    strict      : if True, only use labels from the gold-standard (and not from predicted)
    cached      : if set to True, scores are cached. This means that after computing a score, using the add()
                  or add_training() method may introduce errors.
    '''
    training_iterator=None
    if training_file:
        training_iterator = get_plain_iterator(training_file, train_gold_index, pred_index=None, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi, none=none)

    iterator = get_plain_iterator(fname, gold_index, pred_index=pred_index, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi, ignore=ignore, none=none) 
   
    return getConfusionMatrixFromIterator(iterator, training_iterator=training_iterator, encoding=encoding, strict=strict, compute_none=compute_none, cached=cached)
        
                
def main(test_file, gold_index=-2, pred_index=-1, fsep=None, lsep='_', ignore=[], multi=False, training_file=None, train_gold_index=-1, beta=1, print_cm=True,\
         verbose=True, prediction_file=None, cutoff=None, translation_file=None, compute_none=False, strict=False, print_labels=True,
         pos_label='+1', neg_label='-1', none='NONE', light=False):
         
    # Getting the optional translation
    translation=None
    if translation_file: translation = get_translation(translation_file)
    
    # Get the iterators
    training_iterator=None
    if prediction_file:
        # SVM or Maxent: test file and prediction file are separate
        iterator = get_combination_iterator(test_file, prediction_file, gold_index, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi, cutoff=cutoff, none=none, ignore=ignore, pos_label=pos_label, neg_label=neg_label)
    else:
        # TiMBL: test file and prediction file are one
        iterator = get_plain_iterator(test_file, gold_index, pred_index=pred_index, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi, ignore=ignore, none=none)
        
    # Create the CM
    if training_file: training_iterator = get_plain_iterator(training_file, train_gold_index, pred_index=None, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi, none=none)
    cm = getConfusionMatrixFromIterator(iterator, training_iterator=training_iterator, encoding=encoding, translation=translation, compute_none=compute_none, strict=strict, light=light)
      
    # Size of largest label
    safe_labels = set(cm.observed_labels)
    if training_file: safe_labels.update(set(cm.freq_training.keys()))
    safe_labels = list(safe_labels)
    if None in safe_labels: safe_labels.remove(None)
    s = max(6, max([len(l) for l in safe_labels]))+1
      
    # Get the labels to report on
    if training_file:
        labels = cm.freq_training.keys()
    else:
        labels = cm.observed_labels
    labels.sort()
    if None in labels:
        labels.remove(None)
        labels.append(None)
      
    # Extra info
    if verbose:
        print '\n%s\n' %('='*80)
        print 'STATISTICS'
        
        k=''
        if cm.ng < 10:
            k = cm.keys()
            k.sort()
            if None in k: 
                if compute_none: k.append('*')
                k.remove(None)
            k= ': '+(', '.join(k)).encode(encoding)
        print cm.ng, 'different labels in gold-standard of test corpus', k
                
        k=''
        if cm.np < 10:
            k=set()
            for v in cm.values(): k.update(v.keys())
            k = list(k)
            k.sort()
            if None in k: 
                if compute_none: k.append('*')
                k.remove(None)
            k= ': '+(', '.join(k)).encode(encoding)
        print cm.np, 'different labels in prediction of test corpus   ', k
            
        if training_file:
            k=''
            if cm.nG < 10:
                k=cm.freq_training.keys()
                k.sort()
                k= ': '+(', '.join(k)).encode(encoding)
            print cm.nG, 'different labels in training corpus             ', k
        
        print
        ss='TEST CORPUS'
        if training_file:
            ss='TRAINING CORPUS'
        print 'LABEL FREQUENCIES IN', ss
        
        
        for l in labels:
            # The label to print
            pl=l
            
            # Make a difference if None should be included or not
            if not compute_none:
                if l is None: continue
            else:
                if l is None: pl='*'
            ff = '%%%ds : %%.5f' %s
            
            # Print the frequencies
            if training_file:
                v = cm.freq_training.get(l, 0)
            else:
                v = cm.freq_test.get(l, 0)
                
            try:
                print (ff %(pl, v)).encode(encoding)
            except:
                raise
        
        print '\n%s\n' %('='*80)

    # Print confusionmatrix
    if print_cm and  not light:
        print 'CONFUSION MATRIX'
        print cm
        print '\n%s\n' %('='*80)
        
    if print_labels:
        if light:
            # The counts for all seen labels
            format = '    '.join(['%%%ds' %s for i in range(4)])
            print format %('', 'TP', 'FP', 'FN')
            for l in labels:
                pl=l
                if l is None: pl='*'
                print (format %(pl+':', cm.tp(l), cm.fp(l), cm.fn(l))).encode(encoding)
                
            # precision, recall, fscore per label
            print '\n%s\n' %('='*80)
            format = '    '.join(['%%%ds' %s for i in range(4)])
            print format %('', 'PREC', 'RECALL', 'F(%.1f)' %beta)
            for l in labels:
                pl=l
                if l is None: pl='*'
                print (format %(pl+':', '%.5f' %cm.precision(l), '%.5f' %cm.recall(l), '%.5f' %cm.fscore(l, beta=beta), )).encode(encoding)
            print '\n%s\n' %('='*80)
        else:
            # The counts for all seen labels
            format = '    '.join(['%%%ds' %s for i in range(5)])
            print format %('', 'TP', 'FP', 'TN', 'FN')
            for l in labels:
                pl=l
                if l is None: pl='*'
                print (format %(pl+':', cm.tp(l), cm.fp(l), cm.tn(l), cm.fn(l))).encode(encoding)
            
            # precision, recall, fscore per label
            print '\n%s\n' %('='*80)
            format = '    '.join(['%%%ds' %s for i in range(5)])
            print format %('', 'PREC', 'RECALL', 'F(%.1f)' %beta, 'AUC')
            for l in labels:
                pl=l
                if l is None: pl='*'
                print (format %(pl+':', '%.5f' %cm.precision(l), '%.5f' %cm.recall(l), '%.5f' %cm.fscore(l, beta=beta), '%.5f' %cm.auc(l))).encode(encoding)
            print '\n%s\n' %('='*80)
      
    # averaged scores
    print 'MACRO-AVERAGED'
    print 'PRECISION    : %.5f' %(cm.averaged(level=MACRO, score=PRECISION))
    print 'RECALL       : %.5f' %(cm.averaged(level=MACRO, score=RECALL))
    print 'F-SCORE (%2.1f): %.5f' %(beta, cm.averaged(level=MACRO, score=FSCORE, beta=beta))
    if not light: print 'AUC          : %.5f' %(cm.averaged(level=MACRO, score=AUC, beta=beta))
        
    print
    print 'MICRO-AVERAGED USING LABEL FREQUENCIES'
    print 'PRECISION    : %.5f' %(cm.averaged(level=MICROt, score=PRECISION))
    print 'RECALL       : %.5f' %(cm.averaged(level=MICROt, score=RECALL))
    print 'F-SCORE (%2.1f): %.5f' %(beta, cm.averaged(level=MICROt, score=FSCORE, beta=beta))
    if not light: print 'AUC          : %.5f' %(cm.averaged(level=MICROt, score=AUC, beta=beta))  
    
    print
    print 'MICRO-AVERAGED'
    print 'PRECISION    : %.5f' %(cm.averaged(level=MICRO, score=PRECISION, beta=beta))
    print 'RECALL       : %.5f' %(cm.averaged(level=MICRO, score=RECALL, beta=beta))
    print 'F-SCORE (%2.1f): %.5f' %(beta, cm.averaged(level=MICRO, score=FSCORE, beta=beta))
    if not light: print 'AUC          : %.5f' %(cm.averaged(level=MICRO, score=AUC, beta=beta))

    if verbose:
        print
        if training_file:
            print 'INFO: Computing averaged scores using training corpus'
        else:
            print 'INFO: Computing averaged scores only using test corpus'
    
    # Accuracy  
    print '\n%s\n' %('='*80)
    print 'ACCURACY     : %.5f (%d out of %d)' %(cm.accuracy(), cm.correct, cm.total)
        
    print '\n%s\n' %('='*80)
      
    return cm
      
def print_info():
    print >>sys.stderr, '''INFORMATION ON THE CALCULATION
    
1. SINGLE LABEL
    Each instance has one and exactly one label.

    - true positive  (TP) : +1 for a given label, if that label is the same
                            in gold-standard and prediction
    - false positive (FP) : +1 for a given label, if that label occurs in the
                            prediction and not in gold-standard
    - true negative  (TN) : the total number of instances in gold NOT of a given
                            label minus the FP of that label.
    - false negative (FN) : +1 for a given label, if that label occurs in the
                            gold-standard and not in the prediction

    Note that an error, always leads to a simultaneous increase of the general FP
    and general FN counts.

2. MULTI LABEL
    Each instance can have 0 or more labels.

    - true positive  (TP) : +1 for a given label, if that label occurs in the
                            gold-standard multi-label and the predicted multi-label
    - false positive (FP) : +1 for a given label, if that label occurs in the
                            predicted multi-label and not in gold-standard multi-label
    - false negative (FN) : +1 for a given label, if that label occurs in the
                            gold-standard multi-label and not in the predicted multi-label
    - true negative  (TN) : the total number of occurrences that are not a given label
                            minus the FPs of that label 
    
    Note that a different number of labels in the gold-standard and predicted multi-labels
    leads to split counts in the confusionmatrix. A split count means that a cell in the
    matrix has a different value, depending if you want to add all values of a row or a
    column. A split value of 1/3 means that for getting the row total (= number of occurrences
    in gold) you need to take value 3. For getting the column total (=number of occurrences in
    predicted), you need to take value 1. 
    
    In the confusionmatrix, wrong predictions may lead to split counts. For example,
    consider the gold-standard multi-label [A] and the predicted multi-label [B C] of size n=2.
    This leads to the increase of the cell {A,B} with 1/n=0.5 and also of the cell {A,C} with 
    1/n=0.5 for the row-totals. Leading to a row-total of 1, namely the number of times A
    occurs in the gold. For the column totals, both cells are increased with 1. Leading to
    a column total of 1 for B and for C, because both occur once in the prediction.  
    
    When there are no labels, thus [], the empty label is introduced. The -N switch changes
    the behavior towards the empty label while averaging scores.
    
    Note that the script does not simplify the labels. Thus,
        GOLD LABELS:      [A]
        PREDICTED LABELS: [A, A, A]
    leads to one TP and two FPs. If you want this to lead to a simple TP, you need to make
    sure there are no doubles in the class labels.

3. COMPUTED SCORES
    * Per label
    - precision (P) : TP/(TP+FP) for a given label
    - recall    (P) : TP/(TP+FN) for a given label
    - F-score   (P) : (1+beta^2)*TP/((1+beta^2)*TP+beta^2*FP+FN) for a given label
    - AUC           : 0.5*( TP/(TP+FN) - FP/(FP+TN) + 1 )
    
    Note that a TP of 0 and a FP of 0, leads to a precision of 0.
    Note that a TP of 0 and a FN of 0, leads to a recall of 0, etc. 
    
    * Macro-averaged
    sum(score_per_label)/N
    With N the number of labels in the training set or in the gold standard
    of the test set, depending on -t.
        
    * Label-frequency-based micro-averaged
    sum( label_frequency*score_per_label )
    The label frequency can be retrieved from the training corpus (-t) or the test corpus. 
    Labels that do not occur in the test corpus, but do occur in the training corpus
    are included when -t is set.
        
    * Micro-averaged
    score( sum(tp_per_label), sum(fp_per_label), sum(fn_per_label) )     
        
    * Accuracy
    Accuracy is always computed by taking into account the entire (multi-)label.
        

4 . GENERAL RELATIONS
    * With the -m option set or when there are no multi-labels:
        MICRO-AVERAGED PRECISION always equals MICRO-AVERAGED RECALL.
        MICRO-AVERAGED F-SCORE is insensitive to beta.
    * If the labels are no multi-labels:
        ACCURACY always equals MICRO-AVERAGED F-SCORE.'
    
5. SETTINGS

    * compute_none (-N)

    The reason for this is that for averaged scores, it may not make sense
    for you to take into account the empty label. 
    
    This is only an issue when there are empty predictions. See -e
    
    * strict (-S)
    
    Has only an influence in these cases:
      - with averaged scores, not with per-class scores
      - if there are predicted labels that are not in the gold-standard
      - if training frequencies are not used 
    
    For example, assume only label A has a recall of 0.5, all others have recall 0. Labels A,B,C
    are in the gold-standard, the predicted labels are A,B,D. When strict=False, all labels are
    taken into account. The macro-averaged recall becomes 0.5/4=0.125 because there are 4 labels:
    A,B,C,D.
    With strict=True, this becomes 0.5/3=0.167 because there are only 3 labels in gold: A,B,C.
        
    If you are comparing multiple systems on a test set that does not have all labels from the
    training set in it s gold standard, best is to set strict=True.
    Indeed, if a system predicts a label that is not on gold, strict=False would put a disadvantage
    on this system compared to a system that only predicts labels from the test gold-standard. 
    
    Most of the time, all training labels are in the gold-standard of the test set and strict/non-strict
    is not an issue.
    '''
          
              
def _usage():
    print >>sys.stderr, '''Compute evaluation statistics (version %s)
    
The script reports on:
    - true positive, false positive, true negative and false negative counts for each label.
    - precision, recall, F-score, AUC and accuracy for each label and averaged
      in different manners.
    - the confusionmatrix.
    
USAGE
    $ python confusionmatrix.py [-g index] [-p index] [-G index] [-f fsep] [-l lsep] [-e str] [-b beta] [-m] [-N] [-t training_file][-S] [-C cutoff] [-I pattern] [-T translation] [-H] [-L] [-M] [-v] [-V] test_file [prediction_file]
    
    test_file:       a file with a gold-standard/predicted label values.
                     One pair per line. The line should consist of a number
                     of fields (at least 2), separated by fsep. One field
                     should contain the gold-standard, another should contain
                     the predicted label. The other fields are free. 
              
                     This is useful for evaluating TiMBL output.
              
    OR
    
    test_file:       a file with the gold-standard. One value per line.
                     The line should consist of a number of fields (at least 1),
                     separated by fsep. One field should contain the gold-standard.
                     The other fields are free.
    prediction_file: One value per line. The lines should be paired with those
                     of test_file and should gold the predicted labels.
              
                     This is useful for evaluating e.g. SVM Light output. For SVM Light,
                     -C should be set also.


OPTIONS
    -g index: the index of the field containing the gold-standard.
              Starting at 0. (default: penultimate field)
    -p index: the index of the field containing the predicted label.
              Starting at 0. (default: last field)
    -G index: the index of the field containing the gold-standard in the optional training file.
              Starting at 0. (default: last field)
    
    -f fsep: the field separator (default: any whitespace)
    -l lsep: the label separator (default: _)
    -e str : the empty prediction (default: NONE)
    
    -b beta: set the beta value for the F-scores (default: 1)
    -m: if set, the labels are considered to be combinations of multiple labels
        separated by lsep. The class label, set by -e, is considered the empty prediction.
    -N: do not consider the empty prediction as a class label
        
    -t training_file: a file in the same format as test file. Is used to compute alternative
                      evaluation scores. See AVERAGE SCORES 
    -S: not be strict in the used class labels. If set, all occurring class-labels in the
        test set are used. Per default, only class labels that are in the gold-standard are
        taken into account. See -V.
    
    -C cutoff: A float. When set, a prediction_file should be given also. When a prediction has a
               value below this cutoff, the label becomes -1; otherwise it becomes 1. Useful for
               evaluating SVM output.
    -I pattern: a regex pattern. Lines in the test_file that match this pattern are ignored.
                For example, to evaluate MBT files, set -I "<utt>".
                For example, to leave out lines starting with #, set -I "^#"
    -T translation: A path to a file with the format:
                      old_label > new_label
                        ...
                    When provided, the labels as they are in training, prediction and test files are
                    translated into the new labels.
    -u label: the label that is given to SVM instances that have a value under cutoff -C (default: -1)
    -a label: the label that is given to SVM instances that have a value above cutoff -C (default: +1)
    
    -H: Trade in functionality for speed. Twice as fast, but true negatives, AUC, printing of
        the confusion matrix, ... are not available.
    -L: Suppress reporting on individual labels. Useful when there are a lot of labels.
    -M: Don't print the confusionmatrix. Useful when there are a lot of different labels.
    -v: Print more information
    -V: Print information about the method of calculation
    
    
AVERAGE SCORES
    There are two ways to compute averaged scores: by using information from the training corpus
    or not. Option -t.
    
    For the macro-averaged scores, using only information from the test corpus may lead to
    different scores compared to using label information from the training corpus when
    there are labels in the training corpus that do not occur in the test corpus. Indeed, 
    the number of labels is used to average the scores.
    
    For the label-frequency-based micro-averaged scores, the relative frequencies of the
    labels are used. The frequencies may originate from the test or the training file.
    
    - When comparing different test files that are labeled with the same training set, it
      is preferred to use option -t.
    - When comparing the same test file that is labeled using different training files,
      it is preferred not to use option -t.
    - When the same test file is labeled with the same training file in different runs
      (for example because the machine learner had other settings), you should choose
       and stick to your choice.
    - When different test files are labeled with different training files, you should
      decide whether the test files or the training files are more similar in regard
      to relative label frequencies. If the training files are more similar, use -t, 
      otherwise don't.
    
    
EXAMPLES
    1. TiMBL
    +++++++++++
    To use on a TiMBL output file:
        $ timbl -f dimin.train -t dimin.test
        $ ./confusionmatrix.py -f ',' -t dimin.train dimin.test.IB1.O.gr.k1.out
    
    2. SVMLight (binary)
    +++++++++++++++++++++
    To use on SVMLight test and prediction file:
        $ ./svm_learn train_file model
        $ ./svm_classify test_file model prediction_file
        $ ./confusionmatrix.py -C 0 -g 0 test_file prediction_file
        
        or if you want to use the training file:
        $ ./confusionmatrix.py -C 0 -g 0 -t train_file test_file prediction_file
        
    SVM Light itself reports accuracy and precision/recall for label 1 when the training
    file is not used.
        
    Note that this only works for binary, non-multilabel SVMs. The class labels should be -1 or 1.
    The values below cutoff c, get label -1.
    
    Tested with SVM-light V6.02
    
ACKNOWLEDGEMENTS
    Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze. (2008) Introduction to Information Retrieval, Cambridge University Press. (http://nlp.stanford.edu/IR-book)
    Grigorios Tsoumakas, Ioannis Katakis, and Ioannis Vlahavas, Mining Multi-label Data, Data Mining and Knowledge Discovery Handbook, Part 6, O. Maimon, L. Rokach (Ed.), Springer, 2nd edition, pp. 667-685, 2010.
    Daelemans, W., Zavrel, J., van der Sloot, K., & van den Bosch, A. (2010). Timbl: Tilburg memory-based learner, version 6.3. Tech. Rep. ILK 10-01, Tilburg University
    van Rijsbergen, C. J. (1975). Information Retrieval . London, UK: Butterworths.
    
SEE ALSO
    http://www.clips.ua.ac.be/~vincent/pdf/microaverage.pdf

%s, %s''' %(__version__, __date__, __author__)
    
    

if __name__ == '__main__':        
    try:
        opts,args=getopt.getopt(sys.argv[1:],'ht:mf:l:g:c:p:G:vMb:C:T:VNI:SLu:a:e:H', ['help'])
    except getopt.GetoptError:
        # print help information and exit:
        _usage()
        sys.exit(2)

    training_file=None
    multi=False
    verbose=False
    beta=1
    compute_none=True
    light=False
    
    fsep=None
    lsep='_'
    none = 'NONE'
    
    gold_index=-2
    pred_index =-1
    train_gold_index=-1
    
    print_cm=True
    print_more_info=False
    print_labels=True
    
    strict=True
    
    # for SVM
    prediction_file=None
    cutoff=None
    pos_label='+1'
    neg_label='-1'
    
    ignore=[]
    translation_file=None
    
    for o, a in opts:
        if o in ('-h', '--help'):
            _usage()
            sys.exit()
        if o == '-t':
            training_file = a
        if o == '-m':
            multi=True
        if o == '-f':
            fsep = a
        if o == '-l':
            lsep = a
        if o == '-e':
            none = a
        if o in ('-c', '-g'):
            # Added -c for backwards compatibility
            gold_index=int(a)
        if o == '-p':
            pred_index=int(a)
        if o == '-G':
            train_gold_index=int(a)
        if o == '-v':
            verbose=True
        if o == '-M':
            print_cm=False
        if o == '-b':
            beta=float(a)
        if o == '-C':
            cutoff=float(a)
        if o =='-T':
            translation_file=a
        if o == '-V':
            print_more_info=True
        if o == '-N':
            compute_none=False
        if o == '-I':
            ignore=[re.compile(a)]
        if o == '-S':
            strict=False
        if o == '-L':
            print_labels=False
        if o == '-u':
            neg_label=a
        if o == '-a':
            pos_label=a
        if o == '-H':
            light=True


    if not args and print_more_info:
        print_info()
        sys.exit(0)

    if len(args) not in [1,2]:
        _usage()
        sys.exit(1)
        
    if lsep == fsep:
        print >>sys.stderr, 'ERROR: fsep cannot be the same as lsep'
        sys.exit(1)
        
    test_file = args[0]

    if len(args) == 2: prediction_file = args[1]

    # RUN
    main(test_file, gold_index=gold_index, pred_index=pred_index, fsep=fsep, lsep=lsep, none=none, ignore=ignore, multi=multi, training_file=training_file,\
         train_gold_index=train_gold_index, beta=beta, print_cm=print_cm, verbose=verbose, cutoff=cutoff, prediction_file=prediction_file, translation_file=translation_file,\
         print_labels=print_labels, pos_label=pos_label, neg_label=neg_label, compute_none=compute_none, strict=strict, light=light)

    if print_more_info: print_info()
        
