import theano
import theano.tensor as T

def get_manhattan_fn():
    a = T.fvector('a')
    b = T.fvector('b')
    idxs = T.ivector('idxs')
    distance = T.sum(T.abs_(a[[idxs]] - b[[idxs]]), axis=1)
    return theano.function([a, b, idxs], distance, allow_input_downcast=True)

def get_euclidean_fn():
    a = T.fvector('a')
    b = T.fvector('b')
    idxs = T.ivector('idxs')
    distance = T.sqrt(T.sum(T.sqr(a[[idxs]] - b[[idxs]]), axis=1))
    return theano.function([a, b, idxs], distance, allow_input_downcast=True)

def get_minmax_fn():
    a = T.fvector('a')
    b = T.fvector('b')
    idxs = T.ivector('idxs')
    comb = T.stack([a[[idxs]], b[[idxs]]], axis=1)
    distance = 1.0 - ( T.sum(T.min(comb, axis=1)) / (T.sum(T.max(comb, axis=1)) + 1e-6) )
    return theano.function(inputs=[a, b, idxs], outputs=distance, allow_input_downcast=True)
