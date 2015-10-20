import theano
import numpy as np
import theano.tensor as T

from distance_metrics import minmax

def get_manhattan_fn():
    matrix = T.fmatrix('matrix')
    vector = T.fvector('vector')
    distances = T.sum(T.abs_(matrix - vector), axis=1)
    return theano.function([matrix, vector], distances, allow_input_downcast=True)

def get_euclidean_fn():
    matrix = T.fmatrix('matrix')
    vector = T.fvector('vector')
    distances = T.sqrt(T.sum(T.sqr(matrix - vector), axis=1))
    return theano.function([matrix, vector], distances, allow_input_downcast=True)

def get_minmax_fn():
    vector1 = T.fvector('vector1')
    vector2 = T.fvector('vector2')
    comb = T.stack([vector1, vector2])
    dist = 1.0 - (T.sum(T.min(comb, axis=1)) / (T.sum(T.max(comb, axis=1))+1e-6))
    return theano.function([vector1, vector2], dist, allow_input_downcast=True)

def get_minmax_fn():
    vector1 = T.fvector('vector1')
    vector2 = T.fvector('vector2')
    matrix = T.fmatrix('matrix')

    def minmax(vector1, vector2):
        comb = T.stack([vector1, vector2])
        return 1.0 - (T.sum(T.min(comb, axis=1)) / (T.sum(T.max(comb, axis=1)) + 1e-6))

    distances, updates = theano.scan(fn=minmax,
                            outputs_info=None,
                            sequences=[matrix],
                            non_sequences=vector2)

    return theano.function(inputs=[matrix, vector2], outputs=distances, allow_input_downcast=True)

"""
manhattan = get_manhattan_fn()
euclidean = get_euclidean_fn()

#mat = np.asarray(np.ones((10, 5))*5, dtype=('float32'))
mat = np.asarray([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [5, 3, 1, 2, 6]], dtype='float32')
v2 = np.asarray([5, 2, 3, 3, 6], dtype='float32')

f = get_minmax_fn()

print(f(mat, v2))
print(manhattan(mat, v2))
print(euclidean(mat, v2))

f = get_minmax_fn()
mins = f(v1, v2)
print(mins)

print(minmax(v1, v2))


print(mat)
print(v)
print(mat-v)


print(manhattan(mat, v))
print(euclidean(mat, v))
"""

