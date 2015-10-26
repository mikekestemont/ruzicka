import theano
import theano.tensor as T

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
    matrix = T.fmatrix('matrix')

    def minmax(vector1, vector2):
        comb = T.stack([vector1, vector2], axis=1)
        return 1.0 - ( T.sum(T.min(comb, axis=1)) / (T.sum(T.max(comb, axis=1)) + 1e-6) )

    distances, updates = theano.scan(fn=minmax,
                            outputs_info=None,
                            sequences=matrix,
                            non_sequences=vector2)

    return theano.function(inputs=[matrix, vector2], outputs=distances, allow_input_downcast=True)
