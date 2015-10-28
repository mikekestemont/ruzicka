'''Module to calculate combinations of elements

This module contains a function to calculate all subsets of a given length of
a set of n elements.

E.g. All subsets of length 2 from a set of 5 elements.

There exist 10 subsets:
    
    >>> ncombinations(5, 2)
    10

And the subsets are:
    
    >>> list(combinations(5, 2))
    [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
  

subsets() is a convenience function that yields the actual subsets from a list.

Based on: Kenneth H. Rosen, Discrete Mathematics and Its Applications, 2nd edition (NY: McGraw-Hill, 1991), pp. 284-286

But also interesting:
http://code.activestate.com/recipes/190465-generator-for-permutations-combinations-selections

# License: GNU General Public License, see http://www.clips.ua.ac.be/~vincent/scripts/LICENSE.txt
'''
__date__ = 'July 2013'
__author__ = 'Vincent Van Asch'
__version__ = '1.2.0'


### Helper functions #################################################
def fact(number, bound=1):
    '''Return the faculty: number*(number-1)*(number-2)*...*bound'''
    if number < bound:
        raise ValueError('number should be equal or greater than bound')
    
    if number == bound:
        return bound
    else:
        return number*fact(number-1, bound)
        
def _next(a, numLeft, n, r, total):
    '''Calculate next step'''
    if numLeft == total:
        return a, numLeft-1
    else:
        i=r-1
        while a[i] == n-r+i:
            i=i-1

        a[i] = a[i] + 1
        for j in xrange(i+1, r):
            a[j] = a[i] + j - i

        return a, numLeft-1
        
### Main function ####################################################

def ncombinations(n, r):
    """
    Returns the total number of unique subsets of length r
    you can take from n elements. 
    
    n is the number of elements.
    r is the length of the subsets.
    """
    # Check
    if n < 0: raise ValueError('n should be positive.')
    if r < 0: raise ValueError('r should be positive.')
    if r == 0: return 1
    
    # total = fact(n, 1)/(fact(r, 1)*fact(n-r, 1)) # Don't use this because calculating fact() of high numbers gives a RuntimeError
    return fact(n, n-r+1)/fact(r, 1)

def combinations(n, r):
    """
    Yields all unique subsets of length r that you
    can take from n elements. 
    
    n is the number of elements.
    r is the length of the subsets.
    """
    # Check
    if n < 0: raise ValueError('n should be positive.')
    if r < 0: raise ValueError('r should be positive.')
    if r == 0: 
        yield ()
        raise StopIteration
    if r > n: raise StopIteration
    if n < 1: raise StopIteration
        
    if r == n:
        yield range(0, n)
        raise StopIteration
        
    # Initialize
    getallen = xrange(0,n)
    a = range(r)
    
    # The total number of possible combinations
    total = ncombinations(n, r)

    # Produce all pairs
    numLeft = total
    while numLeft > 0:
        comb=[]  
        a, numLeft = _next(a, numLeft, n, r, total)
        for i in a:
            comb.append(getallen[i])
        
        yield comb
        
        
        
def subsets(l, r):
    '''Takes a list with elements and yields all
    unique subsets of length r.
    
    l: a list
    r: an integer (length of the subset)
    '''
    for c in combinations(len(l), r):
        yield tuple([l[x] for x in c])



def subcombinations(*sizes):
    '''Yields all element combinations.
    
    For example:
        >>> subcombinatins(3,2)
        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
        
    Thus, each element of range(3) is combined with each element of
    range(2). Yielding 2*3 element combinations.
    The number of arguments is free.
    '''
    total = reduce(lambda x,y:x*y, sizes)
    
    limit=10000000
    if total > limit: raise ValueError('The number of combinations would exceed the limit %d' %limit)
    
    data=[[]]
    for size in sizes:
        cache=[]
        for part in data:
            for i in range(size):
                cache.append(part + [i])
        data = cache[:]
        
    assert len(data) == total
    return data
