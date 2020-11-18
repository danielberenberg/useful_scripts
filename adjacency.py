#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# adjacency.py
# author: dan berenberg

"""
Inducing adjacency matrix from distance map.
"""

MISSING_RESIDUE = 1000.

class Composer(object):
    """
    Composes several callables, assumes the i/o types of each callable are amicable
    """
    def __init__(self, *callables):
        self._callables = callables
        self._composition = self._compose(iter(callables), self.identity)

    def identity(self, x):
        return x

    def __len__(self):
        return len(self._callables)

    def feeding_to(self, fn):
        return Composer(*self._callables, fn)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return Composer(*self._callables[start:stop:step])
        elif isinstance(key, int):
            return self._callables[key]
        elif isinstance(key, tuple):
            return NotImplementedError(f"tuple as index")
        else:
            raise TypeError(f"Invalid argument type: {type(key)}")

    def _compose(self, funcerator, f):
        try:
            g = next(funcerator)
            @functools.wraps(g)
            def h(*args, **kwargs):
                return g(f(*args, **kwargs))
            return self._compose(funcerator, h)
        except StopIteration:
            return f

    def __call__(self, x):
        return self._composition(x)

class AdjacencyMatrixMaker(object):
    """Converts a distance map to an adjacency matrix"""
    def __init__(self, threshold, selfloop=True):
        self._threshold = threshold
        self._selfloop  = selfloop

    @property
    def threshold(self):
        return self._threshold

    def convert(self, distance_map):
        A = distance_map.clone()
        A[A <= 0] = MISSING_RESIDUE
        A[torch.isnan(A)] = MISSING_RESIDUE
        
        A[ A <= self._threshold] = -1.
        A[ A > self._threshold ]= 0.
        A = torch.abs(A)
        A = A - torch.diag(torch.diagonal(A))
        if self._selfloop:
            A = A + torch.eye(A.shape[1])
        
        return A

    def __call__(self, distance_map):
        return self.convert(distance_map)

if __name__ == '__main__':
    pass

