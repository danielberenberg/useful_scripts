#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# adjacency.py
# author: dan berenberg

"""
Inducing adjacency matrix from distance map.
"""

import functools
import torch

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
    def weighted(self):
        return self._weighted

    @property
    def threshold(self):
        return self._threshold

    def convert(self, distance_map):
        A = distance_map.clone()
        A = ( A <= self._threshold ).float()
        if not self._selfloop:
            n = A.shape[0]
            mask = torch.eye(n).bool()
            A.masked_fill_(mask, 0)
        return A

    def __call__(self, distance_map):
        return self.convert(distance_map)

class CoordLoader(object):
    """
    Converts an N x 3 matrix to an distance matrix
    """
    def __init__(self, silent_if_square=True):
        """
        initialize
        args:
            :silent_if_square (bool) - do nothing if the input matrix is already square
        """
        self.silent_if_square = silent_if_square

    def convert(self, coords):
        shape = coords.shape
        assert len(shape) == 2 
        
        if (shape[0] == shape[1]) and self.silent_if_square:
            return coords # do nothing when the input is already a square matrix
        else:
            assert shape[1] == 3
            return torch.cdist(coords, coords, p=2)

    def __call__(self, coords):
        return self.convert(coords)

if __name__ == '__main__':
    pass

