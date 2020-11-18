#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import faiss
import joblib
import numpy as np
from sklearn.neighbors import KDTree

from .mmdb import MemoryMappedDatasetReader

__all__ = ['KNNDatabase', 'load_knn_db']

class KNNDatabase(object):
    """
    Object encapsulating the combined functionality of 
    the underlying embedding database, the trained cluster index, and a
    queryable interface.
    """
    def __init__(self, reader, index):
        """
        Initialize the index.
        args:
            :reader (MemoryMappedDatasetReader)
            :index  (sklearn.neighbors.KDTree) 
        """
        self.__db  = reader
        self.__idx = index
        if hasattr(self.__idx, 'query'):
            self.query = self.__idx.query
        elif hasattr(self.__idx, 'search'):
            self.query = self.__idx.search

    def nearest_neighbors(self, xq, k=8):
        distances, neighbor_idx = self.query(xq, k=k, return_distance=True)
        neighbor_idx = np.squeeze(neighbor_idx)
        distances    = np.squeeze(distances)
        keys = []
        for nid in neighbor_idx:
            nid = int(nid)
            result = self.__db.get_id(nid)
            _, key = result
            keys.append(key)
        return keys, distances
    
    @property
    def db(self):
        return self.__db
    
    @property
    def idx(self):
        return self.__idx
    
    @property
    def keys(self):
        return self.db.ids()
    
    def embedding(self, key):
        return self.__db.get(key)

    def __getindex__(self, key):
        return self.embedding(key)

    def __del__(self):
        self.db.close()


def load_knn_db(database_path):
    """
    Load indexed MemoryMappedDatabase
    args:
        :database_path (Path or str) - root of database 
    returns:
        :KNNDatabase
    """
    db = MemoryMappedDatasetReader(database_path)
    db.open()

    index_file = list(Path(database_path).glob("trained*index"))[0] 
    if "kdtree" in index_file.stem:
        index = joblib.load(index_file)
    elif "faiss" in index_file.stem:
        index = faiss.read_index(str(index_file))  
    else:
        raise ValueError(f"Cannot infer index type from {index_file}")
    
    return KNNDatabase(db, index)

if __name__ == '__main__':
    pass

