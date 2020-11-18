#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import mmap
import csv
import sqlite3
import tempfile
from enum import Enum
from pathlib import Path

import numpy as np

__all__ = ['MemoryMappedDatasetReader',
           'TemporaryMemmap',
           'MemoryMappedDatasetWriter',
           'OneToOneMap', 'save_shard']

def save_shard(array, outfile):
    """
    Writes an array to a numpy readable format and 
    returns the shape of the array to record into the keyfile
    """
    array = np.asarray(array)
    ptr = np.memmap(outfile, dtype='float32', mode='w+', shape=array.shape)
    ptr[:] = array[:]
    del ptr
    return array.shape

def _create_connection(db_file):
    """
    Create connection to local SQLite database, given by db_file
    args:
        :db_file (Path or str) - database file
    returns:
        :sqlite3.Connection or None
    """
    conn = sqlite3.connect(db_file)
    return conn

def _create_table(connection, create_table_sql):
    """Create a table from the create_table_sql statement
    args:
        :conn (sqlite3.Connection)
        :create_table_sql: a create table command
    returns:
        :None, creates the table by side effect
    """
    c = connection.cursor()
    c.execute(create_table_sql)

class OneToOneMap(object):
    """
    Manages a sqlite database mapping one entity to another and back.
    """
    def __init__(self, db_file, read_only=False):
        self.db    = str(db_file)
        self.__connection = None
        self.__read_only = read_only
    
    @property
    def read_only(self):
        return self.__read_only
    
    def toggle_readonly(self):
        self.__read_only = not self.__read_only

    def commit(self):
        self.__connection.commit()


    def keys(self):
        """Yields all of the text keys"""
        underlying_query = "select prot_id from backward;"
        c = self.__connection.cursor()
        yield from c.execute(underlying_query)


    def add(self, src, dst, commit=False):
        """
        Append a relationship to the map
        """
        if not self.read_only:
            fst_query = "insert into forward (id, prot_id) values (?,?)"
            snd_query = "insert into backward (prot_id, id) values (?,?)"
            c = self.__connection.cursor()

            c.execute(fst_query, (src, dst))
            c.execute(snd_query, (dst, src))
            if commit:
                self.commit()

    def retrieve(self, id, direction="forward"):
        """Retrieves a relationship from underlying database"""

        if direction not in ['forward', 'backward']:
            raise ValueError("Bad direction (not forward/backward)")
        else:
            key = "id" if direction == "forward" else "prot_id" 

        query = f"select * from {direction} where {key}=?"
        c = self.__connection.cursor()
        c.execute(query, (id,))
        one = c.fetchone()
        return one

    def open(self):
        CREATE_INDEX_TABLE = """CREATE TABLE IF NOT EXISTS forward (
                            id integer PRIMARY KEY,
                            prot_id text NOT NULL
                            );"""

        CREATE_KEY_TABLE   = """CREATE TABLE IF NOT EXISTS backward (
                            prot_id text PRIMARY KEY,
                            id integer NOT NULL,
                            FOREIGN KEY (id) REFERENCES indices (id)
                            );"""
    
        if self.__connection is None:
            self.__connection = _create_connection(self.db) 
            _create_table(self.__connection, CREATE_INDEX_TABLE)
            _create_table(self.__connection, CREATE_KEY_TABLE)

    def close(self):
        if self.__connection is not None:
            self.__connection.commit()
            self.__connection.close()
            self.__connection = None


class MemoryMappedDatasetComponents(Enum):
    keys     = Path("map.db")
    shards   = Path("shards")
    metadata = Path("metadata.tsv")

class MemoryMappedDatasetWriter(object):
    """
    Generate a potentially very large indexable dataset of keys mapping to vectors 
    ---
    Keys are stored in a two-way sqlite3 database, enabling index- and protein-id-based
    queries

    Vectors are stored in a large memory mapped array.
    Where keymap(protein-id) = some i where mmapped_array[i] = protein-id's vector
    """
    def __init__(self, path,
                 embedding_dim=512,
                 shard_size=2**17,
                 start=False):
        """
        Initialize a dataset writer that will write to `path`.
        args:
            :path (Path or str)  - Path to 'dataset'
            :embedding_dim (int) - Dimensionality of feature vectors
            :shard_size (int)    - Number of records per 'shard'
        """
        self.path = Path(path)
        self._n = shard_size
        self._d = embedding_dim 

        self._s = 0 # shard count
        self._t = 0 # num. lines

        for item in MemoryMappedDatasetComponents:
            setattr(self, item.name, self.path / item.value) 

        self.keydb = OneToOneMap(self.keys) 
        self._shard = None
        self.__open = False
        if start:
            self.open()
    
    def open(self):
        """Open the dataset for writing"""
        if not self.__open:
            self.path.mkdir(exist_ok=True, parents=True)
            self.shards.mkdir(exist_ok=True, parents=True)
            self.keydb.open()

            self._shard_md_pointer = open(self.metadata, 'w')
            self._shard_md_writer  = csv.DictWriter(self._shard_md_pointer, delimiter='\t',
                                                    fieldnames=["shard", "shard_id", "n", "d"])
            self._shard_md_writer.writeheader()

            self._reset_shard()
            self.__open = True

    def _reset_shard(self):
        self._shard = np.zeros((self._n, self._d))

    def close(self):
        """Close the dataset"""
        if self.__open:
            self._save_shard()
            self._record()
            self.keydb.close()
            self._shard_md_pointer.close()
            self.__open = False
            self._shard = None

    def _save_shard(self):
        """
        Save a shard by generating its filename and writing it to a np.memmap pointer.
        """
        shardfile = self.shardfilename
        save_shard(self._shard, self.shardfilename)

    def _record(self):
        row = dict(shard=self.shardfilename, shard_id=self._s, n=self._n, d=self._d)
        self._shard_md_writer.writerow(row)
        self._s += 1

    @property
    def shardfilename(self):
        return f"shards_{self._s:06d}.shrd"

    def set(self, key, value, commit=False):
        """Append an item to the database"""
        reset_shard = False
        if self.__open:
            # if the shard is complete (nonzero t, reached shard capacity)
            # then record the shard and reset
            if self._t and not self._t % self._n:
                self._save_shard()
                self._record()
                self._reset_shard()
                self.keydb.commit()
                reset_shard = True

            self._shard[self._t % self._n] = value 
            # add key
            self.keydb.add(self._t, key, commit=commit)
            self._t += 1 
        return reset_shard


class MemoryMappedDatasetReader(object):
    """
    Read from a potentially very large indexable dataset of keys mapping to vectors 
    ---
    Keys are stored in a two-way sqlite3 database, enabling index- and protein-id-based
    queries

    Vectors are stored in a large memory mapped array.
    Where keymap(protein-id) = some i where mmapped_array[i] = protein-id's vector

    """
    def __init__(self, path, start=False):
        self.path = Path(path)
        for item in MemoryMappedDatasetComponents:
            setattr(self, item.name, self.path / item.value) 

        self.__validate()
        self.keydb = OneToOneMap(self.keys, read_only=True) 
        self.__open = False
        self.__embedding_matrix = None
        self.__shape = None
        self.__members = []
        if start:
            self.open()

    def __validate(self):
        """Superficially assess whether the necessary components exist."""
        return self.shards.exists() and self.keys.exists() and self.metadata.exists() 

    @property
    def shape(self):
        return self.__shape

    def __len__(self):
        return self.shape[0]

    def open(self):
        """Open the reader for business.
        Setup the underlying embedding matrix and the key database
        """
        if not self.__open:
            self.keydb.open()
            # construct the underlying array
            N = 0
            shard_md = []
            with open(self.metadata, 'r') as metadata:
                reader = csv.DictReader(metadata, delimiter='\t')
                for row in reader:
                    N += int(row['n'])
                    shard_md.append(row)

            d = int(shard_md[0]['d'])
            self.__shape = (N, d)
            offset = 0
            self.__embedding_matrix = TemporaryMemmap(shape=self.__shape, dtype=np.float32)
            for row in shard_md:
                n, d = map(int, (row['n'], row['d']))
                spath = self.shards / Path(row['shard']).name
                mmarr = np.memmap(spath, mode='r+', shape=(n,d), dtype='float32')
                self.__embedding_matrix[offset:offset+n,...] = mmarr
                self.__members.append(mmarr)
                offset += n

    def close(self):
        if self.__open:
            self.keydb.close()
            del self.__members
            del self.__embedding_matrix
            self.__embedding_matrix = None
            self.__members = []
            self.__shape = None
            self.__open = False
    
    @property
    def embedding_matrix(self):
        return self.__embedding_matrix

    def get(self, key):
        """Retrieve embedding for the input key"""
        
        result = self.get_id(key)
        direction = self.get_direction(key)

        if result is None:
            raise ValueError(f"{key} not found")
        
        vector_id = result[0] if direction == 'forward' else result[1]
        return self.__embedding_matrix[vector_id]

    def get_direction(self, key):
        if isinstance(key, str):
            direction = 'backward'
        elif isinstance(key, int):
            direction = 'forward'
        else:
            raise TypeError(f"{type(key)} is not in [str,int]")
        return direction


    def get_id(self, key):
        direction = self.get_direction(key) 
        result = self.keydb.retrieve(key, direction=direction)
        if result is None:
            raise ValueError(f"{key} not found")
        return result


    def __getitem__(self, key):
        return self.get(key)

    def ids(self):
        """Yields all of the protein ids in the dataset"""
        yield from self.keydb.keys()

class TemporaryMemmap(np.memmap):
    """
    Extension of numpy memmap to automatically map to a file stored in temporary directory.
    Usefull as a fast storage option when numpy arrays become large and we just want to do some quick experimental stuff.

    [source
    https://stackoverflow.com/questions/24178460/in-python-is-it-possible-to-overload-numpys-memmap-
        to-delete-itself-when-the-m/24219413#24219413
    ]
    """
    def __new__(subtype, dtype='float32', mode='w+', offset=0,
                shape=None, order='C'):
        ntf = tempfile.NamedTemporaryFile()
        self = np.memmap.__new__(subtype, ntf, dtype, mode, offset, shape, order)
        self.temp_file_obj = ntf
        return self

    def __del__(self):
        if hasattr(self,'temp_file_obj') and self.temp_file_obj is not None:
            self.temp_file_obj.close()
            del self.temp_file_obj
