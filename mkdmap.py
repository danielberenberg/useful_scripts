#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates a PyTorch distance map from PDB file
"""

import re
import json
import gzip
import argparse
import warnings
import itertools
import functools
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

from biotoolbox.structure_file_reader import build_structure_container_for_pdb
from biotoolbox.contact_map_builder   import DistanceMapBuilder

def make_distance_map(pdbfile, gzip_compressed=False, atom="CA"):
    """
    Generate (diagonalized) atomic distance matrix from a pdbfile 

    args:
        :pdbfile (str or Path)  - path to structure file
        :gzip_compressed (bool) - file is gzip compressed
        :atom (str)             - atom name to generate distance map
    """
    assert atom in ["CA","CB"], f'Unrecognized atom: {atom}'

    if gzip_compressed:
        opener = functools.partial(gzip.open, mode='rb')
    else:
        opener = functools.partial(open, mode='r')

    with opener(pdbfile) as pdb_handle:
        pdb_raw = pdb_handle.read()
        if hasattr(pdb_raw, 'decode'):
            pdb_raw = pdb_raw.decode()

        structure_container = build_structure_container_for_pdb(pdb_raw) 
        
        mapper = DistanceMapBuilder(atom=atom, glycine_hack=-1, verbose=False) # get distances
        map_ = mapper.generate_map_for_pdb(structure_container) 
    return map_.chains

def arguments():
    parser = argparse.ArgumentParser(description="Save PDB file(s) as distance matrices")
    parser.add_argument("input_pdb",
                        type=Path,
                        help="Input pdbfile")
    
    parser.add_argument("output_pt",
                        type=Path,
                        help="Output PyTorch tensor file")

    parser.add_argument("--xyz",
                        action='store_true',
                        help="Rather than saving the distance map, save the (x,y,z) coordinates of each CA atom")

    parser.add_argument("-atom",
                        choices=["CA", "CB"],
                        default="CA",
                        help="Atom type")

    return parser.parse_args()

def write_tensor(filename, tensor):
    torch.save(torch.from_numpy(tensor), filename)

def filter_map_output(chaindict):
    output_dict = defaultdict(dict)
    for chain in chaindict:
        map_ = chaindict[chain]
        if 'final-seq' not in map_:
            map_['final-seq']  = map_['seq']

        output_dict[chain]['method']      = map_['method']
        output_dict[chain]['seq']         = str(map_['seq'])
        output_dict[chain]['final-seq']   = str(map_['final-seq'])
        output_dict[chain]['contact-map'] = map_['contact-map'].tolist()
        output_dict[chain]['xyz'] = map_['xyz']
    return dict(output_dict)

if __name__ == '__main__':
    args = arguments()
    
    atom = args.atom
    pdb  = args.input_pdb
    pt   = args.output_pt

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dmap_info = filter_map_output(make_distance_map(pdb, gzip_compressed=False, atom=atom))
    
    # extract only the first contact map for a specific chain!!!!!!
    # needs to be changed to emit all chains ...
    chain = list(dmap_info.keys())[0]
    dmap = np.array(dmap_info[chain]['contact-map'])
    xyz  = np.array(dmap_info[chain]['xyz'])

    save_tensor = xyz if args.xyz else dmap
    
    write_tensor(pt, save_tensor)
    
    print(f"{pdb} -> {pt} ({'xyz' if args.xyz else 'dmap'})")

