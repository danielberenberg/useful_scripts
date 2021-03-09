#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# plot_map.py
# author: dan berenberg

"""
Visualize one contact map.
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# see adjacency.py
from .biotoolbox.adjacency import Composer, AdjacencyMatrixMaker, CoordLoader

def arguments():
    parser = argparse.ArgumentParser("Invokes matshow on a distance or probability map.")
    parser.add_argument("input_protein",
                        type=Path,
                        help="input protein distance map/3d coordinate file")

    parser.add_argument("output_file",
                        type=Path,
                        help="Output filename")

    parser.add_argument("-t", "--threshold",
                        type=float, dest='t',
                        help="Distance threshold, if any. Absence implies no thresholding.")

    return parser.parse_args()

def to_numpy(tnsr):
    return tnsr.numpy()

def load_pt(filename):
    return torch.load(filename, map_location=torch.device("cpu"))

if __name__ == '__main__':
    args = arguments()
    if args.t is not None:
        adjmapper = AdjacencyMatrixMaker(args.t)
    else:
        adjmapper = lambda x: x
    
    mat = Composer(load_pt, CoordLoader(silent_if_square=True), adjmapper, to_numpy)(args.input_protein)
    
    fig, ax = plt.subplots(1)
    
    if args.t is None:
        ax.matshow(mat, cmap="RdYlBu")
    else:
        ax.matshow(mat, cmap="binary")

    print(args.output_file)
    
    plt.tight_layout()
    plt.savefig(args.output_file)



if __name__ == '__main__':
    pass

