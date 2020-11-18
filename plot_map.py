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
from adjacency import Composer, AdjacencyMatrixMaker

def arguments():
    parser = argparse.ArgumentParser("Invokes matshow on a distance or probability map.")
    parser.add_argument("input_protein",
                        type=Path,
                        help="input protein distance map file")

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
        adjmapper = AdjacencyMatrixMaker(threshold=args.t, weighted=args.weighted)
    else:
        adjmapper = lambda x: x
    
    mat = Composer(load_pt, adjmapper, to_numpy)(args.input_protein)
    
    fig, ax = plt.subplots(1)

    ax.matshow(mat, cmap="binary")
    print(args.input_protein.stem)
    
    plt.tight_layout()
    plt.savefig(args.output_file)



if __name__ == '__main__':
    pass

