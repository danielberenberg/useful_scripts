#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Split or filter a 
"""

import io
import re
import sys
import pathlib
import secrets 
import operator
import argparse
import textwrap
import itertools

clear = f"\r{100 * ' '}\r"
FASTA_STOP_CODON = '*'

def _valid_condition(cond):
    condition_structure = "(<|>|<=|>=)(\d+)" 
    match = re.match(condition_structure, cond)
    if match:
        op, operand = match.groups()
        return op, int(operand)
    else:
        raise ValueError(f"Invalid condition: {cond}")

def _construct_conditional(conditions):
    if not conditions:
        return lambda x: True, []
    conditionals = []
    op2op = {'>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le} 
    operations, operands = zip(*conditions)
    operations = list(map(lambda op: op2op[op], operations))
    operands   = list(map(int, operands))

    def conditional(X):
        evaluated = [op(X, operand) for op, operand in zip(operations, operands)]
        return all(evaluated)
    return conditional, conditionals

def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", help="Input filename", type=pathlib.Path, default=None, metavar="INPUT")
    parser.add_argument("-o", help="Output path", type=pathlib.Path, default="fastas", metavar="OUTPUT")
    parser.add_argument("-s", "--include-stops", help="Include sequences with stop codons",
                        default=False, action='store_true', dest='allow_stop_codons')
    parser.add_argument("--filter", dest="filter_only", action='store_true', default=False)
    parser.add_argument("--assert", dest='assertion',
                        type=_valid_condition,
                        nargs='+',
                        help="Condition for sequences of the form '[>|<|>=|<=]\d+'",
                        default=None)

    args = parser.parse_args()
    args.condition, _ = _construct_conditional(args.assertion)
    if not args.o.exists() and not args.filter_only:
        args.o.mkdir(parents=True)

    if args.i is not None and not args.i.exists():
        raise FileNotFoundError(f"{args.i} doesn't exist")
    else:
        args.i = (sys.stdin if args.i is None else open(args.i, 'r'))

    return args

def fasta_reader(handle, width=None):
    """
    Reads a FASTA file, yielding header, sequence pairs for each sequence recovered
    args:
        :handle (str, pathliob.Path, or file pointer) - fasta to read from
        :width (int or None) - formats the sequence to have max `width` character per line.
                               If <= 0, processed as None. If None, there is no max width.
    yields:
        :(header, sequence) tuples
    returns:
        :None
    """
    FASTA_STOP_CODON = "*"
    import io, textwrap, itertools

    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    width  = width if isinstance(width, int) and width > 0 else None
    try:
        for is_header, group in itertools.groupby(handle, lambda line: line.startswith(">")):
            if is_header:
                header = group.__next__().strip()
            else:
                seq    = ''.join(line.strip() for line in group).strip().rstrip(FASTA_STOP_CODON)
                if width is not None:
                    seq = textwrap.fill(seq, width)
                yield header, seq
    finally:
        if not handle.closed:
            handle.close()

if __name__ == "__main__":
    args = arguments()
    spliterator = fasta_reader(args.i)
    if not args.allow_stop_codons:
        spliterator = itertools.filterfalse(lambda tup: FASTA_STOP_CODON in tup[1], spliterator)

    dumped = 0
    total = 0
    if args.filter_only:
        outfile = open(args.o, 'w')
        def emit_outfile(outpath, ID):
            return outfile 
    else:
        def emit_outfile(outpath, ID):
            filename = outpath / (ID + '.fasta')
            i = 1
            while filename.exists():
                filename = outpath / ID + f'.{i}.fasta'
                i += 1
            return open(filename, 'w')

    for i, (header, sequence) in enumerate(spliterator, 1):
        total += 1
        if not args.condition(len(sequence.replace('\n', ''))):
            continue
        ID = header.lstrip(">").rstrip().split(" ")[0].replace("/", "-").replace("|", "__")
        outfile = emit_outfile(args.o, ID)
        print(header, file=outfile)
        print(sequence, file=outfile)
        print(f"{clear}[{i}] {outfile.name}", end='', flush=True)
        if not args.filter_only:
            outfile.close()
        dumped += 1

    if args.filter_only:
        outfile.close()

    print(f"{clear}Done! Dumped {dumped}/{total} separate fasta files into {args.o}.")
