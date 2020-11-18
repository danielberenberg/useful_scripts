#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

if __name__ == '__main__':
    pass

