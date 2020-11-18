# Useful scripts

- "General" purpose scripts

# Requirements
```
# there's probably something unneccessary or missing here
pip install numpy scipy matplotlib torch Biopython 
```

# What's inside
- `mkdmap.py` - make a distance map from pdb file
- `split_fasta.py` - split and/or filter sequences by length from a fasta file
- `plot_map.py` - plots a contact map

- potentially more to come


# Usage examples
```
# filter a fasta so all sequences are have length between 100 and 300, output the results
python split_fasta.py -i samples/multiple_sequences.fasta -o test.fsa --assert "<300" ">100" --filter
```

