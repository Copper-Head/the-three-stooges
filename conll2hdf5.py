import argparse
from collections import Counter
from itertools import chain
from numpy import save, array
from os import listdir
from os import path

import dataproc as dp

LIMIT = 5

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, help='Source directory to conll-data.')
parser.add_argument('-n', '--name', type=str, help='Specifies the corpus name.')

args = parser.parse_args()

target_file = './data/'+args.name+'_.hdf5'
alphabet_file = './data/'+args.dir+'_ix2tok.npy'

seqs = []
files = filter(lambda f: f.endswith('.conll'), listdir(args.dir))
for fname in files:
    with open(path.join(args.dir, fname)) as f:
        data = f.read()
    for sentence in data.split('\n\n'):
        seq = []
        for line in sentence.split('\n'):
            if line.strip():
                seq.append(line.split('\t')[1])
        seqs.append(seq)

counter = Counter(list(chain(*seqs)))
ix_seq = []
ix_seqs = []
tok2ix = {'<UNKNOWN>': 0}
ix = 1
for seq in seqs:
    for tok in seq:
        if counter[tok] < LIMIT:
            ix_seq.append(0)
        else:
            if tok in tok2ix:
                ix_seq.append(tok2ix[tok])
            else:
                tok2ix[tok] = ix
                ix_seq.append(ix)
                ix += 1
    ix_seqs.append(ix_seq)
    ix_seq = []

seq_arr = array(ix_seqs)

# save sentences
split_n = int(.9*seq_arr.shape[0])
dp.split_hdf5_file(target_file, seq_arr[:split_n], seq_arr[split_n:], varlen=True)

# save vocab indexing
ix2tok = {v: k for k, v in tok2ix.items()}
save(array(ix2tok))