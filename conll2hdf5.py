import argparse
from itertools import chain
from numpy import save, array, unique
from os import listdir
from os import path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, help='Source directory to conll-data.')
parser.add_argument('-n', '--name', type=str, help='Specifies the corpus name.')

args = parser.parse_args()

target_file = './data/'+args.name+'_.hdf5'
alphabet_file = './data/'+args.dir+'_ix2tok.npy'

ix = 0
tok2ix = {}
seqs = []
files = filter(lambda f: f.endswith('.conll'), listdir(args.dir))
for fname in files:
    with open(path.join(args.dir, fname)) as f:
        data = f.read()
    for sentence in data.split('\n\n'):
        seq = []
        for line in sentence.split('\n'):
            tok = line.split('\t')[1]
            if tok in tok2ix:
                seq.append(tok2ix[tok])
            else:
                tok2ix[tok] = ix
                seq.append(ix)
                ix += 1
        seqs.append(seq)

seq_arr = array(seqs)
# save as hdf5
print(unique(list(chain(*seqs))).size)

exit()
# save vocab
ix2tok = {v: k for k, v in tok2ix.items()}
save(array(ix2tok))