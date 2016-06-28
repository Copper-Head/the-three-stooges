import argparse
from collections import Counter
from itertools import chain
from numpy import save, array
from os import listdir
from os import path
import sys

import dataproc as dp

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, help='Source directory to conll-data.')
parser.add_argument('-n', '--name', type=str, help='Specifies the corpus name.')
parser.add_argument('-l', '--maxlength', type=int, default=sys.maxsize, help='Max sentence length to avoid memory errors.')
parser.add_argument('-w', '--wordfreq', type=int, default=5, help='Minimum frequence of words, words with lower'
                                                                  +' frequence will be replaced by <UNKNOWN>. Default=5')
parser.add_argument('-c', '--case', type=int, default=1, help='Determines, if the vocabulary should be case sensitive. It is on per default, 0 means non-case sensitive.')

args = parser.parse_args()

target_file = './data/'+args.name+'_data.hdf5'
alphabet_file = './data/'+args.name+'_ix2tok.npy'
len_limit = args.maxlength
word_freq_limit = args.wordfreq

EOS = '<EOS>'

if args.case:
    def transform(s):
        return s
else:
    def transform(s):
        return s.lower()

seqs = []
drop_seqs = []
files = filter(lambda f: f.endswith('.conll'), listdir(args.dir))
for fname in files:
    with open(path.join(args.dir, fname)) as f:
        data = f.read()
    for sentence in data.split('\n\n'):
        seq = []
        for line in sentence.split('\n'):
            if line.strip():
                seq.append(transform(line.split('\t')[1]))
        if len(seq) <= len_limit:
            seq.append(EOS)
            seqs.append(seq)
        else:
            drop_seqs.append(seq)

print('Dropping', len(drop_seqs), 'sentences containing', len(list(chain(*drop_seqs))), 'tokens.')
all_words = list(chain(*seqs))

print(len(seqs), 'sentences with', len(all_words), 'tokens remaining.')

counter = Counter(all_words)
ix_seq = []
ix_seqs = []
tok2ix = {'<UNKNOWN>': 0}
ix = 1
for seq in seqs:
    for tok in seq:
        if counter[tok] < word_freq_limit:
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

print('Vocabulary size:', len(tok2ix))
# save vocab indexing
ix2tok = {v: k for k, v in tok2ix.items()}
save(alphabet_file, array(ix2tok))