import argparse
from fuel.datasets import H5PYDataset
import h5py
from numpy import array, save
import re
from os.path import exists

import dataproc

PROGRESS_STEP = 10000

OUT_FILE_NAME = './data/lk.hdf5'
RAW_DATA_OUT_NAME = './data/lk_cleaned.data'
ALPHABET_FILE = './data/lk_ix2char.npy'

comment_pattern = re.compile('//|/\*')


def get_raw_data(file_path):
    with open(file_path) as f:
        c = '#'  # dummy value, not used
        trigger = ''
        read = 1
        raw = ''
        while c:
            try:
                c = f.read(1)
            except UnicodeDecodeError:
                pass
            else:
                if read:
                    if c == '/':
                        c += f.read(1)
                        if comment_pattern.match(c):
                            read = 0
                            trigger = c
                    else:
                        raw += c
                else:
                    if trigger == '//' and c == '\n':
                        read = 1
                        raw += c
                    elif trigger == '/*' and c == '*':
                        c += f.read(1)
                        if c == '*/':
                            read = 1
    return raw

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setsize', type=float, default=.95, help='Determines the proportion of training data, value between 0 and 1.')
    parser.add_argument('-f', '--file', type=str, default='', help='The file to be converted.')


    args = parser.parse_args()

    training_size = args.setsize
    test_size = 1 - training_size

    if not args.file:
        with open(RAW_DATA_OUT_NAME) as f:
            raw = f.read()
    else:
        raw = get_raw_data(args.file)
        with open(RAW_DATA_OUT_NAME, 'w') as f:
            f.write(raw)

    seq = []
    seqs = []
    char2int = {}
    ix2char = {}
    ix = 0
    upper_limit = int((len(raw) * 15000000)/500000000)
    min_len = 1500

    batch_size = 10000

    seqs = []
    p_old = 0
    step = int(upper_limit/batch_size)

    for b in range(0, upper_limit, step):
        seq = raw[b:b+step]
        seq_ix = []
        for c in seq:
            if c in char2int:
                seq_ix.append(char2int[c])
            else:
                char2int[c] = ix
                seq_ix.append(ix)
                ix += 1
        seqs.append(seq_ix)

    data = array(seqs)

    split_n = int(training_size * data.shape[0])

     # write hdf5

    f = h5py.File(name=OUT_FILE_NAME, mode='w')
    character_seqs = f.create_dataset("character_seqs", data.shape, dtype='int32')
    character_seqs[...] = data

    split_dict = {"train": {"character_seqs": (0, split_n)},
                  "valid": {"character_seqs": (split_n, data.shape[0])}}

    f.attrs["split"] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()

    # store alphabet
    alphabet = array(ix2char)
    save(ALPHABET_FILE, alphabet)


