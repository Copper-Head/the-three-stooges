import argparse
from fuel.datasets import H5PYDataset
import h5py
from numpy import array, save

PROGRESS_STEP = 10000

OUT_FILE_NAME = './data/lk.hdf5'
ALPHABET_FILE = './data/lk_ix2char.npy'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setsize', type=float, default=.95, help='Determines the proportion of training data, value between 0 and 1.')
    parser.add_argument('-f', '--file', type=str, default='./lk.data', help='The file to be converted.')
    args = parser.parse_args()

    training_size = args.setsize
    test_size = 1 - training_size

    data_location = args.file

    seq = []
    seqs = []
    char2int = {}
    ix2char = {}
    ix = 0
    di = 0  # debug
    with open(data_location) as f:
        c = '#'
        while c and di < 100:
            di += 1
            try:
                c = f.read(1)
            except UnicodeDecodeError:
                pass
            else:
                if c and not c in char2int:
                    char2int[c] = ix
                    ix2char[ix] = c
                    seq.append(ix)
                    ix += 1
                elif c:
                    seq.append(char2int[c])
            if len(seq) == 100:
                seqs.append(seq)
                seq = []

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
