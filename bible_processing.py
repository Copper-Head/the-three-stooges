import re
import cPickle

import h5py
import numpy
from fuel.datasets.hdf5 import H5PYDataset


def find_verse_marker(line):
    # a verse marker is a number followed by a colon followed by another number
    verse_marker = re.compile(r"[0-9]+:[0-9]+")
    searcher = verse_marker.search(line)
    if searcher is not None:
        return searcher.start()
    return -1


def biblefile_to_hdf5(open_file):  # TODO REMOVE LINES WITH THE BOOK OF BLABLA
    """Everything in one function because we have variable-length sequences, so no intermediate arrays..."""
    char_to_ind = {"<S>": 0, "</S>": 1}
    current_char_ind = 2  # starts at 2 because 0, 1 are reserved for "end/start-of-sequence" character
    all_verses = []
    # TODO I still don't know what the readout initial_output really does; maybe we need to put <S> into every sequence
    current_verse = []
    for line in open_file:
        # first we need to check if a new verse begins somewhere in the line (not just beginning...)
        verse_marker_pos = find_verse_marker(line)
        if len(line.split()) > 0 and verse_marker_pos > -1:
            # if so, save the verse up to the verse marker and start a new one from the rest of the line
            current_verse += list(line[:verse_marker_pos])
            # also replace all characters by integers, creating more mappings if necessary
            for (ind, char) in enumerate(current_verse):
                if char not in char_to_ind:
                    char_to_ind[char] = current_char_ind
                    current_char_ind += 1
                current_verse[ind] = char_to_ind[char]
            current_verse.append(1)  # for sequence generator we need to explicitly append this end-of-sequence char
            all_verses.append(numpy.array(current_verse, dtype="int32"))
            current_verse = list(line[verse_marker_pos:])
        # otherwise, just put everything into the current verse
        else:
            current_verse += list(line)
    all_verses = numpy.array(all_verses)  # I think this conversion is necessary for the indexing below?

    # at this point we have all our verses =) now we build our .hdf5 dataset
    # make a little validation set
    val_indices = numpy.random.choice(a=len(all_verses), replace=False, size=1500)
    test_set = list(all_verses[val_indices])
    train_set = list(numpy.delete(all_verses, val_indices, 0))

    # if you don't get what's happening here, check the Fuel tutorial on variable-length data (only the 1D part)
    f = h5py.File(name="bible.hdf5", mode="w")
    dtype_varlen_int = h5py.special_dtype(vlen=numpy.dtype("int32"))
    character_seqs = f.create_dataset("character_seqs", (len(all_verses),), dtype=dtype_varlen_int)
    character_seqs[...] = train_set + test_set

    split_dict = {"train": {"character_seqs": (0, len(train_set))},
                  "valid": {"character_seqs": (len(train_set), len(all_verses))}}
    f.attrs["split"] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()

    # we also save the current_char_ind (equal to dimensionality of our one-hot character vectors) to a file
    numpy.save("onehot_size.npy", current_char_ind)
    # also the word-to-index dict
    cPickle.dump(char_to_ind, open("char_to_ind.pkl", mode="w"))
    # make a quick dirty reverse dict (actually a list) to map from indices to characters, so we can get readable output
    # later
    ind_to_char = [""]*len(char_to_ind)
    ind_to_char[0] = "<S>"
    ind_to_char[1] = "</S>"
    for char in char_to_ind:
        ind_to_char[char_to_ind[char]] = char
    cPickle.dump(ind_to_char, open("ind_to_char.pkl", mode="w"))


biblefile_to_hdf5(open("king_james.txt"))
