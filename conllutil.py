from os import path
import sys

class CoNLLData(object):

    def __init__(self, root_dir, file_list, tok_to_ix_map, min_len=0, max_len=sys.maxsize, lazy_loading=False, word_transform=lambda s: s):
        self._sequences = []
        for fname in file_list:
            with open(path.join(root_dir, fname)) as f:
                lines = f.readlines()
            sequence = []
            for line in lines:
                line = line.strip()
                if line:
                    tpl = line.split('\t')[:8]
                    if word_transform(tpl[1]) in tok_to_ix_map: sequence.append(tuple([tpl[0]]+[word_transform(tpl[1])]+tpl[2:]))
                else:
                    if len(sequence) >= min_len and len(sequence) <= max_len:
                        self._sequences.append(sequence)
                    sequence = []

        # TODO: Remove ID, because we can just say head-id -1 = index in list
        self._wordseqs = None if lazy_loading else [[tup[1] for tup in seq] for seq in self._sequences]
        self._posseqs = None if lazy_loading else [[tup[4] for tup in seq] for seq in self._sequences]

        if lazy_loading:
            self._word_transform = word_transform

    def wordsequences(self):
        return self._wordseqs if self._wordseqs else [[self._word_transform(tup[1]) for tup in seq] for seq in self._sequences]

    def sequences(self):
        return self._sequences

    def possequences(self):
        return self._posseqs if self._posseqs else [[tup[4] for tup in seq] for seq in self._sequences]

