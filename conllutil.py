from nltk.parse.dependencygraph import DependencyGraph
from os import path
import re, sys

# TODO this class could actually implement a generator (for the trees)
class CoNLLData(object):

    def __init__(self, root_dir, file_list, tok_to_ix_map, min_len=0, max_len=sys.maxsize, lazy_loading=False, word_transform=lambda s: s):
        self._sequences = []
        punctuation = re.compile('\.|,|\?|!|:|;')
        for fname in file_list:
            with open(path.join(root_dir, fname)) as f:
                lines = f.readlines()
            sequence = []
            roots = 0
            drop_it = False
            for line in lines:
                line = line.strip()
                if line:
                    tpl = line.split('\t')[:8]
                    if tpl[-1] == 'S':
                        roots += 1
                    if word_transform(tpl[1]) in tok_to_ix_map:
                        sequence.append([tpl[0]]+[word_transform(tpl[1])]+tpl[2:])
                    elif punctuation.match(tpl[0][0]): drop_it = True
                else:
                    if not drop_it and roots == 1 and len(sequence) >= min_len and len(sequence) <= max_len:
                        self._sequences.append(sequence)
                    sequence = []
                    roots = 0
                    drop_it = False

        self._fix_indices()

        self._wordseqs = None if lazy_loading else [[tup[1] for tup in seq] for seq in self._sequences]
        self._posseqs = None if lazy_loading else [[tup[4] for tup in seq] for seq in self._sequences]

        if lazy_loading:
            self._word_transform = word_transform

        self.reset()

    def wordsequences(self):
        return self._wordseqs if self._wordseqs else [[self._word_transform(tup[1]) for tup in seq] for seq in self._sequences]

    def sequences(self):
        return self._sequences

    def possequences(self):
        return self._posseqs if self._posseqs else [[tup[4] for tup in seq] for seq in self._sequences]

    def _trees(self):
        for seq in self._sequences:
            gs = ''
            for nd in seq:
                gs += '\t'.join(nd+['_','_']) + '\n'
            try:
                yield DependencyGraph(gs, top_relation_label='S', cell_separator='\t')
            except UserWarning:
                yield None

    def tree(self):
        try:
            return next(self._tree)
        except StopIteration:
            return

    def trees(self):
        self.reset()
        return self._trees()

    def reset(self):
        self._tree = self._trees()

    def _fix_indices(self):
        for seq in self._sequences:
            expected_id = 1
            for elem in seq:
                if elem[0] != str(expected_id):
                    self._fix_index(seq, elem[0], str(expected_id))
                expected_id += 1

    def _fix_index(self, seq, i_old, i_new):
        for elem in seq:
            if elem[0] == i_old:
                elem[0] = i_new
            if elem[-2] == i_old:
                elem[-2] = i_new


