import cPickle

import numpy
from scipy.stats import pearsonr

from network import NetworkType, Network
from util import StateComputer, mark_seq_len


lstm_net = Network(NetworkType.LSTM)
lstm_net.set_parameters('seqgen_lstm_512_512_512.pkl')
map_chr_2_ind = cPickle.load(open("char_to_ind.pkl"))

sc = StateComputer(lstm_net.cost_model, map_chr_2_ind)

verse = "1:7 And God made the firmament, and divided the waters which were " \
        "under the firmament from the waters which were above the firmament: " \
        "and it was so."
seq_len_correlator = mark_seq_len(verse)

state_seqs = sc.read_single_sequence(verse)
corrs_with_length = dict()
for state in state_seqs:
    single_state_seq = state_seqs[state]  # seq_len x 512 array
    corrs = numpy.zeros(single_state_seq.shape[1])
    for dim in xrange(single_state_seq.shape[1]):
        corrs[dim] = pearsonr(single_state_seq[:, dim], seq_len_correlator)[0]
    print "FOR THIS STATE", state, "WE GOT:"
    print corrs
    raw_input()
