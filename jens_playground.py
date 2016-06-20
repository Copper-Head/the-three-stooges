import cPickle

import numpy

from network import NetworkType, Network
from util import StateComputer


lstm_net = Network(NetworkType.LSTM)
lstm_net.set_parameters('./seqgen_lstm.pkl')
map_chr_2_ind = cPickle.load(open("char_to_ind.pkl"))

sc = StateComputer(lstm_net.cost_model, map_chr_2_ind)

verse = "1:7 And God made the firmament, and divided the waters which were " \
        "under the firmament from the waters which were above the firmament: " \
        "and it was so."

for pair in map_chr_2_ind.items():
    print pair
