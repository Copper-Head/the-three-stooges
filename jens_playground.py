import cPickle

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
import numpy
from scipy.stats import pearsonr

from custom_blocks import PadAndAddMasks
from network import NetworkType, Network
from util import StateComputer, mark_seq_len, mark_word_boundaries


lstm_net = Network(NetworkType.LSTM)
lstm_net.set_parameters('seqgen_lstm_512_512_512.pkl')
map_chr_2_ind = cPickle.load(open("char_to_ind.pkl"))
map_ind_2_chr = cPickle.load(open("ind_to_char.pkl"))

sc = StateComputer(lstm_net.cost_model, map_chr_2_ind)
correlation_dict = dict()
for name in sc.state_var_names:
    correlation_dict[name] = numpy.zeros(512, dtype=float)  # TODO NOT VERY GENERAL!!!


valid_data = H5PYDataset("bible.hdf5", which_sets=("valid",), load_in_memory=True)
data_stream = PadAndAddMasks(
    DataStream.default_stream(dataset=valid_data, iteration_scheme=SequentialScheme(valid_data.num_examples,
                                                                                    batch_size=128)),
    produces_examples=False)
iterator = data_stream.get_epoch_iterator()
try:
    while iterator:
        seq_batch, mask_batch = next(iterator)
        state_batch_dict = sc.read_sequence_batch(seq_batch, mask_batch)
        # unfortunately we don't have a "batched correlation", and padding could lead to problems here, so one by one...
        # first iterate over the different layers and states/cells
        for state_type in state_batch_dict:
            state_batch = state_batch_dict[state_type]
            # and then over the different sequences in the batch (always remember, axis 1, not 0)
            for sequence_ind in xrange(state_batch.shape[1]):
                state_seq = state_batch[:, sequence_ind, :]
                mask = mask_batch[sequence_ind, :]  # mask is NOT transposed!!
                state_seq = state_seq[mask==1, :]  # throw away padding
                # now get a marker and compute separately the correlation of each state seq with the marker seq
                seq_len_correlator = mark_word_boundaries([map_ind_2_chr[ind] for ind in seq_batch[sequence_ind]])
                for dim in xrange(state_seq.shape[1]):
                    correlation_dict[state_type][dim] += pearsonr(state_seq[:, dim], seq_len_correlator)[0]
        print "MADE IT THROUGH BATCH"
except StopIteration:
    pass
# at the very end, we need to divide all our summed up correlations by the number of sequences to get the average
for state_name in correlation_dict:
    correlation_dict[state_name] /= valid_data.num_examples
    print state_name
    print correlation_dict[state_name]
    print "LARGEST:", max(correlation_dict[state_name]), min(correlation_dict[state_name])
    print "\n\n"

"""
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
    print "LARGEST:", max(corrs), "(index:", numpy.argmax(corrs), ");", min(corrs), "(index:", numpy.argmin(corrs), ")"
    raw_input()
"""