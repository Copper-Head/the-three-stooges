import cPickle

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
import numpy
from scipy.stats import pearsonr

from custom_blocks import PadAndAddMasks
from network import NetworkType, Network
from util import StateComputer, mark_seq_len, mark_word_boundaries


def mark_seq_len_batch(seq_batch, mask_batch):
    # get markers separately, then reshape
    padded_markers = numpy.array([numpy.arange(len(seq)) for seq in seq_batch])
    padded_markers = padded_markers.flatten(order="C")
    # throw away padding
    return padded_markers[mask_batch.flatten(order="C") == 1]


def mark_word_boundaries_batch(seq_batch, mask_batch):
    padded_markers = numpy.array([mark_word_boundaries([map_ind_2_chr[ind] for ind in seq]) for seq in seq_batch])
    padded_markers = padded_markers.flatten(order="C")
    return padded_markers[mask_batch.flatten(order="C") == 1]


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
"""
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
                state_seq = state_seq[mask == 1, :]  # throw away padding
                # now get a marker and compute separately the correlation of each state seq with the marker seq
                seq_len_correlator = mark_word_boundaries([map_ind_2_chr[ind] for ind in seq_batch[sequence_ind, mask == 1]])
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

state_super_dict = dict()
for name in sc.state_var_names:
    state_super_dict[name] = numpy.empty(shape=(0, 512))  # TODO NUMBER OF CELLS NOT VERY GENERAL
super_marker = numpy.empty(shape=(0,))

prediction_alignment = True
try:
    while iterator:
        seq_batch, mask_batch = next(iterator)
        if not prediction_alignment:
            # "remove" last element of each sequence by modifying mask
            mask_batch[numpy.arange(mask_batch.shape[0]), mask_batch.sum(axis=1) - 1] = 0
        state_batch_dict = sc.read_sequence_batch(seq_batch, mask_batch)
        # reshape mask: we only need to do this once per batch, not again for each state_type
        # mask is in shape batch_size x seq_len, so NOT transposed, so it is flattened in C order
        mask_reshaped = mask_batch.flatten(order="C")
        # get marker (very preliminary...)
        seq_len_correlator = mark_word_boundaries_batch(seq_batch, mask_batch)
        super_marker = numpy.append(super_marker, seq_len_correlator)
        for state_type in state_batch_dict:
            state_batch = state_batch_dict[state_type]
            if not prediction_alignment:
                # "throw away" initial state by rolling array backwards -- hacky, but sidesteps problems with needing
                # different masks for the sequences (the modified one further above) and for states (the "regular" one)
                state_batch = numpy.roll(state_batch, shift=-1, axis=0)
            # note: order of reshape is Fortran because states are "transposed" into seq_len x batch_size x dim
            state_reshaped = state_batch.reshape((state_batch.shape[0]*state_batch.shape[1], state_batch.shape[2]),
                                                 order="F")
            # throw away padding
            state_reshaped = state_reshaped[mask_reshaped == 1, :]
            # vstack will be very slow; here's hoping the relatively small number of operations will make it bearable...
            state_super_dict[state_type] = numpy.vstack((state_super_dict[state_type], state_reshaped))
        print "MADE IT THROUGH BATCH"
except StopIteration:
    pass
# do correlations between super long sequences...
for state_name in correlation_dict:
    for dim in xrange(correlation_dict[state_name].shape[0]):
        correlation_dict[state_name][dim] = pearsonr(state_super_dict[state_name][:, dim], super_marker)[0]
    print state_name
    print correlation_dict[state_name]
    print "LARGEST:", max(correlation_dict[state_name]), min(correlation_dict[state_name])
    print "\n\n"
