import cPickle

import numpy
from blocks.filter import VariableFilter
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from scipy.stats import pearsonr
from theano import function

from custom_blocks import PadAndAddMasks
from network import NetworkType, Network
from util import StateComputer, mark_seq_len, mark_word_boundaries


numpy.set_printoptions(precision=8, suppress=True)


# TODO get these into util.py, maybe in a prettier form
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


def mark_letter(seq_batch, mask_batch, letter):
    padded_markers = 1*numpy.array([[map_ind_2_chr[char] == letter for char in seq] for seq in seq_batch])
    padded_markers = padded_markers.flatten(order="C")
    return padded_markers[mask_batch.flatten(order="C") == 1]


map_chr_2_ind = cPickle.load(open("char_to_ind.pkl"))
map_ind_2_chr = cPickle.load(open("ind_to_char.pkl"))

lstm_net = Network(NetworkType.LSTM, input_dim=len(map_ind_2_chr), hidden_dims=[512, 512, 512])
lstm_net.set_parameters('seqgen_lstm_512_512_512.pkl')


# having a look at connectioneros from the cellsinas to the outputsos
params = lstm_net.cost_model.get_parameter_values()
for param in params:
    print param


# this section deals with prediction probabilities
"""
readouts = VariableFilter(theano_name="readout_readout_output_0")(lstm_net.cost_model.variables)[0]
char_probs = lstm_net.generator.readout.emitter.probs(readouts)

prob_function = function([lstm_net.x, lstm_net.mask], char_probs)

lord_original = "3:15 And the LORD came."
lord = [map_chr_2_ind[char] for char in lord_original]
print lord
zaza = prob_function([lord], numpy.ones((1, len(lord)), dtype="int8"))[:, 0, :]
print zaza
print zaza.shape
for (ey, row) in enumerate(zaza):
    print "PREDICTION PROBABILITIES FOR POSITION", ey, "LETTER", repr(lord_original[ey])
    sorted_thing = [(prob, ind) for (ind, prob) in enumerate(row)]
    sorted_thing.sort(reverse=True)
    for (prob, ind) in sorted_thing:
        print repr(map_ind_2_chr[ind]), ":", prob
    print "\n"
"""

# define a function that gets the overall "sum of scores" at a given time step
readouts = VariableFilter(theano_name="readout_readout_output_0")(lstm_net.cost_model.variables)[0]
score_function = function([lstm_net.x, lstm_net.mask], readouts.sum(axis=2))

# this section of the playground has some fun rides that revolve around various correlation stuff. uncomment to access
# =)
sc = StateComputer(lstm_net.cost_model, map_chr_2_ind)
# storage for the correlations at the very end
correlation_dict = dict()
for name in sc.state_var_names:
    correlation_dict[name] = numpy.zeros(lstm_net.hidden_dims[0], dtype=float)

# get validation data to run over
valid_data = H5PYDataset("bible.hdf5", which_sets=("valid",), load_in_memory=True)
data_stream = PadAndAddMasks(
    DataStream.default_stream(dataset=valid_data, iteration_scheme=SequentialScheme(valid_data.num_examples,
                                                                                    batch_size=128)),
    produces_examples=False)
iterator = data_stream.get_epoch_iterator()

# storage for the "supersequences" concatenated over all sequences
state_super_dict = dict()
for name in sc.state_var_names:
    state_super_dict[name] = numpy.empty(shape=(0, lstm_net.hidden_dims[0]))
super_marker = numpy.empty(shape=(0,))

# storage for the connections from states to output (softmax)
# this later allows easier connection between each layer's states and the corresponding output connection
connection_dict = dict()
standard_name = "/sequencegenerator/readout/merge/transform_states"
for name in sc.state_var_names:
    if name[-1] == "2":
        name_here = standard_name + "#2.W"
    elif name[-1] == "1":
        name_here = standard_name + "#1.W"
    else:
        name_here = standard_name + ".W"
    connection_dict[name] = params[name_here][:, map_chr_2_ind["O"]]

# if this is true, each state will be aligned with the character (or event derived from it) that it is used to *predict*
# if false, each state will be aligned with the character that was most recently read
prediction_alignment = False
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
        seq_len_correlator = mark_letter(seq_batch, mask_batch, letter="L")
        super_marker = numpy.append(super_marker, seq_len_correlator)
        # TESTING total score thingy -- should be 2D, seq_len x batch_size
        #overall_scores = score_function(seq_batch, mask_batch)
        for state_type in state_batch_dict:
            state_batch = state_batch_dict[state_type]
            if not prediction_alignment:
                # "throw away" initial state by rolling array backwards -- hacky, but sidesteps problems with needing
                # different masks for the sequences (the modified one further above) and for states (the "regular" one)
                state_batch = numpy.roll(state_batch, shift=-1, axis=0)
            #state_batch *= connection_dict[state_type][None, None, :]
            #state_batch /= overall_scores[:, :, None]
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
