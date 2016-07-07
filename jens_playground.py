import argparse
import cPickle

import numpy
from blocks.filter import VariableFilter
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from theano import function

from custom_blocks import PadAndAddMasks
from network import NetworkType, Network
from util import StateComputer, mark_seq_len, mark_word_boundaries


parser = argparse.ArgumentParser()
parser.add_argument("function", help="Correlation function to use.")
parser.add_argument("-p", "--pca", type=float, help="Whether to use PCA (and if so, how much variance explained?")
args = parser.parse_args()

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


if args.function == "len":
    corr_function = mark_seq_len_batch
elif args.function == "bound":
    corr_function = mark_word_boundaries_batch
else:
    raise ValueError("Invalid correlation function specified!")


map_chr_2_ind = cPickle.load(open("char_to_ind.pkl"))
map_ind_2_chr = cPickle.load(open("ind_to_char.pkl"))
lstm_net = Network(NetworkType.SIMPLE_RNN, len(map_ind_2_chr), hidden_dims=[1024])
lstm_net.set_parameters('seqgen_simple_1024.pkl')


# having a look at connectioneros from the cellsinas to the outputsos
params = lstm_net.cost_model.get_parameter_values()
for param in params:
    print param

# define a function that gets the overall "sum of scores" at a given time step
readouts = VariableFilter(theano_name="readout_readout_output_0")(lstm_net.cost_model.variables)[0]
score_function = function([lstm_net.x, lstm_net.mask], readouts.sum(axis=2))

# this section of the playground has some fun rides that revolve around various correlation stuff. uncomment to access
# =)
sc = StateComputer(lstm_net.cost_model, map_chr_2_ind)
# storage for the correlations at the very end
correlation_dict = dict()
for name in sc.state_var_names:
    correlation_dict[name] = numpy.zeros(lstm_net.hidden_dims[0], dtype="float32")

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
    state_super_dict[name] = numpy.empty(shape=(0, lstm_net.hidden_dims[0]), dtype="float32")
super_marker = numpy.empty(shape=(0,))

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
        seq_len_correlator = corr_function(seq_batch, mask_batch)
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

# optionally do PCA
if args.pca:
    print "APPLYING PCA..."
    for state_name in state_super_dict:
        pca = PCA(n_components=args.pca)  # not sure if we have to redefine it each time, and how long that takes...
        state_super_dict[state_name] = pca.fit(state_super_dict[state_name])
        print state_name, "got", pca.n_components_, "components!"
else:
    print "NO PCA USED!"

# do correlations between super long sequences...
for state_name in correlation_dict:
    for dim in xrange(correlation_dict[state_name].shape[0]):
        correlation_dict[state_name][dim] = pearsonr(state_super_dict[state_name][:, dim], super_marker)[0]
    print state_name
    print correlation_dict[state_name]
    print "LARGEST:", max(correlation_dict[state_name]), min(correlation_dict[state_name])
    print "\n\n"
