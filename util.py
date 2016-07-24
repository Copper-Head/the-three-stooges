from collections import defaultdict
from itertools import chain
import logging
import numpy
from theano import function
import warnings

logger = logging.getLogger(__name__)

class StateComputer(object):
    """
    Convenient interface to theano function for computing states/cells
    of a trained SequenceGenerator model.

    Expects a Model instance as argument. This model must also be created
    from a SequenceGenerator.cost application. Furthermore, expects a dict
    that maps from characters to indices.
    Extracts the model's inputs and auxiliary variables into attributes
    and creates theano function from inputs to auxiliary vars.

    Finally, provides a method that wraps the output of the theano function in
    a dict keyed by auxiliary variable labels.
    """

    def __init__(self, cost_model, map_char_to_ind):
        raw_state_vars = filter(self._relevant, cost_model.auxiliary_variables)
        self.state_variables = sorted(raw_state_vars, key=lambda var: var.name)
        self.state_var_names = [var.name for var in self.state_variables]
        self.inputs = sorted(cost_model.inputs, key=lambda var: var.name)
        self.func = function(self.inputs, self.state_variables)
        self.map_char_to_ind = map_char_to_ind
        self._prob_func = None

    def _relevant(self, aux_var):
        not_final_value = "final_value" not in aux_var.name
        cell_or_state = "states" in aux_var.name or "cells" in aux_var.name
        return cell_or_state and not_final_value

    def read_single_sequence(self, sequence):
        """
        Combines list of aux var values (output from theano func) with their
        corresponding labels.

        New and improved with (I think) more convenient interface.
        - as input provide a list (or something array-like, i.e. that can be
        converted to a numpy array) of either characters or their representing
        integers. If a list of characters is provided, the conversion is done
        in this method. NOTE that a string is more or less a list of characters
        for the purpose of this method =)
        - the theano function expects a 2D input where the first dimension is
        batch_size. This "fake" dimension is added below, so don't wrap the
        input sequence yourself.
        - because the cost was originally defined with a mask, the theano
        function needs one, as well. This mask is constructed below so you
        don't have to provide it yourself.
        """
        if all(isinstance(entry, str) for entry in sequence):
            indices = [self.map_char_to_ind[char] for char in sequence]
            converted_sequence = numpy.array([indices], dtype="int32")
        elif all(isinstance(entry, int) for entry in sequence):
            converted_sequence = numpy.array([sequence], dtype="int32")
        else:
            raise ValueError("Some or all sequence elements have invalid type "
                             "(should be str or int)!")
        mask = numpy.ones(converted_sequence.shape, dtype="int8")
        computed_aux_vars = self.func(converted_sequence, mask)
        flattened_aux_vars = map(drop_batch_dim, computed_aux_vars)
        return dict(zip(self.state_var_names, flattened_aux_vars))

    def read_sequence_batch(self, sequences, mask=None):
        """
        Basically read_single_sequence but for a whole batch to speed things up.

        Note that, because I find it unlikely that someone wants to pass in
        a batch of strings, this only works with sequences of integers atm.
        That is, the way the data is stored on disk.
        Mask can be computed beforehand, in which case sequences arg should
        be padded.
        If no mask is passed, we use pad_mask function to create it (and padding).
        """
        if mask is None:
            # tries to pad sequences as well
            sequences, mask = pad_mask(sequences)
        computed_states = self.func(sequences, mask)
        return dict(zip(self.state_var_names, computed_states))

    def compute_model_perplexity(self, test_sequences, prob_func_inputs=None, prob_func_outputs=None, mask=None):
        if mask is None:
            test_sequences, mask = pad_mask(test_sequences)
        if self._prob_func is None:
            if prob_func_inputs is None or prob_func_outputs is None: raise ValueError('Inputs needed for initializing probability function.')
            self.set_prob_func(prob_func_inputs, prob_func_outputs)
        elif prob_func_inputs or prob_func_outputs:
            logger.info('State computer already has a probability function, inputs argument will be ignored. You can overwrite it by calling set_prob_func.')
        probs = self._prob_func(test_sequences, mask).swapaxes(0, 1)  # sentence-dimension first
        seq_probs = []
        seq_pps = []
        for s_ix in range(len(probs)):
            rows = numpy.arange(len(test_sequences[s_ix]))
            elem_wise_probs = probs[s_ix][rows, test_sequences[s_ix]]
            s_prob = elem_wise_probs.prod()
            seq_probs.append(s_prob)
            seq_pps.append(s_prob ** (-1/len(test_sequences[s_ix])))
        return sum(seq_pps)/len(seq_pps)

    def set_prob_func(self, inputs, outputs):
        self._prob_func = function(inputs, outputs)


def drop_batch_dim(array_with_batch_dim):
    """When reading in one sentence at a time the batch dimension is superflous.

    Using numpy.squeeze we get rid of it. This relies on two assumptions:
    - that dimension being "1"
    - that being the second dimension (shape[1])
    """
    return numpy.squeeze(array_with_batch_dim, axis=1)


def pad_mask(batch):
    maxlen = max(len(example) for example in batch)  # clearer than example.shape[0]
    # For one, zero-pad the sequences
    # Also build the mask
    dims = (len(batch), maxlen)
    padded_seqs = numpy.zeros(dims, dtype="int32")
    # mask is int8 because apparently it's converted to float later,
    # and higher ints would complain about loss of precision
    mask = numpy.zeros(dims, dtype="int8")
    for (example_ind, example) in enumerate(batch):
        # we go through the sequences and simply put them into the padded array;
        # this will leave 0s wherever the sequence is shorter than maxlen.
        # similarly, mask will be set to 1 only up to the length of the respective sequence.
        # note that the transpose is done implicitly by essentially swapping indices
        padded_seqs[example_ind, :len(example)] = example
        mask[example_ind, :len(example)] = 1
    return padded_seqs, mask


def select_positions(aux_var_dict, indx=1):
    """Select certain indices from auxiliary var dictionary.

    Note that indx can be any slicing/indexing construct valid in numpy.
    """
    return {var_name: var_val[indx] for var_name, var_val in aux_var_dict.items()}

def mark_seq_len(seq):
    return numpy.arange(len(seq))


def mark_word_boundaries(seq):
    # define what's considered a word boundary. It's a set on purpose to permit
    # adding more character types as needed.
    wb = {
        " ",
        "\n"
    }
    return numpy.array([1 if char in wb else 0 for char in seq])

def mark_char_property(seq, bool_property_fun):
    """
    the func arg is supposed to be a str function, e.g.:
    str.isupper
    str.islower
    str.isalnum
    str.isalpha
    ...

    A complex lambda expression or wrapper function is also possible, of course
    :param seq:
    :param bool_property_fun:
    :return:
    """
    return numpy.array([1 if bool_property_fun(seq[i]) else 0 for i in range(len(seq))])


def mark_bracketing(seq, opening, closing, marking_fun, ignore_starts=None, **kwargs):
    """

    :param seq:
    :param opening:
    :param closing:
    :param marking_fun:
    :param ignore_starts: a set of start indexes to be ignored, occures rarely
    :param kwargs:
    :return:
    """
    ix = 0
    trigger = [opening, closing]
    collect = False
    ix_list = []
    ret_val = numpy.zeros(len(seq))
    while ix < len(seq):
        if seq[ix] == trigger[int(collect)] and not ix in ignore_starts:
            collect = not collect
            if not collect:
                ix_list.append(ix)
                ret_val[ix_list,] = marking_fun(ret_val, ix_list, **kwargs)
                ix_list = []
        if collect:
            ix_list.append(ix)
        ix += 1
    return ret_val


def filter_by_threshold(neuron_array, threshold=1):
    """Tells which neuron activations surpass a certain threshold.

    Args
    neuron_array: numpy array of neuron activation values
    threshold: numeric value by which to filter neurons.

    Returns: indices of neurons with activations greater than threshold.
    """
    return numpy.nonzero(neuron_array > threshold)


def unpack_value_lists(some_dict):
    """Pairs every key in some_dict with every item in its value (list)."""
    return ((key, v) for key in some_dict for v in some_dict[key])


def dependencies(dep_graph):
    """Turns nltk.parse.DependencyGraph into dict keyed by dependency labels.

    Returns dict that maps dependency labels to lists.
    Each list consists of pairs of (head word index, dependent word index).
    Here's an example entry:
    'DET' : [(9, 0), (4, 5)]
    """
    dep_dict = defaultdict(list)
    for index in dep_graph.nodes:
        node = dep_graph.nodes[index]
        for dep_label, dep_index in unpack_value_lists(node['deps']):
            dep_dict[dep_label].append((index - 1, dep_index - 1))
    return dep_dict

def simple_mark_dependency(dep_graph, dep_label):
    """Simple marking function for dependencies.

    Takes dictionary of dependencies and dependency label.
    Constructs a numpy array of zeros and marks with 1 the positions of words
    that take part in the dependency.
    """
    warnings.warn('Use mark_dependency with marking_fun='+MarkingFunctions.__name__+'.'+MarkingFunctions.simple_mark.__name__+' instead.', DeprecationWarning)
    dep_dict = dependencies(dep_graph)
    indeces = dep_dict[dep_label]
    unique_index_list = list(set(chain.from_iterable(indeces)))
    marked = numpy.zeros(len(dep_dict) - 1)
    marked[unique_index_list] = 1
    return marked


def mark_dependency(dep_graph, dep_label, prior_long=True, precomputed_dependencies=None, marking_fun=lambda x, ix: numpy.ones(len(ix)), **fun_kwargs):
    """
    This function marks a dependency like simple_mark_dependency, so head and dependent. But in addition
    the tokens between head and dependent are marked as well. The marking can be a simple sequence of 1s,
    which is the default, or a sequence given by a marking function. The marking function will be called
    for each sequence of tokens from head to dependent (or vice versa) SEPARATELY. The priorities for sequences
    of tokens including smaller sequences of tokens that also show the same dependency as in
            ______________________
           /        __            \
          /       /   \            \
        Das auf dem Berg stehende Haus

    is handle by :param prior_long:, which is True by default causing that the given sequence will be treated
    as a whole and there will be no marking call for the included. If you put :param prior_long: on False, "dem Berg"
    would be marked before the whole sequence.
    :param dep_graph:
    :param dep_label:
    :param prior_long:
    :param marking_fun:
    :param fun_args:
    :param fun_kwargs:
    :return:
    """
    raw_deps = precomputed_dependencies if not precomputed_dependencies is None else dependencies(dep_graph)[dep_label]  # note: if empty list provided instead of None, nothing is marked -> that's desired
    raw_indices = {frozenset(range(sorted(tpl)[0], sorted(tpl)[1]+1)) for tpl in raw_deps}
    filtered_indices = []
    if prior_long:
        for ixset in raw_indices:
            diffset = raw_indices.difference({ixset})
            if not [e for e in diffset if ixset.issubset(e)]:
                filtered_indices.append(sorted(list(ixset)))
    else:
        # create discontinuous and coniuous spans of indexes
        for ixset in raw_indices:
            diffset = raw_indices.difference({ixset})
            app_ixs = list(ixset)
            for subset in [e for e in diffset if e.issubset(ixset)]:
                for i in subset:
                    app_ixs.remove(i)
            filtered_indices.append(sorted(app_ixs))

    marked = numpy.zeros(len(dep_graph.nodes)-1)
    for ixlist in filtered_indices:
        marked[ixlist,] = marking_fun(marked, ixlist, **fun_kwargs)
    return marked


class MarkingFunctions(object):
    """
    This class is a collection of marking functions in the sense of functions to be used by
    the markers above (mark_dependency, mark_property) to mark a phenomenon of interest.
    """
    @staticmethod
    def rising_flank_linear(vec, ix_list):
        # TODO this is actually a special case of flank and should be removed in the future
        """
        this function only overwrites zeros
        """
        offset = 1/(len(vec[ix_list,])+1)
        raw_val = numpy.arange(offset, 1+offset, offset)
        return raw_val[-len(raw_val):]

    # might make more sense for longer sequences
    @staticmethod
    def flank(vec, ix_list, nonlinearity=lambda x: x, frm=0, to=1, coeff=1):
        return coeff*nonlinearity(numpy.arange(frm, to, abs((to-frm)/len(ix_list))))[-len(ix_list):]

    @staticmethod
    def simple_mark(vec, ixlist):
        ret_val = numpy.zeros(len(ixlist))
        ret_val[0] = 1
        ret_val[-1] = 1
        return ret_val
