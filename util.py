import numpy
from theano import function


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

    def read_sequence_batch(self, sequences, mask):
        """
        Basically read_single_sequence but for a whole batch to speed things up.

        Note that, because I find it unlikely that someone wants to pass in a batch of strings, this only works with
        sequences of integers atm. That is, the way the data is stored on disk. Mask needs to be computed beforehand
        (easy via transformer).
        """
        computed_states = self.func(sequences, mask)
        return dict(zip(self.state_var_names, computed_states))


def drop_batch_dim(array_with_single_dims):
    """
    When reading in one sentence at a time the batch dimension is superflous.

    Using numpy.squeeze we get rid of it. This relies on that dimension being "1".
    Moreover, `squeeze` will remove ALL dimensions that are equal to 1, so be careful
    what you pass to this function.
    """
    return numpy.squeeze(array_with_single_dims)


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


def filter_by_threshold(neuron_array, threshold=1):
    """Tells which neuron activations surpass a certain threshold.

    Args
    neuron_array: numpy array of neuron activation values
    threshold: numeric value by which to filter neurons.

    Returns: indices of neurons with activations greater than threshold.
    """
    return numpy.nonzero(neuron_array > threshold)
