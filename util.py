from theano import function


def get_state_computer(cost_model):
    """Everything you need to know about this: Expects as input a model which was built from a SequenceGenerator's cost
    function. Output is a theano function that you can put in a (batch of) sequences and puts out a list with all of the
    network's states/cells. This means - put the character sequence in and you get the full state sequences in one go!
    NOTES:
    - output is sorted alphabetically. That is, if you're using an LSTM first will be all the cells, then states. Cells/
    states are ordered by layer (no suffix is first layer, #1 is second, etc.).
    - because the cost was originally defined with a mask, you also need to provide one when using this function. If
    you're just putting in a single sequence, you can just put in a "dummy mask" that is 1 everywhere, with size equal
    to the sequence.
    - input is provided in the form batch_size x seq_len. i.e. if your sequence is "I am so cool", provide a numpy array
    based on the following list: [["I", " ", "a", "m", ..., "o", "o", "l"]] (of course you actually put in the indices,
    not the characters) as well as a mask of numpy.ones of the same shape
    - input is ordered (character_sequence, mask)
    - on the output: the first state of each sequence is the *initial state* before the network has seen any input. The
    very last state (after seeing the last input) is not included because it is not used for predictions! This means:
    The output states sequences will have the same length as the input sequence. but they will be "aligned" such that
    the ith state is not the *result* of the ith character, but is the state that is *used to predict* the ith char!
    - as said, the input is to be provided as batch_size x seq_gen. But the output will "of course" be in the format
    seq_gen x batch_size x state_dim. So be mindful to swapaxes(0, 1) of either input or output before trying to align
    them!
    """
    state_variables = []
    for var in cost_model.auxiliary_variables:
        if ("states" in var.name or "cells" in var.name) and "final_value" not in var.name:
            state_variables.append(var)
    state_variables.sort(key=lambda var: var.name)

    inputs = cost_model.inputs
    inputs.sort(key=lambda var: var.name)

    return function(inputs, state_variables)
