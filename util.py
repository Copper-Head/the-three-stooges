from theano import function


class StateComputer(object):
    """Convenient interface to theano function for computing states/cells
    of a trained SequenceGenerator model.

    Expects a Model instance as argument. This model must also be created
    from a SequenceGenerator.cost application.
    Extracts the model's inputs and auxiliary variables into attributes
    and creates theano function from inputs to auxiliary vars.

    Finally, provides a method that wraps the output of the theano function in
    a dict keyed by auxiliary variable labels.
    """

    def __init__(self, cost_model):

        raw_state_vars = filter(self._relevant, cost_model)
        self.state_variables = sorted(raw_state_vars, key=lambda var: var.name)
        self.inputs = sorted(cost_model.inputs, key=lambda var: var.name)
        self.func = function(self.inputs, self.state_variables)

    def _relevant(self, aux_var):
        not_final_value = "final_value" not in aux_var.name
        cell_or_state = "states" in aux_var.name or "cells" in aux_var.name
        return cell_or_state and not_final_value

    def read(self, sequence, mask):
        """Combines list of aux var values (output from theano func) with their
        corresponding labels.

        - because the cost was originally defined with a mask, you also need
        to provide one when using this function. If you're just putting in
        a single sequence, you can just put in a "dummy mask" that is 1 everywhere,
        with size equal to the sequence.
        - input is provided in the form batch_size x seq_len. i.e. if your sequence
        is "I am so cool", provide a numpy array based on the following list:
        [["I", " ", "a", "m", ..., "o", "o", "l"]]
        (of course you actually put in the indices, not the characters)
        as well as a mask of numpy.ones of the same shape
        - input is ordered (character_sequence, mask)
        """
        return dict(zip(self.state_variables, self.func(sequence, mask)))
