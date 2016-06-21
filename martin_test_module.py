from blocks.algorithms import StepRule
from blocks.monitoring.evaluators import AggregationBuffer

from collections import OrderedDict

class OverrideStateReset(StepRule):

    def __init__(self, init_states_to_states_dict):
        self._dict = init_states_to_states_dict

    def compute_steps(self, previous_steps):
        """Build a Theano expression for steps for all parameters.

        Override this method if you want to process the steps
        with respect to all parameters as a whole, not parameter-wise.

        Parameters
        ----------
        previous_steps : OrderedDict
            An :class:`~OrderedDict` of
            (:class:`~tensor.TensorSharedVariable`
            :class:`~tensor.TensorVariable`) pairs. The keys are the
            parameters being trained, the values are the expressions for
            quantities related to gradients of the cost with respect to
            the parameters, either the gradients themselves or steps in
            related directions.

        Returns
        -------
        steps : OrderedDict
            A dictionary of the proposed steps in the same form as
            `previous_steps`.
        updates : list
            A list of tuples representing updates to be performed.

        """
        init_states = [(param, self._dict[param][0][-1] if param in self._dict else param) for param, value in previous_steps.items()]
        print('>>>>>>>>>>>>>>>>>>>>> DEBUG >>>>>>>>>>>>>>>>>>>>>>', init_states)
        return_dict = self._dict.copy()
        for param, value in previous_steps.items():
            if not param in self._dict:
                return_dict[param] = value
        return return_dict, init_states

