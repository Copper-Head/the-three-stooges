from __future__ import print_function

from blocks.bricks.recurrent import LSTM, SimpleRecurrent, GatedRecurrent

from blocks.algorithms import StepRule
from blocks.bricks.recurrent import SimpleRecurrent, recurrent
from blocks.bricks.base import Application, application, lazy
from blocks.initialization import NdarrayInitialization
from blocks.roles import add_role, INITIAL_STATE, WEIGHT
from blocks.utils import pack, dict_union, dict_subset, is_shared_variable, shared_floatx_nans, shared_floatx_zeros
from theano import tensor, shared, Variable
import theano
from collections import OrderedDict
from functools import wraps
import inspect
import logging
from numpy import array
from picklable_itertools.extras import equizip
from theano.tensor import TensorVariable

logger = logging.getLogger(__name__)
unknown_scan_input = """

Your function uses a non-shared variable other than those given \
by scan explicitly. That can significantly slow down `tensor.grad` \
call. Did you forget to declare it in `contexts`?"""


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
        updates = [(param, self._dict[param]) for param, value in previous_steps.items() if param in self._dict]
        print('>>>>>>>>>>>>>>>>>>>>> DEBUG >>>>>>>>>>>>>>>>>>>>>>', updates)
        step_dict = self._dict.copy()
        for param, value in previous_steps.items():
            if not param in self._dict:
                step_dict[param] = value
        return step_dict, updates


class ZeroInitLSTM(LSTM):

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.zeros((batch_size, self.dim)),
                tensor.zeros((batch_size, self.dim))]