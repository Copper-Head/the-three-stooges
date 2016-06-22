from __future__ import print_function

from blocks.bricks.recurrent import LSTM, SimpleRecurrent, GatedRecurrent, BaseRecurrent, recurrent
from blocks.algorithms import StepRule
from blocks.bricks.base import Application, application, lazy
from blocks.bricks import Tanh, Logistic, Initializable
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


class ZeroInitLSTM(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None, **kwargs):
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        children = ([self.activation, self.gate_activation] +
                    kwargs.get('children', []))
        super(ZeroInitLSTM, self).__init__(children=children, **kwargs)

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(ZeroInitLSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4*self.dim),
                                          name='W_state')
        self.W_cell_to_in = shared_floatx_nans((self.dim,),
                                               name='W_cell_to_in')
        self.W_cell_to_forget = shared_floatx_nans((self.dim,),
                                                   name='W_cell_to_forget')
        self.W_cell_to_out = shared_floatx_nans((self.dim,),
                                                name='W_cell_to_out')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim,),
                                                 name="initial_cells")
        add_role(self.W_state, WEIGHT)
        add_role(self.W_cell_to_in, WEIGHT)
        add_role(self.W_cell_to_forget, WEIGHT)
        add_role(self.W_cell_to_out, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)

        self.parameters = [
            self.W_state, self.W_cell_to_in, self.W_cell_to_forget,
            self.W_cell_to_out, self.initial_state_, self.initial_cells]

    def _initialize(self):
        for weights in self.parameters[:4]:
            self.weights_init.initialize(weights, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        def slice_last(x, no):
            return x[:, no*self.dim: (no+1)*self.dim]

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = self.gate_activation.apply(
            slice_last(activation, 0) + cells * self.W_cell_to_in)
        forget_gate = self.gate_activation.apply(
            slice_last(activation, 1) + cells * self.W_cell_to_forget)
        next_cells = (
            forget_gate * cells +
            in_gate * self.activation.apply(slice_last(activation, 2)))
        out_gate = self.gate_activation.apply(
            slice_last(activation, 3) + next_cells * self.W_cell_to_out)
        next_states = out_gate * self.activation.apply(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.zeros((batch_size, self.dim)),
                tensor.zeros((batch_size, self.dim))]