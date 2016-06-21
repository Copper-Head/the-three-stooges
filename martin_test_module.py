from blocks.algorithms import StepRule
from blocks.bricks.recurrent import SimpleRecurrent, recurrent
from blocks.bricks.base import application, lazy
from theano import tensor, shared
from collections import OrderedDict
from numpy import array

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


class NoResetSimpleRecurrent(SimpleRecurrent):
    """The traditional recurrent transition.
    The most well-known recurrent transition: a matrix multiplication,
    optionally followed by a non-linearity.
    Parameters
    ----------
    dim : int
        The dimension of the hidden state
    activation : :class:`~.bricks.Brick`
        The brick to apply as activation.
    Notes
    -----
    See :class:`.Initializable` for initialization parameters.
    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, **kwargs):
        self.dim = dim
        children = [activation]
        kwargs.setdefault('children', []).extend(children)
        super(NoResetSimpleRecurrent, self).__init__(dim, activation, **kwargs)

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=[])
    def apply(self, inputs, states, mask=None):
        """Apply the simple transition.
        Parameters
        ----------
        inputs : :class:`~tensor.TensorVariable`
            The 2D inputs, in the shape (batch, features).
        states : :class:`~tensor.TensorVariable`
            The 2D states, in the shape (batch, features).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if
            there is data available, 0 if not. Assumed to be 1-s
            only if not given.
        """
        next_states = inputs + tensor.dot(states, self.W)
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return tensor.repeat(shared(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float32'))[None, :], batch_size, 0)  # for testing I now only return a vector with a very characteristic sequence of floats

    ## TODO: possible options: 1) return states as they are and not initial states in initial_states() OR 2) have a look at recurrent-definition in BaseRecurrent