from __future__ import print_function

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


def no_reset_recurrent(*args, **kwargs):
    """Wraps an apply method to allow its iterative application.

    This decorator allows you to implement only one step of a recurrent
    network and enjoy applying it to sequences for free. The idea behind is
    that its most general form information flow of an RNN can be described
    as follows: depending on the context and driven by input sequences the
    RNN updates its states and produces output sequences.

    Given a method describing one step of an RNN and a specification
    which of its inputs are the elements of the input sequence,
    which are the states and which are the contexts, this decorator
    returns an application method which implements the whole RNN loop.
    The returned application method also has additional parameters,
    see documentation of the `recurrent_apply` inner function below.

    Parameters
    ----------
    sequences : list of strs
        Specifies which of the arguments are elements of input sequences.
    states : list of strs
        Specifies which of the arguments are the states.
    contexts : list of strs
        Specifies which of the arguments are the contexts.
    outputs : list of strs
        Names of the outputs. The outputs whose names match with those
        in the `state` parameter are interpreted as next step states.

    Returns
    -------
    recurrent_apply : :class:`~blocks.bricks.base.Application`
        The new application method that applies the RNN to sequences.

    See Also
    --------
    :doc:`The tutorial on RNNs </rnn>`

    """

    most_recent_state_values = {}  # FIXME: REMOVE

    def recurrent_wrapper(application_function):
        arg_spec = inspect.getargspec(application_function)
        arg_names = arg_spec.args[1:]

        @wraps(application_function)
        def recurrent_apply(brick, application, application_call,
                            *args, **kwargs):
            """Iterates a transition function.

            Parameters
            ----------
            iterate : bool
                If ``True`` iteration is made. By default ``True``.
            reverse : bool
                If ``True``, the sequences are processed in backward
                direction. ``False`` by default.
            return_initial_states : bool
                If ``True``, initial states are included in the returned
                state tensors. ``False`` by default.

            """
            logger.info('>>>>>>>>>>>>>>> recurrent_apply called')
            logger.info('RECURRENT APPLY ARGS: '+str(args))
            logger.info('RECURRENT APPLY KWARGS: '+str(kwargs))
            state_info = kwargs['states']
            logger.info('\nSTATES: '+(str(dir(state_info)) if state_info else 'None')+'\n')
            # Extract arguments related to iteration and immediately relay the
            # call to the wrapped function if `iterate=False`
            iterate = kwargs.pop('iterate', True)
            if not iterate:
                return application_function(brick, *args, **kwargs)
            reverse = kwargs.pop('reverse', False)
            scan_kwargs = kwargs.pop('scan_kwargs', {})
            #scan_kwargs['truncate_gradient'] = 10 # FIXME !!! better: CHECK
            return_initial_states = kwargs.pop('return_initial_states', False)

            # Push everything to kwargs
            for arg, arg_name in zip(args, arg_names):
                kwargs[arg_name] = arg

            # Make sure that all arguments for scan are tensor variables
            scan_arguments = (application.sequences + application.states +
                              application.contexts)
            for arg in scan_arguments:
                if arg in kwargs:
                    if kwargs[arg] is None:
                        del kwargs[arg]
                    else:
                        kwargs[arg] = tensor.as_tensor_variable(kwargs[arg])

            # Check which sequence and contexts were provided
            sequences_given = dict_subset(kwargs, application.sequences,
                                          must_have=False)
            contexts_given = dict_subset(kwargs, application.contexts,
                                         must_have=False)

            # Determine number of steps and batch size.
            if len(sequences_given):
                # TODO Assumes 1 time dim!
                shape = list(sequences_given.values())[0].shape
                n_steps = shape[0]
                batch_size = shape[1]
            else:
                # TODO Raise error if n_steps and batch_size not found?
                n_steps = kwargs.pop('n_steps')
                batch_size = kwargs.pop('batch_size')

            # Handle the rest kwargs
            rest_kwargs = {key: value for key, value in kwargs.items()
                           if key not in scan_arguments}
            for value in rest_kwargs.values():
                if (isinstance(value, Variable) and not
                        is_shared_variable(value)):
                    logger.warning("unknown input {}".format(value) +
                                   unknown_scan_input)

            # Ensure that all initial states are available.
            initial_states = brick.initial_states(batch_size, as_dict=True,
                                                  *args, **kwargs)
            for state_name in application.states:  # TODO another thing to try would be to just empty the application.states and see if that works
                logger.info(most_recent_state_values)
                dim = brick.get_dim(state_name)
                if state_name in kwargs:
                    if isinstance(kwargs[state_name], NdarrayInitialization):
                        logger.info('Case NdarrayInitialization')
                        """
                        kwargs[state_name] = tensor.alloc(
                            kwargs[state_name].generate(brick.rng, (1, dim)),
                            batch_size, dim)
                        """
                        try:
                            kwargs[state_name] = most_recent_state_values[state_name]
                        except KeyError:
                            raise KeyError("no most recent value for {} of brick {}".format(state_name, brick.name))
                    elif isinstance(kwargs[state_name], Application):
                        logger.info('Case Application')
                        kwargs[state_name] = (
                            kwargs[state_name](state_name, batch_size,
                                               *args, **kwargs))
                        most_recent_state_values[state_name] = kwargs[state_name]
                # I suspect the following lines to be responsible for the reset of the states, so all I have to do -- at
                # least from what I think -- is to set the state to its own value instead of to the init-state's value
                else:
                    logger.info('Case else, reset (?) procedure, state: '+str(state_info))
                    try:
                        # kwargs[state_name] = initial_states[state_name]  # OLD
                        # kwargs[state_name] = point on value of tensor variable, problem: HOW TO GET THEM IN HERE?
                        if state_name in most_recent_state_values:
                            kwargs[state_name] = most_recent_state_values[kwargs]
                        else:
                            kwargs[state_name] = initial_states[state_name]  # necessary for first initialization (?)
                        logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DID IT WORK?')
                    except KeyError:
                        raise KeyError(
                            "no initial state for '{}' of the brick {}".format(
                                state_name, brick.name))
            states_given = dict_subset(kwargs, application.states)

            # Theano issue 1772 (FROM WHAT I GOT THIS IS NONE OF OUR BUSINESS)
            for name, state in states_given.items():
                states_given[name] = tensor.unbroadcast(state,
                                                        *range(state.ndim))

            #logger.info('STATES_GIVEN: '+str(states_given.items())+ ' ('+brick.name+')')
            def scan_function(*args):
                args = list(args)
                arg_names = (list(sequences_given) +
                             [output for output in application.outputs
                              if output in application.states] +
                             list(contexts_given))
                kwargs = dict(equizip(arg_names, args))
                kwargs.update(rest_kwargs)
                outputs = application(iterate=False, **kwargs)
                # We want to save the computation graph returned by the
                # `application_function` when it is called inside the
                # `theano.scan`.
                application_call.inner_inputs = args
                application_call.inner_outputs = pack(outputs)
                return outputs
            outputs_info = [
                states_given[name] if name in application.states
                else None
                for name in application.outputs]
            result, updates = theano.scan(
                scan_function, sequences=list(sequences_given.values()),
                outputs_info=outputs_info,
                non_sequences=list(contexts_given.values()),
                n_steps=n_steps,
                go_backwards=reverse,
                name='{}_{}_scan'.format(
                    brick.name, application.application_name),
                **scan_kwargs)
            logger.info('THEANO_SCAN_UPDATES: '+str(updates.items()))  # TODO: Remove print, but keep it for now: It shows, that scan is not responsible for the state reset
            result = pack(result)
            if return_initial_states:
                logger.warn('return_initial_states turned true. Result before: '+str(result))
                # Undo Subtensor
                for i in range(len(states_given)):
                    assert isinstance(result[i].owner.op,
                                      tensor.subtensor.Subtensor)
                    result[i] = result[i].owner.inputs[0]
                logger.warn('...result after: '+str(result))
            if updates:
                application_call.updates = dict_union(application_call.updates,
                                                      updates)

            #logger.info('\n')
            #logger.info('APPLICATION_CALL: '+str(dir(application_call)))
            #logger.info('.... application: '+str(dir(application_call.application)))
            #logger.info('.... brick check: '+str(application_call.application.brick.name == brick.name))  # TRUE
            #logger.info('.... appl.states: '+str(application.states))
            #logger.info('.........  brick: '+str(dir(brick)))
            #logger.info('. brick.aux_vars: '+str(brick.auxiliary_variables))
            #logger.info('... brick.params: '+str(brick.parameters))
            #logger.info('\n')

            return result

        logger.info('RETURNING recurrent_apply='+str(recurrent_apply))

        return recurrent_apply

    # Decorator can be used with or without arguments
    assert (args and not kwargs) or (not args and kwargs)
    if args:
        application_function, = args
        return application(recurrent_wrapper(application_function))
    else:
        def wrap_application(application_function):
            return application(**kwargs)(
                recurrent_wrapper(application_function))
        return wrap_application


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
        self._state = None

    @DeprecationWarning
    def register_state(self, state):
        """
        very dirty
        :param states: list of states to not be reset
        :return:
        """
        self._state = state
        logger.info(self.name+' received and registered state: '+self._state.name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                                  name="W"))
        add_role(self.parameters[0], WEIGHT)
        # self.parameters.append(shared_floatx_zeros((self.dim,),
          #                                         name="initial_state"))
        # add_role(self.parameters[1], INITIAL_STATE)


    @no_reset_recurrent(sequences=['inputs', 'mask'], states=['states'],
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
        logger.info('APPLY CALLED, value of states: '+str(states)+', brick='+str(self.name))
        next_states = inputs + tensor.dot(states, self.W)
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.outputs)
    def initial_states(self, batch_size, *args, **kwargs):
        print('args:', *args)
        print('self:', *dir(self))
        print('aux_vars: ', *self.auxiliary_variables)
        print('children: ', *self.children)
        print('bound appls: ', str(self._bound_applications))
        print('updates: ', str(self.updates))
        logger.info('INITIAL_STATES KWARGS: '+ str(kwargs))
        logger.info('INITIAL_STATES CALLED, brick='+str(self.name))
        # return tensor.repeat(self._state[0][-1][None, :], batch_size, 0)  # this does not work, since this method is called BEFORE state is registered
        # return tensor.repeat(shared(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float32'))[None, :], batch_size, 0)  # for testing I now only return a vector with a very characteristic sequence of floats NOTE: WORKED!!!
        result = []
        for state in self.apply.states:

            logger.info('LOOPING OVER APPLY.STATES: '+str(state))
            dim = self.get_dim(state)
            if dim == 0:
                result.append(tensor.zeros((batch_size,)))
            else:
                result.append(tensor.zeros((batch_size, dim)))
        return result

    ## TODO: possible options: 1) return states as they are and not initial states in initial_states() OR 2) have a look at recurrent-definition in BaseRecurrent