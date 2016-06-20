import numpy
from blocks.bricks import Tanh
from blocks.bricks.recurrent import GatedRecurrent, RecurrentStack, LSTM, SimpleRecurrent
from blocks.bricks.sequence_generators import SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback
from blocks.initialization import Uniform
from blocks.model import Model
from blocks.serialization import load_parameters
from theano import tensor


class NetworkType(object):
    """
    This enum represents the three types of networks we're looking at.
    """
    SIMPLE_RNN = 'simple'
    GRU = 'gru'
    LSTM = 'lstm'


class Network(object):
    """
    This class builds our desired network (for type see NetworkType) for training and/or sampling and inspection purposes.
    Note: If you provide the input dimension as int (> 0), the input_dim_file will be ignored!
    The following attributes are available:
    x -- the char sequence tensor
    mask -- the mask
    input_dim -- the input dimension
    output_dim -- the output dimension, always equal to input_dimension
    gen_model -- the model
    cost -- the cost function (type CategoricalCrossEntropy)
    cost_model -- a model of the cost function
    generator -- the blocks SequenceGenerator object
    hidden_dims -- the dimensions of the hidden layers
    """
    def __init__(self, network_type=NetworkType.SIMPLE_RNN, input_dim_file='onehot_size.npy', input_dim=0, hidden_dims=[512, 512, 512], embed_dim=30):
        char_seq = tensor.imatrix("character_seqs")
        mask = tensor.matrix("seq_mask")
        input_dim = input_dim if input_dim else numpy.load(input_dim_file)
        output_dim = input_dim

        # Stack of three RNNs (if this is too much we can of course use a single layer for the beginning).
        # All RNNs are called "layer" so we don't need to use different by-name-filters for different rnn types later
        # when sampling.
        # Note that RecurrentStack automatically appends #0, #1 etc. to the names.
        if network_type == NetworkType.SIMPLE_RNN:
            brick = SimpleRecurrent
        elif network_type == NetworkType.GRU:
            brick = GatedRecurrent
        elif network_type == NetworkType.LSTM:
            brick = LSTM
        else:
            raise ValueError("Invalid RNN type specified!")

        rnns = [brick(dim=dim, activation=Tanh(), name='layer') for dim in hidden_dims]
        stacked_rnn = RecurrentStack(transitions=rnns, skip_connections=True, name="transition")

        # Note: Readout has initial_output 0 because for me that codes a "beginning of sequence" character.
        # The source_names argument looks this way to cope with LSTMs also having cells as part of their "states", but
        # those shouldn't be passed to the readout (since they're for "internal use" only).
        # A note on the side: RecurrentStack takes care of this via a "states_name" argument, which is "states" by
        # default.
        generator = SequenceGenerator(
            Readout(readout_dim=output_dim, source_names=[thing for thing in stacked_rnn.apply.states if "states" in thing],
                    emitter=SoftmaxEmitter(initial_output=0, name="emitter"),
                    feedback_brick=LookupFeedback(num_outputs=output_dim, feedback_dim=embed_dim, name="feedback"),
                    name="readout"),
            transition=stacked_rnn, weights_init=Uniform(width=0.02), biases_init=Uniform(width=0.0001))

        cross_ent = generator.cost(outputs=char_seq.T, mask=mask.T)
        cross_ent.name = "cross_entropy"
        generator.initialize()

        self.x = char_seq
        self.mask = mask
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cost_model = Model(cross_ent)
        self.cost = cross_ent
        self.generator = generator
        self.hidden_dims = hidden_dims
        init_states = []
        id = 0
        for rnn in rnns:
            init_state = None
            i = 0
            while i<len(rnn.parameters) and not init_state:
                if rnn.parameters[i].name == 'initial_state':
                    init_state = rnn.parameters[i]
                i += 1
            if init_state:
                init_states.append(init_state)
                init_state.name += '#'+str(id)
                id += 1
        self.initial_states = init_states

    def set_parameters(self, model_file):
        with open(model_file, 'rb') as f:
            params = load_parameters(f)
        self.cost_model.set_parameter_values(params)