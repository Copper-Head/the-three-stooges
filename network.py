import argparse

import numpy
from blocks.algorithms import GradientDescent, Adam, CompositeRule, StepClipping
from blocks.bricks import Tanh
from blocks.bricks.recurrent import GatedRecurrent, RecurrentStack, LSTM, SimpleRecurrent
from blocks.bricks.sequence_generators import SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback
from blocks.extensions import Timing, Printing, FinishAfter, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.initialization import Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.select import Selector
from blocks.serialization import load_parameters
from enum import Enum
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.streams import DataStream
from theano import tensor

from custom_blocks import PadAndAddMasks


class NetworkType(Enum):
    """
    This enum represents the three types of networks we're looking at.
    """
    SIMPLE_RNN = 'simple'
    GRU = 'gru'
    LSTM = 'lstm'


class Network(object):
    """
    This class builds our desired network (for type see NetworkType) for training and/or sampling and inspection purposes.
    The following attributes are available:
    x -- the char sequence tensor
    mask -- the mask
    input_dim -- the input dimension
    output_dim -- the output dimension, always equal to input_dimension
    gen_model -- the model
    cost -- the cost function (type CategoricalCrossEntropy)
    generator -- the blocks SequenceGenerator object
    hidden_dims -- the dimensions of the hidden layers
    """
    def __init__(self, network_type=NetworkType.SIMPLE_RNN.value, clipping=1.0, input_dim_file='onehot_size.npy', hidden_dims=[512, 512, 512], embed_dim=30):
        char_seq = tensor.imatrix("character_seqs")
        mask = tensor.matrix("seq_mask")
        input_dim = numpy.load(input_dim_file)
        output_dim = input_dim

        # stack of three RNNs (if this is too much we can of course use a single layer for the beginning)
        # all RNNs are called "layer" so we don't need to use different by-name-filters for different rnn types later when
        # sampling
        # note that RecurrentStack automatically appends #0, #1 etc. to the names
        if network_type == NetworkType.SIMPLE_RNN.value:
            rnns = [SimpleRecurrent(dim=hidden_dims[0], activation=Tanh(), name="layer"),
            SimpleRecurrent(dim=hidden_dims[1], activation=Tanh(), name="layer"),
            SimpleRecurrent(dim=hidden_dims[2], activation=Tanh(), name="layer")]
        elif network_type == NetworkType.GRU.value:
            rnns = [GatedRecurrent(dim=hidden_dims[0], name="layer"), GatedRecurrent(dim=hidden_dims[1], name="layer"),
            GatedRecurrent(dim=hidden_dims[2], name="layer")]
        elif network_type == NetworkType.LSTM.value:
            rnns = [LSTM(dim=hidden_dims[0], name="layer"), LSTM(dim=hidden_dims[1], name="layer"),
            LSTM(dim=hidden_dims[2], name="layer")]
        else:
            raise ValueError("Invalid RNN type specified!")

        stacked_rnn = RecurrentStack(transitions=rnns, skip_connections=True, name="transition")

        # note: Readout has initial_output 0 because for me that codes a "beginning of sequence" character
        # the source_names argument looks this way to cope with LSTMs also having cells as part of their "states", but those
        # shouldn't be passed to the readout (since they're for "internal use" only)
        generator = SequenceGenerator(
            Readout(readout_dim=output_dim, source_names=[thing for thing in stacked_rnn.apply.states if "states" in thing],
                    emitter=SoftmaxEmitter(initial_output=0, name="emitter"),
                    feedback_brick=LookupFeedback(num_outputs=output_dim, feedback_dim=embed_dim, name="feedback"),
                    name="readout"),
            transition=stacked_rnn, weights_init=Uniform(width=0.02), biases_init=Uniform(width=0.0001))

        cross_ent = generator.cost(outputs=char_seq, mask=mask)
        generating = generator.generate(n_steps=char_seq.shape[0], batch_size=char_seq.shape[1])
        generator.initialize()

        self.x = char_seq
        self.mask = mask
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gen_model = Model(generating)
        self.cost = cross_ent
        self.generator = generator
        self.hidden_dims = hidden_dims

    def set_parameters(self, model_file):
        with open(model_file, 'rb') as f:
            params = load_parameters(f)
        self.gen_model.set_parameter_values(params)