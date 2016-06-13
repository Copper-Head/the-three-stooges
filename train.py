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
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.streams import DataStream
from theano import tensor

from custom_blocks import PadAndAddMasks
from network import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--clipping", default=1.0, type=float, help="Step clipping threshold (default: 1)")
parser.add_argument("-t", "--type", default="simple", help="Type of RNN ("+NetworkType.SIMPLE_RNN+", "+\
                                                           NetworkType.GRU+" or "+NetworkType.LSTM+")")
parser.add_argument("-e", "--epochs", default=0, type=int, help="Stop after this many epochs (default: 0)")
parser.add_argument("-d", "--dimensions", default="512,512,512", type=str, help="Configure number of layers and dimensions by"+\
                                                                                " just enumerating dimensions (e.g. 100, 100 for"+\
                                                                                " a two-layered network with dims 100)")
args = parser.parse_args()

dimensions = [int(d) for d in args.dimensions.split(',')]

network = Network(network_type=args.type, clipping=args.clipping, hidden_dims=dimensions)

cross_ent = network.cost
generator = network.generator
gen_model = network.gen_model

# the thing that I feed into the parameters argument I copied from some blocks-examples thing. the good old computation
# graph and then cg.parameters should work as well
# on the step rule: This means gradient clipping (threshold passed in as a command line argument) followed by Adam
# performed on the clipped gradient
# if you don't know what gradient clipping is: Define threshold T; if the length of the gradient is > T, scale it such
# that its length is equal to T. This serves to make exploding gradients less severe
algorithm = GradientDescent(cost=cross_ent, parameters=list(Selector(generator).get_parameters().values()),
                            step_rule=CompositeRule(components=[StepClipping(threshold=args.clipping), Adam()]),
                            on_unused_sources="ignore")

# data
train_data = H5PYDataset("bible.hdf5", which_sets=("train",), load_in_memory=True)
valid_data = H5PYDataset("bible.hdf5", which_sets=("valid",), load_in_memory=True)

# see custom_blocks for the transformer
data_stream = PadAndAddMasks(
    DataStream.default_stream(dataset=train_data, iteration_scheme=ShuffledScheme(train_data.num_examples,
                                                                                  batch_size=32)),
    produces_examples=False)  # I don't know what this does or why you have to pass it but apparently you do
data_stream_valid = PadAndAddMasks(
    DataStream.default_stream(dataset=valid_data, iteration_scheme=SequentialScheme(valid_data.num_examples,
                                                                                    batch_size=32)),
    produces_examples=False)
save = Checkpoint("seqgen_" + args.type + '_' + '_'.join(network.hidden_dims) + ".pkl")

# monitor:
# - training cost every 200 batches (computed along the way, so cheap to do), as well as gradient and step lengths to
#   detect exploding gradient problems
# - validation cost once every epoch
monitor_grad = TrainingDataMonitoring(variables=[cross_ent, aggregation.mean(algorithm.total_gradient_norm),
                                                 aggregation.mean(algorithm.total_step_norm)], every_n_batches=200,
                                      prefix="training")
monitor_valid = DataStreamMonitoring(data_stream=data_stream_valid, variables=[cross_ent], every_n_epochs=1,
                                     prefix="validation")

# training will run forever until you cancel manually
# TODO write an extension that saves the best model (set of parameters) so far, wrt validation performance
main_loop = MainLoop(algorithm=algorithm, data_stream=data_stream, model=gen_model,
                     extensions=[monitor_grad, monitor_valid, FinishAfter(after_n_epochs=args.epochs), ProgressBar(),
                                 Timing(), Printing(every_n_batches=200), save])

main_loop.run()
