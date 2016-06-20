import argparse

from blocks.algorithms import GradientDescent, Adam, CompositeRule, StepClipping
from blocks.extensions import Timing, Printing, FinishAfter, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.monitoring import aggregation
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.streams import DataStream
from numpy import load
from theano.tensor import TensorVariable

from custom_blocks import PadAndAddMasks, EarlyStopping
from network import *

from prepare_lk import ALPHABET_FILE, OUT_FILE_NAME as DATA_FILE_LOC
from util import StateComputer

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--clipping", default=1.0, type=float, help="Step clipping threshold (default: 1)")
parser.add_argument("-t", "--type", default="simple", help="Type of RNN ("+NetworkType.SIMPLE_RNN+", " +
                                                           NetworkType.GRU+" or "+NetworkType.LSTM+")")
parser.add_argument("-e", "--epochs", default=0, type=int, help="Stop after this many epochs (default: 0 -- means it will" +
                                                                " run forever until you cancel manually)")
parser.add_argument("-d", "--dimensions", default="512,512,512", type=str, help="Configure number of layers and dimensions by" +
                                                                                " just enumerating dimensions (e.g. 100, 100 for" +
                                                                                " a two-layered network with dims 100)")
args = parser.parse_args()

dimensions = [int(d) for d in args.dimensions.split(',')]
print('Dimensions:', dimensions)

nkwargs = {'network_type': args.type}
if dimensions:
    nkwargs['hidden_dims'] = dimensions

network = Network(**nkwargs)

cross_ent = network.cost
generator = network.generator
cost_model = network.cost_model

# do init_state stuff
initial_states = network.initial_states


# The thing that I feed into the parameters argument I copied from some blocks-examples thing. the good old computation
# graph and then cg.parameters should work as well.
# On the step rule: This means gradient clipping (threshold passed in as a command line argument) followed by Adam
# performed on the clipped gradient.
# If you don't know what gradient clipping is: Define threshold T; if the length of the gradient is > T, scale it such
# that its length is equal to T. This serves to make exploding gradients less severe.
algorithm = GradientDescent(cost=cross_ent, parameters=cost_model.parameters,
                            step_rule=CompositeRule(components=[StepClipping(threshold=args.clipping), Adam()]),
                            on_unused_sources="ignore")

# data
train_data = H5PYDataset(DATA_FILE_LOC, which_sets=("train",), load_in_memory=True)
valid_data = H5PYDataset(DATA_FILE_LOC, which_sets=("valid",), load_in_memory=True)

# see custom_blocks for the transformer
data_stream = PadAndAddMasks(DataStream.default_stream(dataset=train_data, iteration_scheme=ShuffledScheme(train_data.num_examples,
                                                                                  batch_size=1)), produces_examples=False)
data_stream_valid = PadAndAddMasks(DataStream.default_stream(dataset=valid_data, iteration_scheme=SequentialScheme(valid_data.num_examples,
                                                                                    batch_size=1)), produces_examples=False)

# monitor:
# - training cost every 200 batches (computed along the way, so cheap to do), as well as gradient and step lengths to
#   detect exploding gradient problems
# - validation cost once every epoch


char2ix = load(ALPHABET_FILE).item()
ix2char = {}
for k, v in char2ix.items():
    ix2char[v] = k

sc = StateComputer(network.cost_model, ix2char)

monitor_grad = TrainingDataMonitoring(variables=[cross_ent, aggregation.mean(algorithm.total_gradient_norm),
                                                 aggregation.mean(algorithm.total_step_norm), network.initial_states[2]], after_epoch=True,
                                      prefix="training")
#monitor_init_states = TrainingDataMonitoring(variables=[network.initial_states[2]], after_epoch=True, prefix='training')

early_stopping = EarlyStopping(variables=[cross_ent], data_stream=data_stream_valid,
                               path="seqgen_" + args.type + "_" + "_".join([str(d) for d in network.hidden_dims]) + ".pkl",
                               tolerance=4, prefix="validation")

main_loop = MainLoop(algorithm=algorithm, data_stream=data_stream, model=cost_model,
                     extensions=[monitor_grad, early_stopping, FinishAfter(after_n_epochs=args.epochs), ProgressBar(),
                                 Timing(), Printing()])

main_loop.run()
