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

from custom_blocks import PadAndAddMasks, EarlyStopping
from network import *


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--clipping", default=1.0, type=float, help="Step clipping threshold (default: 1)")
parser.add_argument("-t", "--type", default="simple", help="Type of RNN ("+NetworkType.SIMPLE_RNN+", " +
                                                           NetworkType.GRU+" or "+NetworkType.LSTM+")")
parser.add_argument("-e", "--epochs", default=0, type=int, help="Stop after this many epochs (default: 0 -- means it will" +
                                                                " run forever until you cancel manually)")
parser.add_argument("-d", "--dimensions", default="512,512,512", type=str, help="Configure number of layers and dimensions by" +
                                                                                " just enumerating dimensions (e.g. 100, 100 for" +
                                                                         " a two-layered network with dims 100)")
parser.add_argument('-f', '--file', type=str, help='Specifies the data used for training.')
parser.add_argument('-a', '--alphafile', type=str, help='Specifies the location of the alphabet file.')
parser.add_argument('-b', '--batchsize', type=int, help='Set the batch size for training, set 0 for all in one batch and one batch per epoch.')
parser.add_argument('-m', '--memory', type=int, default=1, help='Set to 0 if files should be read from disc, any other number to load into memory.')

args = parser.parse_args()

dimensions = [int(d) for d in args.dimensions.split(',')]
print('Dimensions:', dimensions)

ix2char = load(args.alphafile).item()

nkwargs = {'network_type': args.type, 'input_dim': len(ix2char)}
if dimensions:
    nkwargs['hidden_dims'] = dimensions

network = Network(**nkwargs)

cross_ent = network.cost
generator = network.generator
cost_model = network.cost_model

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
train_data = H5PYDataset(args.file, which_sets=("train",), load_in_memory=args.memory)
valid_data = H5PYDataset(args.file, which_sets=("valid",), load_in_memory=args.memory)

# see custom_blocks for the transformer
data_stream = PadAndAddMasks(
    DataStream.default_stream(dataset=train_data, iteration_scheme=ShuffledScheme(train_data.num_examples,
                                                                                  batch_size=train_data.num_examples if not args.batchsize else args.batchsize)),
    produces_examples=False)  # I don't know what this does or why you have to pass it but apparently you do
data_stream_valid = PadAndAddMasks(
    DataStream.default_stream(dataset=valid_data, iteration_scheme=SequentialScheme(valid_data.num_examples,
                                                                                    batch_size=valid_data.num_examples if not args.batchsize else args.batchsize)),
    produces_examples=False)

# monitor:
# - training cost every 200 batches (computed along the way, so cheap to do), as well as gradient and step lengths to
#   detect exploding gradient problems
# - validation cost once every epoch
monitor_grad = TrainingDataMonitoring(variables=[cross_ent, aggregation.mean(algorithm.total_gradient_norm),
                                                 aggregation.mean(algorithm.total_step_norm)], after_epoch=True,
                                      prefix="training")
early_stopping = EarlyStopping(variables=[cross_ent], data_stream=data_stream_valid,
                               path="seqgen_" + args.type + "_" + "_".join([str(d) for d in network.hidden_dims]) + ".pkl",
                               tolerance=4, prefix="validation")

main_loop = MainLoop(algorithm=algorithm, data_stream=data_stream, model=cost_model,
                     extensions=[monitor_grad, early_stopping, FinishAfter(after_n_epochs=args.epochs), ProgressBar(),
                                 Timing(), Printing()])

main_loop.run()
