import argparse
from collections import OrderedDict

import numpy
from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import MLP, BatchNormalizedMLP, Tanh, Identity
from blocks.bricks.cost import SquaredError
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from fuel.datasets import IndexableDataset
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from theano import tensor


parser = argparse.ArgumentParser(description="All hail the saviour.")
parser.add_argument("file", help=".hdf5 file with the data.")
parser.add_argument("save", help="Path to save main loop to.")
parser.add_argument("-d", "--dim", default="512", help="Dimension of hidden layer(s). Example: 256,256 (default: 512).")
parser.add_argument("-s", "--batchsize", type=int, default=32, help="Guess what (default: 32).")
parser.add_argument("-b", "--batchnorm", action="store_true", help="Flag to use batch normalization.")
args = parser.parse_args()

# define model
states = tensor.matrix("act_seqs")

input_dim = 512
hidden_dims = [int(dim) for dim in args.dim.split(",")]

if args.batchnorm:
    network = BatchNormalizedMLP
else:
    network = MLP

autoencoder = network(activations=[Tanh() for _ in xrange(len(hidden_dims))] + [Identity()],
                  dims=[input_dim] + hidden_dims + [input_dim],
                  weights_init=Uniform(width=0.02), biases_init=Constant(0))
autoencoder.initialize()

hopefully_states_again = autoencoder.apply(states)

cost = SquaredError().apply(hopefully_states_again, states)
cost.name = "squared_error"
cost_model = Model(cost)

algorithm = GradientDescent(cost=cost, parameters=cost_model.parameters,
                            step_rule=Adam())

# handle data
data = H5PYDataset(args.file, which_sets=("train",), load_in_memory=True)
# trash data for testing
"""
dataraw = numpy.zeros((10000, 512), dtype="float32")
for row in xrange(dataraw.shape[0]):
    dataraw[row] = numpy.random.rand(512)
data = OrderedDict()
data["act_seqs"] = dataraw
data = IndexableDataset(data)
"""
datastream = DataStream.default_stream(data, iteration_scheme=ShuffledScheme(data.num_examples,
                                                                             batch_size=args.batchsize))

monitor = TrainingDataMonitoring(variables=[cost, aggregation.mean(algorithm.total_gradient_norm),
                                            aggregation.mean(algorithm.total_step_norm)],
                                 after_epoch=True)

extensions = [monitor, FinishAfter(after_n_epochs=0), ProgressBar(), Timing(), Printing(), Checkpoint(args.save)]

main_loop = MainLoop(data_stream=datastream, algorithm=algorithm, model=cost_model, extensions=extensions)
main_loop.run()
