import argparse

import numpy
from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import MLP, Tanh, Identity
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
parser.add_argument("-d", "--dim", type=int, default=512, help="Dimension of hidden layer (default: 512).")
parser.add_argument("-b", "--batchsize", type=int, default=32, help="Guess what (default: 32).")
args = parser.parse_args()

# define model
states = tensor.matrix("states")

input_dim = 512
encoder_dim = args.dim

autoencoder = MLP(activations=[Tanh(), Identity()], dims=[input_dim, encoder_dim, input_dim],
                  weights_init=Uniform(width=0.02), biases_init=Constant(0))
autoencoder.initialize()

hopefully_states_again = autoencoder.apply(states)

cost = SquaredError().apply(hopefully_states_again, states)
cost.name = "squared_error"
cost_model = Model(cost)

algorithm = GradientDescent(cost=cost, parameters=cost_model.parameters,
                            step_rule=Adam())

# handle data
#data = H5PYDataset(args.file, which_sets="act_seqs", load_in_memory=True)
data = numpy.random.rand(10000, 512)
data = IndexableDataset(data)
datastream = DataStream.default_stream(data, iteration_scheme=ShuffledScheme(data.num_examples,
                                                                             batch_size=args.batchsize))

monitor = TrainingDataMonitoring(variables=[cost, aggregation.mean(algorithm.total_gradient_norm),
                                            aggregation.mean(algorithm.total_step_norm)],
                                 after_epoch=True)

extensions = [monitor, FinishAfter(after_n_epochs=0), ProgressBar(), Timing(), Printing(), Checkpoint(args.save)]

main_loop = MainLoop(data_stream=datastream, algorithm=algorithm, model=cost_model, extensions=extensions)
main_loop.run()
