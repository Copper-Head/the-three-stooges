import argparse

import numpy
from blocks.algorithms import GradientDescent, Adam, CompositeRule, StepClipping
from blocks.bricks.recurrent import GatedRecurrent, RecurrentStack, LSTM
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


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--clipping", default=1.0, type=float, help="Step clipping threshold (default: 1)")
args = parser.parse_args()

# define model
# character sequences: a matrix of dimensions sequence_length x batch_size (remember for RNNs the time dimension comes
# first!). imatrix is important because LookupTable would crash otherwise!
# mask has the same dimensions; it's 1 for those parts of char_seq that are "real" (not padding) and 0 otherwise
char_seq = tensor.imatrix("character_seqs")
mask = tensor.matrix("seq_mask")

# these dimensions are necessary due to the nature of the task
input_dim = numpy.load("onehot_size.npy")
output_dim = input_dim
# these are arbitrary
hidden_dim1 = 512
hidden_dim2 = 512
hidden_dim3 = 512
embed_dim = 30

# stack of three GRUs (of course we can use a simpler transition for the beginning if you want to)
rnns = [GatedRecurrent(dim=hidden_dim1), GatedRecurrent(dim=hidden_dim2), GatedRecurrent(dim=hidden_dim3)]
stacked_rnn = RecurrentStack(transitions=rnns, skip_connections=True, name="transition")
# note: Readout has initial_output 0 because for me that codes a "beginning of sequence" character
generator = SequenceGenerator(
    Readout(readout_dim=output_dim, source_names=[thing for thing in stacked_rnn.apply.states if "states" in thing],
            emitter=SoftmaxEmitter(initial_output=0, name="emitter"),
            feedback_brick=LookupFeedback(num_outputs=output_dim, feedback_dim=embed_dim, name="feedback"),
            name="readout"),
    transition=stacked_rnn, weights_init=Uniform(width=0.02), biases_init=Uniform(width=0.0001))

cross_ent = generator.cost(outputs=char_seq, mask=mask)
generating = generator.generate(n_steps=char_seq.shape[0], batch_size=char_seq.shape[1])
generator.initialize()

gen_model = Model(generating)

# the thing that I feed into the parameters argument I copied from some blocks-examples thing. the good old computation
# graph and then cg.parameters should work as well
# on the step rule: This means gradient clipping (threshold passed in as a command line argument) followed by Adam
# performed on the clipped gradient
# if you don't know what gradient clipping is: Define treshold T; if the length of the gradient is > T, scale it such
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
save = Checkpoint("seqgen_gru.pkl")

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
# TODO write an extension that saves the best model (set of parameters) so far
main_loop = MainLoop(algorithm=algorithm, data_stream=data_stream, model=gen_model,
                     extensions=[monitor_grad, monitor_valid, FinishAfter(after_n_epochs=0), ProgressBar(),
                                 Timing(), Printing(every_n_batches=200), save])

main_loop.run()
