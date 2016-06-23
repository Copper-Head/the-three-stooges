from __future__ import print_function

import argparse
from collections import OrderedDict

from blocks.algorithms import GradientDescent, Adam, CompositeRule, StepClipping
from blocks.extensions import Timing, Printing, FinishAfter, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.training import SharedVariableModifier
from blocks.main_loop import MainLoop
from blocks.monitoring import aggregation
from blocks.monitoring.evaluators import AggregationBuffer
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.streams import DataStream
from numpy import load, array, ones, all, zeros
from theano import function, shared

from custom_blocks import PadAndAddMasks, EarlyStopping
from network import *

from prepare_lk import ALPHABET_FILE, OUT_FILE_NAME as DATA_FILE_LOC
from util import StateComputer
from martin_test_module import OverrideStateReset

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

nkwargs = {'network_type': args.type, 'reset_states': False}
if dimensions:
    nkwargs['hidden_dims'] = dimensions

network = Network(**nkwargs)

cross_ent = network.cost
generator = network.generator
cost_model = network.cost_model

# do init_state stuff
initial_states = network.initial_states
#init_state_2 = initial_states[2]

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

# FIXME too complicated and hard-coded
state_to_compare_0 = list(filter(lambda x: x.name == 'sequencegenerator_cost_matrix_states', sc.state_variables))[0]
state_to_compare_1 = list(filter(lambda x: x.name == 'sequencegenerator_cost_matrix_states#1', sc.state_variables))[0]
state_to_compare_2 = list(filter(lambda x: x.name == 'sequencegenerator_cost_matrix_states#2', sc.state_variables))[0]  # notice: python2 filter seems to return a list, but anyway


# The thing that I feed into the parameters argument I copied from some blocks-examples thing. the good old computation
# graph and then cg.parameters should work as well.
# On the step rule: This means gradient clipping (threshold passed in as a command line argument) followed by Adam
# performed on the clipped gradient.
# If you don't know what gradient clipping is: Define threshold T; if the length of the gradient is > T, scale it such
# that its length is equal to T. This serves to make exploding gradients less severe.
algorithm = GradientDescent(cost=cross_ent, parameters=cost_model.parameters,
                            step_rule=CompositeRule(components=[StepClipping(threshold=args.clipping), Adam()]),
                            on_unused_sources="ignore")


aggr_0 = AggregationBuffer(variables=[state_to_compare_0], use_take_last=True)
aggr_0.initialize_aggregators()
aggr_0.check_value = shared(zeros((network.transitions[-1].dim)))

def modifier_function_0(iterations_done, old_value):
    """
    Note about this method: The accumulator seems to get the state vector in random order
    :param iterations_done:
    :param old_value:
    :return:
    """
    values = aggr_0.get_aggregated_values()
    new_value = values[state_to_compare_0.name]
    aggr_0.initialize_aggregators()  # TODO what's the purpose of that? I observed them do it in the monitoring extensions after every request
    print('0:OLD ..:', old_value)
    print('0:NEW in:', new_value[-1][0], new_value[0][0], sep='\n')
    print('0:CHECK:', aggr_0.check_value.get_value(), sep='\n')
    value_a = new_value[-1][0]
    value_b = new_value[0][0]
    if all(aggr_0.check_value.get_value() == zeros((1, old_value.size), dtype='float32')):
        aggr_0.check_value.set_value(old_value)
    new_value = value_a if all(value_a != aggr_0.check_value) else value_b
    aggr_0.check_value.set_value(new_value)
    print('0:CHOICE:', new_value, sep='\n')
    return new_value

aggr_1 = AggregationBuffer(variables=[state_to_compare_1], use_take_last=True)
aggr_1.initialize_aggregators()
aggr_1.check_value = shared(zeros((network.transitions[-1].dim)))

def modifier_function_1(iterations_done, old_value):
    """
    Note about this method: The accumulator seems to get the state vector in random order
    :param iterations_done:
    :param old_value:
    :return:
    """
    values = aggr_1.get_aggregated_values()
    new_value = values[state_to_compare_1.name]
    aggr_1.initialize_aggregators()  # TODO what's the purpose of that? I observed them do it in the monitoring extensions after every request
    print('1:OLD ..:', old_value)
    print('1:NEW in:', new_value[-1][0], new_value[0][0], sep='\n')
    print('1:CHECK:', aggr_1.check_value.get_value(), sep='\n')
    value_a = new_value[-1][0]
    value_b = new_value[0][0]
    if all(aggr_1.check_value.get_value() == zeros((1, old_value.size), dtype='float32')):
        aggr_1.check_value.set_value(old_value)
    new_value = value_a if all(value_a != aggr_1.check_value) else value_b
    aggr_1.check_value.set_value(new_value)
    print('1:CHOICE:', new_value, sep='\n')
    return new_value

aggr_2 = AggregationBuffer(variables=[state_to_compare_2], use_take_last=True)
aggr_2.initialize_aggregators()
aggr_2.check_value = shared(zeros((network.transitions[-1].dim)))

def modifier_function_2(iterations_done, old_value):
    """
    Note about this method: The accumulator seems to get the state vector in random order
    :param iterations_done:
    :param old_value:
    :return:
    """
    values = aggr_2.get_aggregated_values()
    new_value = values[state_to_compare_2.name]
    aggr_2.initialize_aggregators()  # TODO what's the purpose of that? I observed them do it in the monitoring extensions after every request
    print('2:OLD ..:', old_value)
    print('2:NEW in:', new_value[-1][0], new_value[0][0], sep='\n')
    print('2:CHECK:', aggr_2.check_value.get_value(), sep='\n')
    value_a = new_value[-1][0]
    value_b = new_value[0][0]
    if all(aggr_2.check_value.get_value() == zeros((1, old_value.size), dtype='float32')):
        aggr_2.check_value.set_value(old_value)
    new_value = value_a if all(value_a != aggr_2.check_value) else value_b
    aggr_2.check_value.set_value(new_value)
    print('2:CHOICE:', new_value, sep='\n')
    return new_value



# FIXME hard-coded for 3-layered LSTM
modifier_functions = {
    network.transitions[0].name : modifier_function_0,
    network.transitions[1].name : modifier_function_1,
    network.transitions[2].name : modifier_function_2
}

#init_state_modifier = SharedVariableModifier(network.transitions[-1].initial_state_, function=modifier_function, after_batch=True)
init_state_modifiers = [SharedVariableModifier(trans.initial_state_, function=modifier_functions[trans.name], after_batch=True) for trans in network.transitions]


#state_function = function([state_to_compare], initial_states[2], updates=[(init_state_2, state_to_compare[0][-1])]) #TODO look at this, this is how it basically works!

monitor_grad = TrainingDataMonitoring(variables=[cross_ent, aggregation.mean(algorithm.total_gradient_norm),
                                                 aggregation.mean(algorithm.total_step_norm)],  #+initial_states+[state_to_compare_1],
                                      prefix="training", after_batch=True)

early_stopping = EarlyStopping(variables=[cross_ent], data_stream=data_stream_valid,
                               path="seqgen_" + args.type + "_" + "_".join([str(d) for d in network.hidden_dims]) + ".pkl",
                               tolerance=4, prefix="validation")

prkwargs = {
    #'after_batch':True  # use this for prints after every batch
}

main_loop = MainLoop(algorithm=algorithm, data_stream=data_stream, model=cost_model,
                     extensions=[monitor_grad, early_stopping, FinishAfter(after_n_epochs=args.epochs), ProgressBar(),
                                 Timing(), Printing(**prkwargs)]+init_state_modifiers)

main_loop.algorithm.add_updates(aggr_0.accumulation_updates)
main_loop.algorithm.add_updates(aggr_1.accumulation_updates)
main_loop.algorithm.add_updates(aggr_2.accumulation_updates)



# remove update
# updates = main_loop.algorithm.updates
# init_state_update = list(filter(lambda u: u[0] == init_state_2, updates))[0]
# updates.remove(init_state_update)
# print('REMOVED UPDATE:', init_state_update)
# print('\nUPDATES:', main_loop.algorithm.updates,'\n')

main_loop.run()
