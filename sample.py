import argparse
from network import *

import pip, re, sys


inst_pckgs = pip.get_installed_distributions()
cpkl = [e for e in inst_pckgs if re.match('[cC][Pp]ickle', e.project_name)]
if cpkl:
    import cPickle
else:
    import pickle as cPickle

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", default="simple", help="Type of RNN (simple, gru or lstm)")
parser.add_argument("-f", "--file", type=str, help="The model file generated in training.")
parser.add_argument("-r", "--readdims", type=int, default=0, help="Determines, whether the hidden layer dimensions " +
                                                                  "should be read from the file name. 0 (default) for no," +
                                                                  " any other number for yes.")
args = parser.parse_args()

if not args.file:
    raise ValueError("No model file specified.")

fname = args.file

kwargs = {'network_type': args.type}

if args.readdims:
    kwargs['hidden_dims'] = [e for e in fname.split('.')[0].split('_') if e.isnumeric()]

nt = Network(**kwargs)
nt.set_parameters(fname)

with open("ind_to_char.pkl", 'rb') as icf:
    ind_to_char = cPickle.load(icf)

param_dict = nt.gen_model.get_parameter_dict()

# three layers means three different initial states
init_state0 = param_dict["/sequencegenerator/with_fake_attention/transition/layer#0.initial_state"]
init_state1 = param_dict["/sequencegenerator/with_fake_attention/transition/layer#1.initial_state"]
init_state2 = param_dict["/sequencegenerator/with_fake_attention/transition/layer#2.initial_state"]
# LSTMs have cells that are separate from the states. these have to be updated manually in the same manner as the states
is_lstm = args.type == NetworkType.LSTM
if is_lstm:
    init_cells0 = param_dict["/sequencegenerator/with_fake_attention/transition/layer#0.initial_cells"]
    init_cells1 = param_dict["/sequencegenerator/with_fake_attention/transition/layer#1.initial_cells"]
    init_cells2 = param_dict["/sequencegenerator/with_fake_attention/transition/layer#2.initial_cells"]

rnn_sample = nt.gen_model.get_theano_function()

# save the trained initial state to be able to run multiple samples
init_origin0 = init_state0.get_value()
init_origin1 = init_state1.get_value()
init_origin2 = init_state2.get_value()
if is_lstm:
    cells_origin0 = init_cells0.get_value()
    cells_origin1 = init_cells1.get_value()
    cells_origin2 = init_cells2.get_value()
while 1:
    # reset everything
    init_state0.set_value(init_origin0)
    init_state1.set_value(init_origin1)
    init_state2.set_value(init_origin2)
    if is_lstm:
        init_cells0.set_value(cells_origin0)
        init_cells1.set_value(cells_origin1)
        init_cells2.set_value(cells_origin2)
    current_char = 0
    sequence = []
    while current_char != 1:  # sample until end-of-sequence character is generated
        if is_lstm:
            new_state0, new_cells0, new_state1, new_cells1, new_state2, new_cells2, current_char, cost = rnn_sample([[current_char]])
        else:
            new_state0, new_state1, new_state2, current_char, cost = rnn_sample([[current_char]])
        current_char = current_char[0][0]  # from 2d (axes for sequence length and batch size) to 0d integer
        init_state0.set_value(new_state0[0][0])  # same here
        init_state1.set_value(new_state1[0][0])
        init_state2.set_value(new_state2[0][0])
        if is_lstm:
            init_cells0.set_value(new_cells0[0][0])
            init_cells1.set_value(new_cells1[0][0])
            init_cells2.set_value(new_cells2[0][0])
        sequence.append(ind_to_char[current_char])
    print ("\n")
    print ("".join(sequence))
    if sys.version_info.major < 3:
        raw_input("\nAnother sequence?")
    else:
        input("\nAnother sequence?")