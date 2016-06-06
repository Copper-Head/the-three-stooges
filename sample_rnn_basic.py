from blocks.extensions.saveload import load

import cPickle


ml = load("seqgen_gru.pkl")
ind_to_char = cPickle.load(open("ind_to_char.pkl"))

param_dict = ml.model.get_parameter_dict()
# three layers means three different initial states
init_state0 = param_dict["/sequencegenerator/with_fake_attention/transition/gatedrecurrent#0.initial_state"]
init_state1 = param_dict["/sequencegenerator/with_fake_attention/transition/gatedrecurrent#1.initial_state"]
init_state2 = param_dict["/sequencegenerator/with_fake_attention/transition/gatedrecurrent#2.initial_state"]

rnn_sample = ml.model.get_theano_function()

# save the trained initial state to be able to run multiple samples
init_origin0 = init_state0.get_value()
init_origin1 = init_state1.get_value()
init_origin2 = init_state2.get_value()
while 1:
    # reset everything
    init_state0.set_value(init_origin0)
    init_state1.set_value(init_origin1)
    init_state2.set_value(init_origin2)
    current_char = 0
    sequence = []
    while current_char != 1:  # sample until end-of-sequence character is generated
        new_state0, new_state1, new_state2, current_char, cost = rnn_sample([[current_char]])
        current_char = current_char[0][0]  # from 2d (axes for sequence length and batch size) to 0d integer
        init_state0.set_value(new_state0[0][0])  # same here
        init_state1.set_value(new_state1[0][0])
        init_state2.set_value(new_state2[0][0])
        sequence.append(ind_to_char[current_char])
    print "\n"
    print "".join(sequence)
    raw_input("\nAnother sequence?")
