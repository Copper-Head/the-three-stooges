
# coding: utf-8

# # CHECK models

# In[1]:

from blocks.filter import VariableFilter
from custom_blocks import PadAndAddMasks
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from network import *
import numpy
import pickle
from util import StateComputer


# In[6]:

NTYPE = NetworkType.GRU
IX_2_TOK_FILE = './data/hdt-ncs-eos-np-35-7-1_ix2tok.npy'
DATA_FILE = './data/hdt-ncs-eos-np-35-7-1_data.hdf5'
MODEL_FILE = './seqgen_'+NTYPE+'_512_512.pkl'
BATCH_SIZE = 100


# In[3]:

ix2tok = numpy.load(IX_2_TOK_FILE).item()
nt = Network(NTYPE, input_dim=len(ix2tok), hidden_dims=[512, 512])
nt.set_parameters(MODEL_FILE)


# In[4]:

readouts = VariableFilter(theano_name='readout_readout_output_0')(nt.cost_model.variables)[0]
seq_probs = nt.generator.readout.emitter.probs(readouts)
sc = StateComputer(nt.cost_model, {v: k for k, v in ix2tok.items()})
sc.set_prob_func([nt.x, nt.mask], seq_probs)


# In[8]:

dta_valid = H5PYDataset(DATA_FILE, which_sets=('valid',), load_in_memory=True)
data_stream = PadAndAddMasks(DataStream.default_stream(dataset=dta_valid, iteration_scheme=SequentialScheme(dta_valid.num_examples, batch_size=BATCH_SIZE)), produces_examples=False)


# # iterate over data

# In[9]:

it_data = data_stream.get_epoch_iterator()
probs = []
i = 0
for seqs, mask in it_data:  # do it like this instead of using normal batches to avoid vanishing, maybe you find a better strategy        
    probs.append(sc.compute_raw_sequence_probabilities(seqs, mask=mask))
    i += 1
    print('step', i, 'done')
with open('probs_gru.pkl', 'wb') as f:
    pickle.dump(probs, f)
print('done')