import numpy as np
from os import path, listdir
from fuel.streams import DataStream
from fuel.datasets import H5PYDataset
from fuel.schemes import SequentialScheme
import dataproc as dp
import h5py

SOURCE_DIR = './models/hdt/activations_lstm'
SEQ_FILE = './data/hdt-ncs-eos-np-35-7-1_data.hdf5'
SET_NAMES = 'act_seqs'

fnames = sorted(listdir(SOURCE_DIR))

seq_data = H5PYDataset(SEQ_FILE, which_sets=('train',), load_in_memory=True)
stream = DataStream.default_stream(dataset=seq_data, iteration_scheme=SequentialScheme(seq_data.num_examples,
                                                                              batch_size=1))

line = 0

state_names = ['sequencegenerator_cost_matrix_states', 'sequencegenerator_cost_matrix_states#1', 'sequencegenerator_cost_matrix_cells', 'sequencegenerator_cost_matrix_cells#1']
final_activations = {k: [] for k in state_names}

max_c = 5
c = 0

for state_name in state_names:
    iterator = stream.get_epoch_iterator()
    for fname in fnames:#[:1]:
        activations = np.load(path.join(SOURCE_DIR, fname)).item()[state_name].swapaxes(0, 1)
        for a in activations:
            c += 1
            l = len(next(iterator)[0][0])
            rel_a = a[:l, :]
            final_activations[state_name].append(rel_a)
            #if c >= max_c: break
        #c = 0

fa = {}
for state_name in state_names:
    fa[state_name] = final_activations[state_name][0]
    for arr in final_activations[state_name][1:]:
        fa[state_name] = np.vstack((fa[state_name], arr))

print(fa[state_names[0]].shape)
for state_name in fa:
    f = h5py.File(state_name+'.hdf5', mode='w')

    fx = f.create_dataset(SET_NAMES, fa[state_name].shape, dtype='float32')

    fx[...] = fa[state_name]

    N = fa[state_name].shape[0]  # FIXME

    split_dict = {
        'train': {SET_NAMES: (0, N)},
    }

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()

