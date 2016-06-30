import numpy as np
import h5py

from fuel.datasets import H5PYDataset


def split_hdf5_file(file_path, train_data, val_data, varlen=False):
    """Generates a split HDF5 file.
    Args:
    file_path -> string, filepath to save the dataset
    train_data -> numpy array 2D. first dim is over number of sequences,
    second dim (which can vary) is over sequence length.
    In theory can also be a list of numpy arrays.
    val_data -> same as train_data, but validation.
    varlen -> boolean flag indicates whether or not data has variable length.
    """

    if varlen:
        our_dtype = h5py.special_dtype(vlen=np.dtype("int32"))
    else:
        our_dtype = "int32"

    # We have to combine these as lists in order to handle variable len data
    all_data = list(train_data) + list(val_data)
    split_at = len(train_data)
    data_size = len(all_data)

    with h5py.File(file_path, mode="w") as f:
        dataset = f.create_dataset("character_seqs", (data_size,), dtype=our_dtype)
        dataset[...] = all_data

        split_dict = {"train": {"character_seqs": (0, split_at)},
                      "valid": {"character_seqs": (split_at, data_size)}}
        f.attrs["split"] = H5PYDataset.create_split_array(split_dict)


def random_train_val_split(data, val_size):
    """Splits data into training and validation sets randomly.
    Expets a numpy array `data` argument and an int `val_size` specifying how big
    the validation size should be.
    Returns tuple (training_data, validation_data)
    """

    val_indices = np.random.choice(a=len(data), replace=False, size=val_size)
    val_set = data[val_indices]
    train_set = np.delete(data, val_indices, 0)
    return (train_set, val_set)