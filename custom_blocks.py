import numpy
from blocks.initialization import NdarrayInitialization
from fuel.transformers import Transformer


class PadAndAddMasks(Transformer):
    """
    If you don't know what transformers are or what they do or how they work, but would like to, either try finding
    something in the Fuel docs or ask me.
    """
    def __init(self, data_stream, **kwargs):
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(PadAndAddMasks, self).__init__(data_stream, data_stream.produces_examples, **kwargs)

    @property
    def sources(self):
        # because the transformer adds a source on top of what is in the dataset (namely the mask), we need to define
        # this
        return "character_seqs", "seq_mask"

    def transform_batch(self, batch):
        maxlen = max(example.shape[0] for example in batch[0])
        # for one, zero-pad the sequences
        # also build the mask
        # note that, because the transformer doesn't do enough stuff yet, the sequences (batch[0]) are also transposed
        # to go from one-sequence-per-row (the way it's stored) to the weird Blocks RNN format (one-sequence-per-column)
        padded_seqs = numpy.zeros((maxlen, batch[0].shape[0]), dtype="int32")
        # mask is int8 because apparently it's converted to float later, and higher ints would complain about loss of
        # precision
        mask = numpy.zeros((maxlen, batch[0].shape[0]), dtype="int8")
        for (example_ind, example) in enumerate(batch[0]):
            # we go through the sequences and simply put them into the padded array; this will leave 0s wherever the
            # sequence is shorter than maxlen. similarly, mask will be set to 1 only up to the length of the respective
            # sequence
            # note that the transpose is done implicitly by essentially swapping indices
            padded_seqs[:len(example), example_ind] = example
            mask[:len(example), example_ind] = 1
        return padded_seqs, mask


class LSTMGateBias(NdarrayInitialization):
    """
    >>Might as well ignore this at the moment!<<

    Takes a normal initialization but biases the parts that corresponds to the gates by given values.

    The "forget_value" should be positive; the "inp_out_value" should be negative. Values could e.g. be something
    between 1-5 in absolute value. Supposedly, the precise values don't matter much..."""
    def __init__(self, init, forget_value=3, inp_out_value=-3):
        self.init = init
        self.forget_value = forget_value
        self.inp_out_value = inp_out_value

    def generate(self, rng, shape):
        weights = self.init.generate(rng, shape)
        # len(weights) is bound to be divisible by 4, else you did something wrong
        weights[(len(weights)/4):(len(weights)/2)] += self.forget_value
        weights[:(len(weights)/4)] += self.inp_out_value
        weights[(3*len(weights)/4):] += self.inp_out_value
        return weights
