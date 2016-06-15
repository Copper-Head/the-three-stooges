import logging

import numpy
from blocks.extensions.monitoring import SimpleExtension, MonitoringExtension
from blocks.initialization import NdarrayInitialization
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.serialization import secure_dump, dump_and_add_to_dump
from fuel.transformers import Transformer


logger = logging.getLogger(__name__)  # no idea what this is dude
SAVED_TO = "saved_to"


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
        """
        Because the transformer adds a source on top of what is in the dataset (namely the mask), we need to define
        this.
        """
        return "character_seqs", "seq_mask"

    def transform_batch(self, batch):
        maxlen = max(example.shape[0] for example in batch[0])
        # For one, zero-pad the sequences
        # Also build the mask
        # Note that, because the transformer doesn't do enough stuff yet, the sequences (batch[0]) are also transposed
        # to go from one-sequence-per-row (the way it's stored) to the weird Blocks RNN format (one-sequence-per-column)
        padded_seqs = numpy.zeros((batch[0].shape[0], maxlen), dtype="int32")
        # mask is int8 because apparently it's converted to float later, and higher ints would complain about loss of
        # precision
        mask = numpy.zeros((batch[0].shape[0], maxlen), dtype="int8")
        for (example_ind, example) in enumerate(batch[0]):
            # we go through the sequences and simply put them into the padded array; this will leave 0s wherever the
            # sequence is shorter than maxlen. similarly, mask will be set to 1 only up to the length of the respective
            # sequence
            # note that the transpose is done implicitly by essentially swapping indices
            padded_seqs[example_ind, :len(example)] = example
            mask[example_ind, :len(example)] = 1
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


class EarlyStopping(SimpleExtension, MonitoringExtension):
    """
    Extension that tracks some quantity (PLEASE use cost on the validation set here!) and creates a checkpoint whenever
    a new best-value-so-far has been achieved. In addition, training is stopped if performance hasn't improved since n
    epochs. Note: "Improvement" is not needed over the current best value, but only over the last time we measured.

    You can call this as often as you like. Default is after every epoch. Note that if you want to call it more often
    (e.g. using every_n_batches), you might wanna increase n (i.e. the tolerance for no improvement) accordingly.

    ALSO NOTE that this will combine both a DatastreamMonitoring on the passed quantity as well as a kinda FinishAfter
    extension, so you don't need to pass those to the MainLoop. Of course you still could -- another monitoring would
    just do the work again, which isn't that dramatic (albeit useless), while an "explicit" FinishAfter could be used to
    set a maximum on the number of epochs we want to train for, in addition to the early stopping.
    """
    def __init__(self, variables, data_stream, path, tolerance, updates=None, parameters=None, save_separately=None,
                 save_main_loop=True, use_cpickle=False, **kwargs):
        if len(variables) != 1:
            raise ValueError("Please specify exactly one variable (packed in a list)!")
        # DatastreamMonitoring part
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("before_first_epoch", True)
        super(EarlyStopping, self).__init__(**kwargs)
        self._evaluator = DatasetEvaluator(variables, updates)
        self.data_stream = data_stream
        # Checkpoint part
        super(EarlyStopping, self).__init__(**kwargs)
        self.path = path
        self.parameters = parameters
        self.save_separately = save_separately
        self.save_main_loop = save_main_loop
        self.use_cpickle = use_cpickle
        # additional quantities for early stopping
        self.tolerance = tolerance
        self.n_since_improvement = 0
        self.last_value = float("inf")
        self.best_value = float("inf")

    def do(self, callback_name, *args):
        value_dict = self.do_monitoring()
        # see if we have an improvement from the last measure
        current_value = value_dict.values()[0]
        if current_value < self.last_value:
            # if so, we reset our counter of how-long-since-improvement
            logger.info("Got an improvement from the last measure; tolerance reset.")
            self.n_since_improvement = 0
        else:
            self.n_since_improvement += 1
            if self.tolerance - self.n_since_improvement >= 0:
                logger.info("No improvement since the last measure! Tolerating " +
                            str(self.tolerance - self.n_since_improvement) + " more...")
        self.last_value = current_value
        # if we have an improvement over the best yet, we store that and make a checkpoint
        if current_value < self.best_value:
            logger.info("Got a new best value! Saving model...")
            self.do_checkpoint(callback_name, *args)
            self.best_value = current_value
        # went too long without improvement? Giving up is the obvious solution.
        if self.n_since_improvement > self.tolerance:
            logger.info("Thou hast exceeded that tolerance of mine for thy miserable performance!")
            # Note that the way this comparison is set up enforces the following interpretation on the tolerance
            # parameter: The number of times in a row we will accept no improvement. E.g. if it set to 0, we will stop
            # if we *ever* get no improvement on the tracked value.
            self.main_loop.log.current_row['training_finish_requested'] = True

    def do_monitoring(self):
        logger.info("Monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("Monitoring on auxiliary data finished")
        return value_dict

    def do_checkpoint(self, callback_name, *args):
        logger.info("Checkpointing has started")
        _, from_user = self.parse_args(callback_name, args)
        try:
            path = self.path
            if from_user:
                path, = from_user
            to_add = None
            if self.save_separately:
                to_add = {attr: getattr(self.main_loop, attr) for attr in
                          self.save_separately}
            if self.parameters is None:
                if hasattr(self.main_loop, 'model'):
                    self.parameters = self.main_loop.model.parameters
            object_ = None
            if self.save_main_loop:
                object_ = self.main_loop
            secure_dump(object_, path,
                        dump_function=dump_and_add_to_dump,
                        parameters=self.parameters,
                        to_add=to_add,
                        use_cpickle=self.use_cpickle)
        except Exception:
            path = None
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (path,))
            logger.info("Checkpointing has finished")
