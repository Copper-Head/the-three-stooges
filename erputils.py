import numpy


def var_pos_mean(value_list, var_name, pos_indx=None):
    """Retrieves and averages variable's value given it's name and some index.

    value_list is assumed to be a list of StateComputer.read_single_sequence results.
    pos_indx can in theory be a range of indices.
    """
    var_pos_vals = [v[var_name] for v in value_list]
    if pos_indx:
        var_pos_vals = [v[pos_indx] for v in var_pos_vals]
    return numpy.mean(var_pos_vals, axis=0)


class Char2IdSeqGen(object):
    """docstring for Char2IDWrapper"""

    def __init__(self, char_2_id):
        self.char_2_id = char_2_id
        self.period_id = char_2_id['.']

    def subset_char_ids(self, chars):
        """Return all IDs except the one(s) for chars.

        chars arg can be a single string or a container of strings.
        """
        if isinstance(chars, str):
            return [self.char_2_id[c] for c in self.char_2_id if c != chars]
        return [self.char_2_id[c] for c in self.char_2_id if c not in chars]

    def period_pad_sequence(self, to_pad):
        if isinstance(to_pad, int):
            return numpy.array([to_pad, self.period_id])
        return numpy.append(to_pad, self.period_id)
