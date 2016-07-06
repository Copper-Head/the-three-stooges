import numpy

from dataproc import load_tab_delim_hdt
from util import _dependencies, simple_mark_dependency


def mark_det_dependency(file_path):
    dep_graphs = load_tab_delim_hdt(file_path)
    dep_dicts = map(_dependencies, dep_graphs)
    return numpy.array([simple_mark_dependency(dd, 'DET') for dd in dep_dicts])
