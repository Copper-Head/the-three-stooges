import numpy

from dataproc import load_tab_delim_hdt
from util import dependencies, simple_mark_dependency


def mark_det_dependency(file_path):
    dep_graphs = load_tab_delim_hdt(file_path)
    dep_dicts = map(dependencies, dep_graphs)
    return numpy.array([simple_mark_dependency(dd, 'DET') for dd in dep_dicts])
