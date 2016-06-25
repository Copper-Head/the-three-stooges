import numpy as np
import matplotlib.pyplot as plt


def hinton_diagram(matrix, max_weight=None, ax=None, xticks=None):
    """Draw Hinton diagram for visualizing a weight matrix.

    Note that depending on the size of your matrix you might have to adjust
    the size of your figure in order to display the squares right.
    """
    if ax is None:
        ax = plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    # Not interested in yaxis ticks
    ax.yaxis.set_major_locator(plt.NullLocator())

    if xticks:
        # Don't want the ticks visible, but labels should be legible
        ax.xaxis.set_tick_params(width=0, labelsize=30)
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(list(xticks))
    else:
        # if no xticks specified kill x axis as well
        ax.xaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        half = size / 2
        rect = plt.Rectangle([x - half, y - half], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    # Not quite sure what these do atm
    ax.autoscale_view()
    return ax
