import numpy as np
import matplotlib.pyplot as plt


def hinton_diagram(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix.

    Note that depending on the size of your matrix you might have to adjust
    the size of your figure in order to display the squares right.
    """
    if ax is None:
        ax = plt.gca()

    max_weight = np.abs(matrix).max()
    # if not max_weight:
    #     max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.tanh(np.abs(w)) #/ max_weight
        half = size / 2
        # half = 0.5
        rect = plt.Rectangle([x - half, y - half], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    # Not quite sure what these do atm
    ax.autoscale_view()
    # ax.invert_yaxis()
