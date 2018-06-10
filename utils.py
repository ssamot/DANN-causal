import numpy as np


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]






def load_data(fname):
    # Data Loading

    data = np.loadtxt(open(fname, 'rb'), delimiter=',')
    xdata = data[:, 5:]
    t = data[:, 0]
    yf = data[:, 1]
    ycf = data[:, 2]

    m0 = data[:, 3]
    m1 = data[:, 4]

    return xdata, t, yf, ycf, m0, m1



# def batch_generator(data, batch_size):
#     """Generate batches of data.
#
#     Given a list of numpy data, it iterates over the list and returns batches of the same size
#     """
#     all_examples_indices = len(data[0])
#     while True:
#         mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=False)
#         tbr = [k[mini_batch_indices] for k in data]
#         yield tbr

