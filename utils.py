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



from keras.legacy import interfaces
from keras.optimizers import Optimizer
import keras.backend as K


class NormalizedSGD(Optimizer):
    def __init__(
            self,
            lr: float=0.01,
            lr_update: float=0.001,
            lr_max: float=0.0001,
            lr_min: float=1e-5,
            lr_force: float=0.0,
            norm: str = 'max',
            **kwargs
    ) -> None:
        """NormalizedSGD constructor

        :param lr: initial learning rate
        :param lr_update: relative learning rate update step. New lr is computed
            approximately as lr' = lr * (1 - lr_update * cos(a)), a is the
            angle between gradient in the k-th step and the k-1 step.
        :param lr_max: max value of lr
        :param lr_min: min value of lr
        :param lr_force: relative force lr to increase in consecutive steps.
            This is achieved following update:
                    lr" = lr' + lr_force * lr_update * lr'

        """
        super(NormalizedSGD, self).__init__(**kwargs)

        if norm not in ['max', 'l2']:
            raise ValueError('Unexpected norm type `{norm}`.')

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

        self.lr = lr
        self.lr_update = lr_update
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_force = lr_force
        self.norm = norm
        self.learning_rates = None
        self.old_grads = None

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        lr = self.lr
        lr_update = self.lr_update
        lr_force = self.lr_force

        shapes = [K.int_shape(p) for p in params]
        learning_rates = [K.variable(lr) for _ in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        self.learning_rates = learning_rates
        self.old_grads = old_grads

        self.weights = [self.iterations]
        for p, g, l, old_g in zip(params, grads, learning_rates, old_grads):

            if self.norm == 'max':
                g_max = K.max(K.abs(g), axis=None, keepdims=True)
                denominator = K.epsilon() + g_max
                g_step_normed = g / denominator
            else:
                g_step_normed = K.l2_normalize(g)

            # update parameters with SGD
            new_p = p - l * g_step_normed
            self.updates.append(K.update(p, new_p))

            # update learning rate
            g_normed = K.l2_normalize(g)
            old_g_normed = K.l2_normalize(old_g)

            lr_change = - lr_update * K.sum(g_normed * old_g_normed)
            new_lr = l * (1 - lr_change) + lr_force * l * lr_update
            new_lr = K.clip(new_lr, self.lr_min, self.lr_max)

            self.updates.append(K.update(l, new_lr))
            self.updates.append(K.update(old_g, g))

        return self.updates

    def get_config(self):
        config = {
            'lr': self.lr,
            'lr_update': self.lr_update,
            'lr_max': self.lr_max,
            'lr_min': self.lr_min,
            'lr_force': self.lr_force,
            'norm': self.norm
        }
        base_config = super(NormalizedSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

