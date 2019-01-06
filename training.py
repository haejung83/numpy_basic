import loss
import numpy as np


def loss_function():
    t = np.array([0, 0, 1, 0])
    y_matched = np.array([0.1, 0.2, 0.6, 0.1])
    y_not_matched = np.array([0.6, 0.2, 0.1, 0.1])

    print('Mean Squared Error [matched]: {}'.format(loss.mean_squared_error(y_matched, t)))
    print('Mean Squared Error [not matched]: {}'.format(loss.mean_squared_error(y_not_matched, t)))

    print('Cross Entropy Error [matched]: {}'.format(loss.cross_entropy_error(y_matched, t)))
    print('Cross Entropy Error [not matched]: {}'.format(loss.cross_entropy_error(y_not_matched, t)))

if __name__ == '__main__':
    loss_function()