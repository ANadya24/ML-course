import numpy as np
import numpy.matlib


def prod_non_zero_diag(x):
    a = np.diag(x)
    return np.prod(a[a != 0])


def are_multisets_equal(x, y):
    x = np.sort(x)
    y = np.sort(y)
    return np.array_equal(x, y)


def max_after_zero(x):
    y = np.where(x == 0)
    idx = np.array(y) + 1
    idx = idx[idx < x.size]
    return np.max([x[idx]])


def convert_image(img, vec):
    im = np.dot(img, vec)
    return im


def run_length_encoding(x):
    d = np.diff(x)
    nums = x[np.insert(d.astype(np.bool), 0, True)]
    idx = np.array(np.where(d != 0))
    idx = idx + 1
    idx = np.insert(idx, 0, 0)
    idx = np.append(idx, len(x))
    idxx = idx[1:]
    counts = idxx - idx[0:-1]
    return (nums, counts)


def pairwise_distance(x, y):
    a = np.zeros((x.shape[0], y.shape[0]))
    X = np.repeat(x, (y.shape[0] + np.zeros(x.shape[0]))
                  .astype(np.int), axis=0)
    Y = np.matlib.repmat(y, x.shape[0], 1)
    a = ((X - Y) ** 2)
    a = (np.sum(a, axis=1)) ** 0.5
    a = a.reshape((x.shape[0], y.shape[0]))
    return a

