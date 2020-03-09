import numpy as np


def group_same_size(graphs, labels):
    """
    group graphs of same size to same array
    :param graphs: numpy array of shape (num_of_graphs) of numpy arrays of graphs adjacency matrix
    :param labels: numpy array of labels
    :return: two numpy arrays.
        graphs arrays in the shape (num of different size graphs) where each entry is a numpy array
        in the shape (number of graphs with this size, num vertex labels, num vertex, num. vertex)
        the second array is labels with corresponding shape
    """
    sizes = list(map(lambda t: t.shape[1], graphs))
    indexes = np.argsort(sizes)
    graphs = graphs[indexes]
    labels = labels[indexes]
    r_graphs = []
    r_labels = []
    one_size = []
    start = 0
    size = graphs[0].shape[1]
    for i in range(len(graphs)):
        if graphs[i].shape[1] == size:
            one_size.append(np.expand_dims(graphs[i], axis=0))
        else:
            r_graphs.append(np.concatenate(one_size, axis=0))
            r_labels.append(np.array(labels[start:i]))
            start = i
            one_size = []
            size = graphs[i].shape[1]
            one_size.append(np.expand_dims(graphs[i], axis=0))
    r_graphs.append(np.concatenate(one_size, axis=0))
    r_labels.append(np.array(labels[start:]))
    return r_graphs, r_labels


# helper method to shuffle each same size graphs array
def shuffle_same_size(graphs, labels):
    r_graphs, r_labels = [], []
    for i in range(len(labels)):
        curr_graph, curr_labels = shuffle(graphs[i], labels[i])
        r_graphs.append(curr_graph)
        r_labels.append(curr_labels)
    return r_graphs, r_labels


def split_to_batches(graphs, labels, size):
    """
    split the same size graphs array to batches of specified size
    last batch is in size num_of_graphs_this_size % size
    :param graphs: array of arrays of same size graphs
    :param labels: the corresponding labels of the graphs
    :param size: batch size
    :return: two arrays. graphs array of arrays in size (batch, num vertex, num vertex. num vertex labels)
                corresponds labels
    """
    r_graphs = []
    r_labels = []
    for k in range(len(graphs)):
        r_graphs = r_graphs + np.split(graphs[k], [j for j in range(size, graphs[k].shape[0], size)])
        r_labels = r_labels + np.split(labels[k], [j for j in range(size, labels[k].shape[0], size)])

    # Avoid bug for batch_size=1, where instead of creating numpy array of objects, we had numpy array of floats with
    # different sizes - could not reshape
    ret1, ret2 = np.empty(len(r_graphs), dtype=object), np.empty(len(r_labels), dtype=object)
    ret1[:] = r_graphs
    ret2[:] = r_labels
    return ret1, ret2


# helper method to shuffle the same way graphs and labels arrays
def shuffle(graphs, labels):
    shf = np.arange(labels.shape[0], dtype=np.int32)
    np.random.shuffle(shf)
    return np.array(graphs)[shf], labels[shf]
