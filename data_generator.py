import torch
import helper
import numpy as np
import pickle


class DataGenerator:
    def __init__(self, config):
        self.device = config.device
        self.iter = None # iterator of the data loader
        # load data here
        graphs, labels, train_idx, valid_idx, test_idx = pickle.load(
            open('data/graph_' + config.dataset_name + '.pkl', 'rb'))

        # change labels
        labels = np.array(labels).reshape(-1, config.num_classes) # column vector
        print('shape of labels:', labels.shape)
        train_idx = np.array(train_idx)
        print('shape of train_idx:', train_idx.shape)
        valid_idx = np.array(valid_idx)
        print('shape of valid_idx:', valid_idx.shape)
        test_idx = np.array(test_idx)
        print('shape of test_idx:', test_idx.shape)
        self.batch_size = config.hyperparams.batch_size
        self.labels_dtype = torch.long # type of labels
        self.train_graphs, self.train_labels = graphs[train_idx], labels[train_idx]
        self.valid_graphs, self.valid_labels = graphs[valid_idx], labels[valid_idx]
        self.test_graphs, self.test_labels = graphs[test_idx], labels[test_idx]

        self.num_iterations_train = None
        # Every batch has samples of the the same shape
        self.num_iterations_val, self.val_graphs_batches, self.val_labels_batches = [None] * 3
        self.num_iterations_test, self.test_graphs_batches, self.test_labels_batches = [None] * 3
        self.split_val_test_to_batches()

        self.train_size = len(self.train_graphs)
        self.val_size = len(self.valid_graphs)
        self.test_size = len(self.test_graphs)

    def next_batch(self):
        graphs, labels = next(self.iter)
        graphs = torch.cuda.FloatTensor(graphs) if self.device == "cuda" else torch.FloatTensor(graphs)
        labels = torch.tensor(labels, device=self.device, dtype=self.labels_dtype)
        # delete inf and nan
        '''
        where_are_nan = torch.isnan(graphs)
        where_are_inf = torch.isinf(graphs)
        graphs[where_are_nan] = 0.
        graphs[where_are_inf] = 0.
        
        #change labels to 0 and 1
        labels[labels==1]=0
        labels[labels!=0]=1
        '''
        return graphs, labels

    # initialize an iterator from the data for one training epoch
    def initialize(self, which_set):
        if which_set == 'train':
            self.reshuffle_data()
        elif which_set == 'val' or which_set == 'validation':
            self.iter = zip(self.val_graphs_batches, self.val_labels_batches)
        elif which_set == 'test':
            self.iter = zip(self.test_graphs_batches, self.test_labels_batches)
        else:
            raise ValueError("what_set should be either 'train', 'val' or 'test'")

    def reshuffle_data(self):
        """
        Reshuffle train data between epochs
        """
        graphs, labels = helper.group_same_size(self.train_graphs, self.train_labels)
        graphs, labels = helper.shuffle_same_size(graphs, labels)
        graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_train = len(graphs)
        graphs, labels = helper.shuffle(graphs, labels)
        self.iter = zip(graphs, labels)

    def split_val_test_to_batches(self):
        # Split the val and test sets to batchs, no shuffling is needed
        graphs, labels = helper.group_same_size(self.valid_graphs, self.valid_labels)
        graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_val = len(graphs)
        self.val_graphs_batches, self.val_labels_batches = graphs, labels

        graphs, labels = helper.group_same_size(self.test_graphs, self.test_labels)
        graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_test = len(graphs)
        self.test_graphs_batches, self.test_labels_batches = graphs, labels
