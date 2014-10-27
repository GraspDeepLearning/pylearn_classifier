

from operator import mul
import h5py
import random
import numpy as np

import pylearn2.datasets.dataset
import pylearn2.utils.rng
from pylearn2.utils.iteration import SubsetIterator , resolve_iterator_class
from pylearn2.utils import safe_izip, wraps

class C01B_HDF5_Dataset(pylearn2.datasets.dataset.Dataset):

    def __init__(self, topo_view_key, y_key, hdf5_filepath):
        h5py_dataset = h5py.File(hdf5_filepath)

        self.topo_view = h5py_dataset[topo_view_key]
        self.y = h5py_dataset[y_key]

    def adjust_for_viewer(self, X):
        return X[0:3, :, :, :]

    def get_batch_design(self, batch_size, include_labels=False):

        if include_labels:
            raise NotImplementedError

        topo_batch = self.get_batch_topo(batch_size)
        return topo_batch.reshape(topo_batch.shape[-1], reduce(mul, topo_batch.shape[:-1]))

    def get_batch_topo(self, batch_size):
        range_start = 0
        range_end = self.topo_view.shape[-1]-batch_size

        batch_start = random.randint(range_start, range_end)
        batch_end = batch_start + batch_size

        return self.topo_view[:, :, :, batch_start:batch_end]

    def get_num_examples(self):
        return self.topo_view.shape[-1]

    def get_topo_batch_axis(self):
        return -1

    def has_targets(self):
        return True

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):
        return HDF5_Iterator(self,
                             batch_size=batch_size,
                             num_batches=num_batches,
                             mode=mode)


class HDF5_Iterator():

    def __init__(self, dataset,  batch_size, num_batches, mode, post_processors=[]):

        def validate_batch_size(batch_size, dataset):
            if not batch_size:
                raise ValueError("batch size is none")

            num_examples = dataset.get_num_examples()
            if batch_size > num_examples:
                raise ValueError("batch size:%i is to large, dataset has %i examples", batch_size, num_examples)

            if batch_size < 0:
                raise ValueError("batch size: %i cannot be negative",batch_size)

            if not isinstance(batch_size, int):
                raise ValueError("batch_size is not an int")

        def validate_num_batches(num_batches):
            if not num_batches:
                raise ValueError("num_batches is none")

            if num_batches < 0:
                raise ValueError("num_batches: %i cannot be negative", num_batches)

            if not isinstance(num_batches, int):
                raise ValueError("num_batches is not an int")

        self.dataset = dataset
        dataset_size = dataset.get_num_examples()

        validate_batch_size(batch_size, dataset)
        validate_num_batches(num_batches)

        subset_iterator_class = resolve_iterator_class(mode)
        self._subset_iterator = subset_iterator_class(dataset_size, batch_size, num_batches)

    def __iter__(self):
        return self

    def next(self):

        next_index = self._subset_iterator.next()

        # if we are using a shuffled sequential subset iterator
        # then next_index will be something like:
        # array([13713, 14644, 30532, 32127, 35746, 44163, 48490, 49363, 52141, 52216])
        # hdf5 can only support this sort of indexing if the array elements are 
        # in increasing order
        if isinstance(next_index, np.ndarray):
            next_index.sort()
            #import IPython
            #IPython.embed()
        batch_x = self.dataset.topo_view[:, :, :, next_index]
        batch_y = self.dataset.y[next_index, :]

        return batch_x, batch_y

    @property
    @wraps(SubsetIterator.batch_size, assigned=(), updated=())
    def batch_size(self):
        return self._subset_iterator.batch_size

    @property
    @wraps(SubsetIterator.num_batches, assigned=(), updated=())
    def num_batches(self):
        return self._subset_iterator.num_batches

    @property
    @wraps(SubsetIterator.num_examples, assigned=(), updated=())
    def num_examples(self):
        return self._subset_iterator.num_examples

    @property
    @wraps(SubsetIterator.uneven, assigned=(), updated=())
    def uneven(self):
        return self._subset_iterator.uneven

    @property
    @wraps(SubsetIterator.stochastic, assigned=(), updated=())
    def stochastic(self):
        return self._subset_iterator.stochastic




