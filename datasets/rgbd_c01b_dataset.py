
import h5py
import numpy as np
import warnings
import os
import functools

from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter

from pylearn2.utils import safe_zip
import pylearn2.utils.one_hot
from pylearn2.utils.data_specs import is_flat_specs
from pylearn2.utils.iteration import safe_izip, wraps, SubsetIterator, resolve_iterator_class

import pylearn2.space
from pylearn2.space import CompositeSpace, VectorSpace


PYLEARN_DATA_PATH = os.environ["PYLEARN2_DATA_PATH"]

#these need to go away!
#NUM_TRAINING_SAMPLES = 207000
#NUM_LABELS = 300

#NUM_TRAINING_SAMPLES = 100000
#NUM_LABELS = 894

NUM_TRAINING_SAMPLES = 100000
NUM_LABELS = 3

def get_dataset(which_set='train', dataset_filepath="/nyu_depth_labeled/rgbd_preprocessed_72x72.h5"):

    hdf5_dataset_filename = dataset_filepath
    print "hdf5_dataset_filename: " + hdf5_dataset_filename
    X = which_set + "_patches"
    y = which_set + "_patch_labels"

    return HDF5Dataset(hdf5_dataset_filename, topo_view=X, y=y)


class HDF5Dataset(DenseDesignMatrix):
    """
    Dense dataset loaded from an HDF5 file.

    Parameters
    ----------
    filename : str
        HDF5 file name.
    X : str, optional
        Key into HDF5 file for dataset design matrix.
    topo_view: str, optional
        Key into HDF5 file for topological view of dataset.
    y : str, optional
        Key into HDF5 file for dataset targets.
    load_all : bool, optional (default False)
        If true, datasets are loaded into memory instead of being left
        on disk.
    kwargs : dict, optional
        Keyword arguments passed to `DenseDesignMatrix`.
    """
    def __init__(self, filename, X=None, topo_view=None, y=None,
                 load_all=False, batch_size=100, **kwargs):
        self.batch_size = batch_size
        self.load_all = load_all

        self._file = h5py.File(filename)
        if X is not None:
            X = self.get_dataset(X, load_all)
        if topo_view is not None:
            topo_view = self.get_dataset(topo_view, load_all)
        if y is not None:
            y = self.get_dataset(y, load_all)

        super(HDF5Dataset, self).__init__(X=X, topo_view=topo_view, y=y, axes=('c', 0, 1, 'b'),
                                          **kwargs)

    def _check_labels(self):
        """
        Sanity checks for X_labels and y_labels.

        Since the np.all test used for these labels does not work with HDF5
        datasets, we issue a warning that those values are not checked.
        """
        if self.X_labels is not None:
            assert self.X is not None
            assert self.view_converter is None
            assert self.X.ndim <= 2
            if self.load_all:
                assert np.all(self.X < self.X_labels)
            else:
                warnings.warn("HDF5Dataset cannot perform test np.all(X < " +
                              "X_labels). Use X_labels at your own risk.")

        if self.y_labels is not None:
            assert self.y is not None
            assert self.y.ndim <= 2
            if self.load_all:
                assert np.all(self.y < self.y_labels)
            else:
                warnings.warn("HDF5Dataset cannot perform test np.all(y < " +
                              "y_labels). Use y_labels at your own risk.")

    def get_dataset(self, dataset, load_all=False):
        """
        Get a handle for an HDF5 dataset, or load the entire dataset into
        memory.

        Parameters
        ----------
        dataset : str
            Name or path of HDF5 dataset.
        load_all : bool, optional (default False)
            If true, load dataset into memory.
        """
        if load_all:
            data = self._file[dataset][:]
        else:
            data = self._file[dataset]
            data.ndim = len(data.shape)  # hdf5 handle has no ndim
        return data



    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        if topo is not None or targets is not None:
            if data_specs is not None:
                raise ValueError('In DenseDesignMatrix.iterator, both the '
                                 '"data_specs" argument and deprecated '
                                 'arguments "topo" or "targets" were '
                                 'provided.',
                                 (data_specs, topo, targets))

            warnings.warn("Usage of `topo` and `target` arguments are "
                          "being deprecated, and will be removed "
                          "around November 7th, 2013. `data_specs` "
                          "should be used instead.",
                          stacklevel=2)

            # build data_specs from topo and targets if needed
            if topo is None:
                topo = getattr(self, '_iter_topo', False)
            if topo:
                # self.iterator is called without a data_specs, and with
                # "topo=True", so we use the default topological space
                # stored in self.X_topo_space
                assert self.X_topo_space is not None
                X_space = self.X_topo_space
            else:
                X_space = self.X_space

            if targets is None:
                targets = getattr(self, '_iter_targets', False)
            if targets:
                assert self.y is not None
                y_space = self.data_specs[0].components[1]
                space = CompositeSpace((X_space, y_space))
                source = ('features', 'targets')
            else:
                space = X_space
                source = 'features'

            data_specs = (space, source)
            convert = None

        else:
            if data_specs is None:
                data_specs = self._iter_data_specs

            # If there is a view_converter, we have to use it to convert
            # the stored data for "features" into one that the iterator
            # can return.
            space, source = data_specs
            if isinstance(space, CompositeSpace):
                sub_spaces = space.components
                sub_sources = source
            else:
                sub_spaces = (space,)
                sub_sources = (source,)

            convert = []
            for sp, src in safe_zip(sub_spaces, sub_sources):
                if src == 'features' and \
                   getattr(self, 'view_converter', None) is not None:
                    conv_fn = (lambda batch, self=self, space=sp:
                               self.view_converter.get_formatted_batch(batch,
                                                                       space))
                elif src == "targets":
                    #######################################################
                    #this was added to to expand the y's just when we need them
                    dspace = pylearn2.space.VectorSpace(NUM_LABELS)

                    def fn(batch, dspace=dspace, sp=sp):
                        try:

                            #comment out this line if switching back to dense scene labeling with many categories
                            return batch

                            batch_2 = pylearn2.utils.one_hot.one_hot(batch.astype(int), max_label=NUM_LABELS-1)
                            #return dspace.np_format_as(batch, sp)
                            return dspace.np_format_as(batch_2, sp)

                        except ValueError as e:
                            msg = str(e) + '\nMake sure that the model and '\
                                           'dataset have been initialized with '\
                                           'correct values.'
                            raise ValueError(msg)
                    conv_fn = fn
                    #########################################################
                else:
                    conv_fn = None

                convert.append(conv_fn)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng

        return HDF5DatasetIterator(self,
                                     mode(self.X.shape[0],
                                          batch_size,
                                          num_batches,
                                          rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)

    def set_topological_view(self, V, axes=('c', 0, 1, 'b')):
        """
        Set up dataset topological view, without building an in-memory
        design matrix.

        This is mostly copied from DenseDesignMatrix, except:
        * HDF5ViewConverter is used instead of DefaultViewConverter
        * Data specs are derived from topo_view, not X
        * NaN checks have been moved to HDF5DatasetIterator.next
        * Support for "old pickled models" is dropped.

        Note that y may be loaded into memory for reshaping if y.ndim != 2.

        Parameters
        ----------
        V : ndarray
            Topological view.
        axes : tuple, optional (default ('b', 0, 1, 'c'))
            Order of axes in topological view.
        """
        shape = [V.shape[axes.index('c')],
                 V.shape[axes.index(0)],
                 V.shape[axes.index(1)],
                 V.shape[axes.index('b')]]
        self.view_converter = HDF5ViewConverter(shape[:-1], axes=axes)
        self.X = self.view_converter.topo_view_to_design_mat(V)
        # self.X_topo_space stores a "default" topological space that
        # will be used only when self.iterator is called without a
        # data_specs, and with "topo=True", which is deprecated.
        self.X_topo_space = self.view_converter.topo_space

        # Update data specs
        X_space = VectorSpace(dim=V.shape[axes.index('b')])
        X_source = 'features'

        if self.y is None:
            space = X_space
            source = X_source
        else:
            if self.y.ndim == 1:
                dim = 1
            else:
                dim = self.y.shape[-1]
            y_space = VectorSpace(dim=dim)
            y_source = 'targets'
            space = CompositeSpace((X_space, y_space))
            source = (X_source, y_source)

        self.data_specs = (space, source)
        self.X_space = X_space
        self._iter_data_specs = (X_space, X_source)

    def set_design_matrix(self, X, start):
        self.X[start:start+self.batch_size] = X


class HDF5DatasetIterator(object):

    def __init__(self, dataset, subset_iterator, data_specs=None,
                 return_tuple=False, convert=None):
        self._data_specs = data_specs
        self._dataset = dataset
        self._subset_iterator = subset_iterator
        self._return_tuple = return_tuple

        # Keep only the needed sources in self._raw_data.
        # Remember what source they correspond to in self._source
        assert is_flat_specs(data_specs)

        dataset_space, dataset_source = self._dataset.get_data_specs()
        assert is_flat_specs((dataset_space, dataset_source))

        # the dataset's data spec is either a single (space, source) pair,
        # or a pair of (non-nested CompositeSpace, non-nested tuple).
        # We could build a mapping and call flatten(..., return_tuple=True)
        # but simply putting spaces, sources and data in tuples is simpler.
        if not isinstance(dataset_source, tuple):
            dataset_source = (dataset_source,)

        if not isinstance(dataset_space, CompositeSpace):
            dataset_sub_spaces = (dataset_space,)
        else:
            dataset_sub_spaces = dataset_space.components
        assert len(dataset_source) == len(dataset_sub_spaces)

        all_data = self._dataset.get_data()
        if not isinstance(all_data, tuple):
            all_data = (all_data,)

        space, source = data_specs
        if not isinstance(source, tuple):
            source = (source,)
        if not isinstance(space, CompositeSpace):
            sub_spaces = (space,)
        else:
            sub_spaces = space.components
        assert len(source) == len(sub_spaces)

        self._raw_data = tuple(all_data[dataset_source.index(s)]
                               for s in source)
        self._source = source

        if convert is None:
            self._convert = [None for s in source]
        else:
            assert len(convert) == len(source)
            self._convert = convert

        for i, (so, sp, dt) in enumerate(safe_izip(source,
                                                   sub_spaces,
                                                   self._raw_data)):
            idx = dataset_source.index(so)
            dspace = dataset_sub_spaces[idx]

            init_fn = self._convert[i]
            fn = init_fn

            # If there is an init_fn, it is supposed to take
            # care of the formatting, and it should be an error
            # if it does not. If there was no init_fn, then
            # the iterator will try to format using the generic
            # space-formatting functions.
            if init_fn is None:
                # "dspace" and "sp" have to be passed as parameters
                # to lambda, in order to capture their current value,
                # otherwise they would change in the next iteration
                # of the loop.

                if fn is None:

                    def fn(batch, dspace=dspace, sp=sp):
                        try:
                              return dspace.np_format_as(batch, sp)
                        except ValueError as e:
                            msg = str(e) + '\nMake sure that the model and '\
                                           'dataset have been initialized with '\
                                           'correct values.'
                            raise ValueError(msg)
                else:
                    fn = (lambda batch, dspace=dspace, sp=sp, fn_=fn:
                          dspace.np_format_as(fn_(batch), sp))

            self._convert[i] = fn
    """
    Dataset iterator for HDF5 datasets.

    FiniteDatasetIterator expects a design matrix to be available, but this
    will not always be the case when using HDF5 datasets with topological
    views.

    Parameters
    ----------
    dataset : Dataset
        Dataset over which to iterate.
    subset_iterator : object
        Iterator that returns slices of the dataset.
    data_specs : tuple, optional
        A (space, source) tuple.
    return_tuple : bool, optional (default False)
        Whether to return a tuple even if only one source is used.
    convert : list, optional
        A list of callables (in the same order as the sources in
        data_specs) that will be applied to each slice of the dataset.
    """
    def next(self):
        """
        Get the next subset of the dataset during dataset iteration.

        Converts index selections for batches to boolean selections that
        are supported by HDF5 datasets.
        """
        next_index = self._subset_iterator.next()

        # convert to boolean selection
        #sel = np.zeros(self.num_examples, dtype=bool)
        sel = np.zeros(NUM_TRAINING_SAMPLES, dtype=bool)
        sel[next_index] = True
        next_index = sel

        rval = []
        for data, fn in safe_izip(self._raw_data, self._convert):
            try:
                this_data = data[next_index]
            except TypeError:
                this_data = data[next_index, :]
            if fn:
                #import IPython
                #IPython.embed()
                this_data = fn(this_data)

            assert not np.any(np.isnan(this_data))
            rval.append(this_data)
        rval = tuple(rval)
        if not self._return_tuple and len(rval) == 1:
            rval, = rval

        #import IPython
        #IPython.embed()
        rval = (np.rollaxis(rval[0], 1), rval[1])
        return rval

    def __iter__(self):
        return self

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


class HDF5ViewConverter(DefaultViewConverter):
    """
    View converter that doesn't have to transpose the data.

    In order to keep data on disk, does not generate a full design matrix.
    Instead, an instance of HDF5TopoViewConverter is returned, which
    transforms data from the topological view into the design view for each
    batch.

    Parameters
    ----------
    shape : tuple
        Shape of this view.
    axes : tuple, optional (default ('b', 0, 1, 'c'))
        Order of axes in topological view.
    """
    def topo_view_to_design_mat(self, V):
        """
        Generate a design matrix from the topological view.

        This override of DefaultViewConverter.topo_view_to_design_mat does
        not attempt to transpose the topological view, since transposition
        is not supported by HDF5 datasets.
        """
        v_shape = (V.shape[self.axes.index('c')],
                   V.shape[self.axes.index(0)],
                   V.shape[self.axes.index(1)],
                   V.shape[self.axes.index('b')])

        # if np.any(np.asarray(self.shape) != np.asarray(v_shape[1:])):
        #     raise ValueError('View converter for views of shape batch size '
        #                      'followed by ' + str(self.shape) +
        #                      ' given tensor of shape ' + str(v_shape))

        rval = HDF5TopoViewConverter(V, self.axes)
        return rval


class HDF5TopoViewConverter(object):
    """
    Class for transforming batches from the topological view to the design
    matrix view.

    Parameters
    ----------
    topo_view : HDF5 dataset
        On-disk topological view.
    axes : tuple, optional (default ('b', 0, 1, 'c'))
        Order of axes in topological view.
    """
    def __init__(self, topo_view, axes=('c', 0, 1, 'b')):
        self.topo_view = topo_view
        self.axes = axes
        self.topo_view_shape = (topo_view.shape[axes.index('c')],
                                topo_view.shape[axes.index(0)],
                                topo_view.shape[axes.index(1)],
                                topo_view.shape[axes.index('b')])
        self.pixels_per_channel = (self.topo_view_shape[1] *
                                   self.topo_view_shape[2])
        self.n_channels = self.topo_view_shape[0]
        self.shape = (self.topo_view_shape[3],
                      np.product(self.topo_view_shape[0:3]))
        self.ndim = len(self.shape)
        #import IPython
        #IPython.embed()

    def __getitem__(self, item):
        """
        Indexes the design matrix and transforms the requested batch from
        the topological view.

        Parameters
        ----------
        item : slice or ndarray
            Batch selection. Either a slice or a boolean mask.
        """
        sel = [slice(None)] * len(self.topo_view_shape)
        sel[self.axes.index('b')] = item
        sel = tuple(sel)

        V = self.topo_view[sel]
        batch_size = V.shape[self.axes.index('b')]
        rval = np.zeros((batch_size,
                         self.pixels_per_channel * self.n_channels),
                        dtype=V.dtype)
        for i in xrange(self.n_channels):
            ppc = self.pixels_per_channel
            sel = [slice(None)] * len(V.shape)
            sel[self.axes.index('c')] = i
            sel = tuple(sel)
            rval[:, i * ppc:(i + 1) * ppc] = V[sel].reshape(batch_size, ppc)

        return rval

