# coding: utf-8
import warnings

from sklearn.cluster.spectral import *
from abc import ABCMeta, abstractmethod
import six

from spectral_clustering_fd.laplacian_sketch import *

from sklearn.cluster.k_means_ import k_means
from sklearn.cluster.spectral import discretize
from sklearn.utils.extmath import _deterministic_vector_sign_flip

class SpectralClusteringFD(six.with_metaclass(ABCMeta, SpectralClustering)):
    """ Spectral Clustering with FD Sketch.
    This algorithm avoid to store the NxN graph laplacian matrix by replacing
    SVD methods with a matrix sketching method ``Frequent Direction.''
    The memory comsumption with this algorithm should be NxD, where D is
    the original sample dimension.

    Parameters
    -----------
    n_clusters : integer, optional
        The dimension of the projection subspace.
    affinity : string or callable, default 'rbf'
        If a string, this may be one of 'rbf', or 'cosine'.
        If callable, it should be in the same style with laplacian_sketch.laplacian_sketch function style.
    gamma : float, default=1.0
        Scaling factor of RBF kernel. Ignored for
        ``affinity='cosine'``.
    random_state : int seed, RandomState instance, or None (default)
        A pseudo random number generator used for the initialization
        of the lobpcg eigen vectors decomposition when eigen_solver == 'amg'
        and by the K-Means initialization.
    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    assign_labels : {'kmeans', 'discretize'}, default: 'kmeans'
        The strategy to use to assign labels in the embedding
        space. There are two ways to assign labels after the laplacian
        embedding. k-means can be applied and is a popular choice. But it can
        also be sensitive to initialization. Discretization is another approach
        which is less sensitive to random initialization.
    kernel_params : dictionary of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.
    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
    Attributes
    ----------
    affinity_matrix_ : array-like, shape (n_samples, n_samples)
        Affinity matrix used for clustering. Available only if after calling
        ``fit``.
    labels_ :
        Labels of each point
    Notes
    -----
    This class is a subclass of sklearn.clustering.SpectralClustering.
    This class uses matrix sketch to save memory comsumption by NxN
    graph laplacian matrix. This is done by applying matrix sketch to
    NxN graph laplacian iteratively instead of applying SVD to the large
    graph laplacian matrix.
    ----------
    - Frequent Directions: Simple and Deterministic Matrix Sketching, 2013
      Mina Ghashami, Edo Liberty, Jeff M. Phillips, David P. Woodruf
      http://www.cs.utah.edu/~ghashami/papers/fd_journal.pdf
    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324
    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323
    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      http://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf
    """
    def __init__(self, n_clusters=8, random_state=None,
                 n_init=10, gamma=1., affinity='rbf',
                 assign_labels='kmeans',
                 kernel_params=None, n_jobs=1):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.assign_labels = assign_labels
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Creates an affinity matrix for X using the selected affinity,
        then applies spectral clustering to this affinity matrix.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            OR, if affinity==`precomputed`, a precomputed affinity
            matrix of shape (n_samples, n_samples)
        """

        # this class is not tested with sparse matrix.
        # any contribution (report, coding) is welcome!
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=np.float64)

        ell = 2*(self.n_clusters+1) # +1 for drop_first, x2 for zero suppression in frequent_direction.

        if self.affinity == 'rbf':
            self.affinity_matrix_, dd = laplacian_sketch_rbf_kernel(X, ell,gamma=self.gamma)
        elif self.affinity == 'cosine':
            self.affinity_matrix_, dd = laplacian_sketch_cosine_similarity(X, ell)
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if callable(self.affinity):
                self.affinity_matrix_ = self.affinity(X, 2*self.n_clusters,params)
            else:
                warnings.warn("%s is unknown kernel"%self.affinity)

        random_state = check_random_state(self.random_state)


        # spectral embedding post process.
        maps = spectral_embedding_imitation(self.affinity_matrix_,
                                                 dd,
                                                 n_components=self.n_clusters,
                                                 random_state=random_state,
                                                 drop_first=False)

        if self.assign_labels == 'kmeans':
            _, self.labels_, _ = k_means(maps, n_clusters, random_state=random_state,
                               n_init=n_init)
        else:
            self.labels_ = discretize(maps, random_state=random_state)


def spectral_embedding_imitation(graph_laplacian_sketch, dd, n_components=8,
                       random_state=None, norm_laplacian=True, drop_first=True):
    random_state = check_random_state(random_state)

    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1

    embedding = graph_laplacian_sketch.T[:n_components] * dd
    embedding = _deterministic_vector_sign_flip(embedding)
    if drop_first:
        return embedding[1:n_components].T
    else:
        return embedding[:n_components].T
