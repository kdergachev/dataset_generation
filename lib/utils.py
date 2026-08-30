import numpy as np
import scipy as sp
from scipy.sparse import dia_matrix
rng = np.random.default_rng()
from itertools import accumulate


def nondiag(A):
    n = A.shape[0]
    mask = ~np.eye(n).astype(bool)
    return mask


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def cov2cor(A):
    var = np.sqrt(A.diagonal())
    var = 1/var.reshape(1, -1)
    return A*var*var.T


def _chol_transform_value(L, i, j):
    """
    Compute i, j element of lower diagonal matrix in cholesky decomposition,
    given L - matrix of form scipy.sparse.dia_matrix.data where all elements
    before i, j are already computed.
    """
    # simple if main diagonal
    if j == (L.shape[1]-1):
        temp = L[i, :j]
        return np.sqrt(L[i, j] - temp.dot(temp.T))
    # get 2 arrays with all Lik Ljk alligned, multiply and get final Lij
    else:
        a = L[i, slice(None, j)]
        b = L[i - ((L.shape[1]-1) - j), slice(-j-1, -1)]
        return (L[i, j] - a.dot(b.T))/L[i - ((L.shape[1]-1) - j), L.shape[1]-1]


def diag_cholesky(A):
    """
    Compute a cholesky decomposition (returns lower triangular matrix)
    of a band matrix. Possibly can be computed for any matrix, but would be
    inefficient. Is much more efficient thant scipy.linalg.cholesky for 
    adequate band matrices (where bandwidth is significantly smaller than
    dimension)
    """
    
    # convert to scipy dia_matrix and pad with zeros and sqrt of main diagonal
    # (allows correct use of the algorithm from the first "real" row)
    diags = dia_matrix(A).data
    udiag = (diags.shape[0] - 1)/2
    assert int(udiag) == udiag, "Number of diagonals is not odd"
    udiag = int(udiag)
    diags = diags[udiag:].T[:, ::-1]
    diags = np.r_[np.zeros([diags.shape[1], diags.shape[1]]), diags]
    
    # get the first sqrt. All later operations can be done with _col_transform_value
    diags[:diags.shape[1], -1] = np.sqrt(diags[diags.shape[1], -1])
    
    # transform all values (in the correct order)
    for i in range(diags.shape[1], diags.shape[0]):
        for j in range(diags.shape[1]):
            diags[i, j] = _chol_transform_value(diags, i, j)
    
    # transform result back to a dia_matrix
    diags = diags[diags.shape[1]:, ::-1].T
    diags = dia_matrix((diags, list(range(len(diags)))), shape=[diags.shape[1], diags.shape[1]])
    return diags


def generate_pd_matrix(size, varmag, covscale, rng=rng):
    varmag = np.sqrt(varmag)
    var = rng.normal(size=size, loc=varmag)
    covvar = varmag*covscale
    covvar = (covvar**2/size)**(1/4)
    cov = rng.normal(size=[size, size], scale=covvar)
    cov = cov.dot(cov.T)
    np.fill_diagonal(cov, var)
    return cov



