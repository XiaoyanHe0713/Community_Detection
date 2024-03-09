import torch
import numpy as np
from tensorly.decomposition import tucker
import tensorly as tl
from scipy.stats import ortho_group
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD

def n_mode_product(x, u, n):
    """n-mode product of a tensor and a matrix at the specified mode

    Mathematically: :math:`\\text{tensor} \\times_{\\text{n}} \\text{matrix}`
    Parameters
    ----------
    x : torch tensor
        tensor of shape ``(i_1, ..., i_k, ..., i_N)``
    u : torch tensor
        matrix of shape ``(J, i_k)``
    n : int
        mode along which to take the product

    Returns
    -------
    torch tensor
        n-mode product of tensor by matrix
        * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)`
    """
    n = int(n)
    # We need one letter per dimension
    # (maybe you could find a workaround for this limitation)
    if n > 26:
        raise ValueError('n is too large.')
    ind = ''.join(chr(ord('a') + i) for i in range(n))
    exp = f'{ind}K...,JK->{ind}J...'
    result = tf.einsum(exp, x, u)
    return torch.tensor(result.numpy())

def unfold(tensor, mode):
    
    return torch.reshape(torch.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def fold(unfolded_tensor, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return torch.moveaxis(torch.reshape(unfolded_tensor, full_shape), 0, mode)

def diag(A):
    """ Diagonal of a tensor
    Parameters
    ----------
    A: is a torch tensor, diagonal of A is the tensor whose (i1, . . . , im) element is equal to the corresponding element of A if some 
    indices of i_1, . . . , i_m are identical, and is equal to 0 otherwise.
    Our deﬁnition of diag is different from the conventional one. We use this definition for its convenience in dealing with self-edges.
    Returns
    -------
    diag_A: torch tensor of the same shape as A containing the diagonal of A
    """
    # Get the shape of the tensor
    shape = A.shape
    
    # Initialize the result tensor with zeros of the same shape as A
    diag_A = A
    
    # Iterate over all indices combinations using np.ndindex
    for index in np.ndindex(shape):
        # Check if any two indices are equal (indicating a generalized diagonal)
        if len(index) == len(set(index)):
            # Copy the element from A to diag_A at the current index if it's part of the generalized diagonal
            diag_A[index] = 0
    
    return diag_A
        
def multi_mode_dot(tensor, matrices, modes):
    """Chain several mode_dot (n-mode product) in one go
    
    Parameters
    ----------
    tensor : torch tensor
        tensor of shape ``(i_1, ..., i_k, ..., i_N)``
    matrices : list of torch tensors
        list of 1D or 2D arrays
        matrix or vectors to which to n-mode multiply the tensor
    modes : int list, optional
        list of the modes on which to perform the n-mode product
    
    Returns
    -------
    torch tensor
    `mode`-mode product of `tensor` by `matrix_or_vector`
        * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
        * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector
    """
    res = tensor
    for mode, matrix in zip(modes, matrices):
        res = n_mode_product(res, matrix, mode)
    return res

def mode_dot(tensor, matrix_or_vector, mode, transpose=False):
    """n-mode product of a tensor and a matrix or vector at the specified mode

    Mathematically: :math:`\\text{tensor} \\times_{\\text{mode}} \\text{matrix or vector}`

    Parameters
    ----------
    tensor : ndarray
        tensor of shape ``(i_1, ..., i_k, ..., i_N)``
    matrix_or_vector : ndarray
        1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
        matrix or vectors to which to n-mode multiply the tensor
    mode : int
    transpose : bool, default is False
        If True, the matrix is transposed.

    Returns
    -------
    torch tensor
        `mode`-mode product of `tensor` by `matrix_or_vector`
        * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
        * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector
    """
    # the mode along which to fold might decrease if we take product with a vector
    fold_mode = mode
    new_shape = list(tensor.shape)

    if matrix_or_vector.dim() == 2:  # Tensor times matrix
        # Test for the validity of the operation
        dim = 0 if transpose else 1
        if matrix_or_vector.shape[dim] != tensor.shape[mode]:
            raise ValueError(
                f"shapes {tensor.shape} and {matrix_or_vector.shape} not aligned in mode-{mode} multiplication: "
                f"{tensor.shape[mode]} (mode {mode}) != {matrix_or_vector.shape[dim]} (dim 1 of matrix)"
            )

        if transpose:
            matrix_or_vector = torch.conj(torch.transpose(matrix_or_vector))

        new_shape[mode] = matrix_or_vector.shape[0]
        vec = False

    elif matrix_or_vector.dim() == 1:  # Tensor times vector
        if matrix_or_vector.shape[0] != tensor.shape[mode]:
            raise ValueError(
                f"shapes {tensor.shape} and {matrix_or_vector.shape} not aligned for mode-{mode} multiplication: "
                f"{tensor.shape[mode]} (mode {mode}) != {matrix_or_vector.shape[0]} (vector size)"
            )
        if len(new_shape) > 1:
            new_shape.pop(mode)
        else:
            new_shape = ()
        vec = True

    else:
        raise ValueError(
            "Can only take n_mode_product with a vector or a matrix."
            f"Provided array of dimension {torch.ndim(matrix_or_vector)} not in [1, 2]."
        )

    res = torch.dot(matrix_or_vector, unfold(tensor, mode))

    if vec:  # We contracted with a vector, leading to a vector
        return torch.reshape(res, shape=new_shape)
    else:  # tensor times vec: refold the unfolding
        return fold(res, fold_mode, new_shape)
    
def tucker_to_tensor(tucker_tensor, skip_factor=None, transpose_factors=False):
    """Converts the Tucker tensor into a full tensor

    Parameters
    ----------
    tucker_tensor : tl.TuckerTensor or (core, factors)
        core tensor and list of factor matrices
    skip_factor : None or int, optional, default is None
        if not None, index of a matrix to skip
        Note that in any case, `modes`, if provided, should have a lengh of ``tensor.ndim``
    transpose_factors : bool, optional, default is False
        if True, the matrices or vectors in in the list are transposed

    Returns
    -------
    torch tensor
       full tensor of shape ``(factors[0].shape[0], ..., factors[-1].shape[0])``
    """
    core, factors = tucker_tensor
    return multi_mode_dot(core, factors, skip=skip_factor, transpose=transpose_factors)

def tucker_decomp(tensor, rank, n_iter_max=100):
    """Tucker decomposition of a tensor

    Parameters
    ----------
    tensor : torch tensor
    rank : int list
        rank of the decomposition

    Returns
    -------
    core : torch tensor
        core tensor of the Tucker decomposition
    factors : list of torch tensors
        list of factor matrices
    """
    tl.set_backend('pytorch')
    return tucker(tensor, rank, n_iter_max=n_iter_max)

def HOOI(tensor, ranks, n_iter_max=100, tol=1e-8):
    """High-Order Orthogonal Iteration (HOOI) algorithm for Tucker decomposition
    Perform Higher Order Orthogonal Iteration (HOOI) on a 3D tensor.

    Parameters:
    - tensor: Input tensor of size I x J x K.
    - ranks: Tuple of target ranks (r1, r2, r3).
    - n_iter_max: Maximum number of iterations.
    - eps: Convergence criterion.

    Returns:
    - core_tensor: Core tensor of the Tucker decomposition.
    - [L, R, V] factor matrices.
    L ∈ R^I×r1, R ∈ R^J×r2, V ∈ R^K×r3.
    """
    # Initialize R and V factor matrices with orthonormal columns
    I, J, K = tensor.shape
    R = ortho_group.rvs(J)[:,:ranks[1]]
    R = torch.tensor(R, dtype=torch.float32)
    V = ortho_group.rvs(K)[:,:ranks[2]]
    V = torch.tensor(V, dtype=torch.float32)

    for _ in range(n_iter_max):
        # C = A ×2 R^T ×3 V^T
        C = multi_mode_dot(tensor, [R.T, V.T], modes=[1, 2])
        C_1 = unfold(tensor, 0)

        # L = SVD(r1, C_1), where U = SVD(k, C) means compute the k’th order truncated 
        # SVD of C and then set U = [u1, u2, . . . , uk] to the matrix whose columns are 
        # the k largest left singular vectors ui of C
        U, _, _ = torch.svd(C_1)
        L = U[:, :ranks[0]]

        # D = A ×1 L^T ×3 V^T
        D = multi_mode_dot(tensor, [L.T, V.T], modes=[0, 2])
        D_2 = unfold(tensor, 1)

        #R = SVD(r2, D_2)
        U, _, _ = torch.svd(D_2)
        R = U[:, :ranks[1]]

        # E = A ×1 L^T ×2 R^T
        E = multi_mode_dot(tensor, [L.T, R.T], modes=[0, 1])
        E_3 = unfold(tensor, 2)

        # V = SVD(r3, E_3)
        U, _, _ = torch.svd(E_3)
        V = U[:, :ranks[2]]

        # Compute the approximation error
        approx_tensor = n_mode_product(E, V.T, 2)
        if torch.norm(tensor - approx_tensor) < tol:
            break

    # Compute the core tensor
    core_tensor = n_mode_product(E, V.T, 2)

    return core_tensor, [L, R, V]



def scale_invariant(matrix):
    """Scale-invariant tensor, proposed the following row-wise normalization on given matrix
    Parameters
    ----------
    matrix : torch tensor
    Returns
    -------
    torch tensor
        scale-invariant matrix
    """
    # For each row, divide by the first coordinate of the row
    return matrix / matrix[:, 0].unsqueeze(1)