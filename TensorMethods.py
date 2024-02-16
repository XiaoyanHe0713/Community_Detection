import torch
import numpy as np

def diag(A):
    "A is a torch tensor, diagonal of A is the tensor whose (i1, . . . , im)"
    "element is equal to the corresponding element of A if some indices of i_1, . . . , i_m are identical, and is equal to 0 otherwise"
    "Our deﬁnition of diag is different from the conventional one. We use this definition for its convenience in dealing with self-edges."
    "Returns a torch tensor with the diagonal of A"
    
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

