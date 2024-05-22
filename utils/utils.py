import torch


def vector_matrix_cosine_similarity(matrix, vector):
    """
    Find cosine similarity of a vector with each column of a matrix
    :param matrix: (D, ?)
    :param vector: (D,)
    :return: ()
    """
    vector_norm = vector / torch.norm(vector)  # (D,)
    matrix_norm = matrix / torch.norm(matrix, dim=0)  # (D, ?) - Normalizes each column
    return torch.matmul(vector_norm, matrix_norm).squeeze()  # (?)


def vector_vector_cosine_similarity(a, b):
    """
    Find cosine similarity of two vectors
    :param a: (D,)
    :param b: (D,)
    :return: ()
    """
    a_norm = a / torch.norm(a)
    b_norm = b / torch.norm(b)
    return torch.dot(a_norm, b_norm).item()


def get_current_gpu_use(return_value=False):
    bytes_in_use = torch.cuda.max_memory_allocated(device=None)
    gb_in_use = bytes_in_use / 1024 ** 3
    print(f'{round(gb_in_use, 2)} GB')
    if return_value:
        return gb_in_use
