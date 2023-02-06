import numpy as np
import torch

def to_list(arr):
    if isinstance(arr, list):
        return arr
    elif isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().tolist()
    elif arr == None:
        return 0
    else:
        raise ValueError('Unknown type: {}'.format(type(arr)))

def to_tensor(arr):
    if isinstance(arr, list):
        return torch.tensor(arr)
    elif isinstance(arr, np.ndarray):
        return torch.tensor(arr)
    elif isinstance(arr, torch.Tensor):
        return arr
    else:
        raise ValueError('Unknown type: {}'.format(type(arr)))

def to_tensor_device(arr):
    tensor_arr = to_tensor(arr)
    if torch.cuda.is_available():
        tensor_arr = tensor_arr.cuda()
    return tensor_arr

def to_numpy(arr):
    if isinstance(arr, list):
        return np.array(arr)
    elif isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    else:
        raise ValueError('Unknown type: {}'.format(type(arr)))
