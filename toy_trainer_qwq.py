from CIT_warpper import run_CIT
from icecream import ic

import torch
from torch import nn
import numpy as np
def get_optm_loss(batch_data):
    x1 = np.array(batch_data['input'])[:,0].reshape(-1,1)
    x2 = np.array(batch_data['input'])[:,1].reshape(-1,1)
    y = np.array(batch_data['y'])
    env = np.array(batch_data['env'])
    estimate_env = x1
    estimate_y = x1/2 + x2/2 + env/2
    loss = np.mean((estimate_y - y)**2)
    R_loss = np.mean((estimate_env - env)**2)
    CIT_loss = run_CIT(x1, x2, y)[2]+run_CIT(env, x2, y)[2]
    ic(loss, R_loss, CIT_loss)
    return loss, R_loss, CIT_loss

if __name__ == '__main__':

    from data import get_simulatedDataset_IRM
    dataset = get_simulatedDataset_IRM(n = 25600, group_number = 1)
    batch_data = dataset[:256]
    get_optm_loss(batch_data)

