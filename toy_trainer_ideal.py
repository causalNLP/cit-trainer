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
    ic(x1[0:10], estimate_env[0:10], env[0:10])
    return loss, R_loss, CIT_loss


def test_idel_CIT_loss():
    from data import get_simulatedDataset_IRM
    dataset = get_simulatedDataset_IRM(n = 25600, group_number = 1)
    dataset = dataset.train_test_split(test_size=0.2)
    print(dataset)
    test_data = dataset['test']
    length = len(test_data)
    batch_size = 256
    CI_list1 = []
    CI_list2 = []
    CI_estimated_list1 = []
    CI_estimated_list2 = []
    CI_hypo_list1 = []
    CI_hypo_list2 = []
    CI_hypo_list3 = []
    from tqdm import trange
    for i in trange(0, length, batch_size):
        batch_data = test_data[i:i+batch_size]
        x_1 = np.array(batch_data['input'])[:,0].reshape(-1,1)
        x_2 = np.array(batch_data['input'])[:,1].reshape(-1,1)
        y = np.array(batch_data['y'])
        env = np.array(batch_data['env'])
        estimated_x_1 = 2.8580*x_1 + 0.0218*x_2
        estimated_x_2 = 1*x_1 - 2.6020*x_2
        CI_list1.append(run_CIT(x_1, x_2, y)[2].detach().numpy())
        CI_list2.append(run_CIT(env, x_2, y)[2].detach().numpy())
        CI_estimated_list1.append(run_CIT(estimated_x_1, estimated_x_2, y)[2].detach().numpy())
        CI_estimated_list2.append(run_CIT(env, estimated_x_2, y)[2].detach().numpy())
        CI_hypo_list1.append(run_CIT(env, 0.1654*x_1, y)[2].detach().numpy())
        CI_hypo_list2.append(run_CIT(env, -2.6020*x_2, y)[2].detach().numpy())
        CI_hypo_list3.append(run_CIT(env, x_1, y)[2].detach().numpy())
    ic(np.mean(CI_list1), np.mean(CI_estimated_list1))
    ic(np.mean(CI_list2), np.mean(CI_estimated_list2))
    ic(np.mean(CI_hypo_list1), np.mean(CI_hypo_list2), np.mean(CI_hypo_list3))
if __name__ == '__main__':

    #from data import get_simulatedDataset_IRM
    #dataset = get_simulatedDataset_IRM(n = 25600, group_number = 1)
    #batch_data = dataset[:256]
    #get_optm_loss(batch_data)
    #from toy_trainer import estimate_upperBound_performance
    #estimate_upperBound_performance()
    test_idel_CIT_loss()

