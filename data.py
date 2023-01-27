import datasets
import numpy as np
import torch
from icecream import ic

def get_dataset():
    dataset_dict = np.load("/cluster/project/sachan/zhiheng/CI_Train/dataset/simulated/causal_simulated_1k_linear.npy", allow_pickle = True).tolist()
    from datasets import Dataset
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

def get_simulatedDataset_IRM(n = 100000):
    from sample_datapoint import generate_datapoint_IRM
    dataset_dict = {"x_1": [], "x_2": [], "y": [], "env": [], "input": []}
    for i in range(100000):
        x_1, x_2, y, env = generate_datapoint_IRM(env = np.random.random())
        dataset_dict["x_1"].append(x_1)
        dataset_dict["x_2"].append(x_2)
        dataset_dict["y"].append(y)
        dataset_dict["env"].append(env)
        # concat x_1, x_2, np.array(torch.randn(6))
        dataset_dict["input"].append(np.concatenate([x_1, x_2, np.array(torch.randn(6))]))
    from datasets import Dataset
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

if __name__ == "__main__":
    """
    dataset = get_dataset()
    ic(dataset)
    #ic(dataset['X_indY'][0:2])
    ic(dataset['Z'][0:2])
    from CIT_warper import run_CIT
    ic(run_CIT(dataset['X_indY'][0:512], dataset['X_YnZ'][0:512], dataset['Z'][0:512]))
    ic(run_CIT(dataset['X_indY'][0:512], dataset['X_YnZ'][0:512], [[0.0]]*512))
    ic(run_CIT(dataset['X_indZ'][0:512], dataset['X_YnZ'][0:512], dataset['Y'][0:512]))
    ic(run_CIT(dataset['X_indZ'][0:512], dataset['X_YnZ'][0:512], [[0.0]]*512))
    """
    datset = get_simulatedDataset_IRM()
    ic(datset)
    ic(datset[0])