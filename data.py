import datasets
import numpy as np
import torch
from icecream import ic
import os

# fix the random seed
np.random.seed(42)
torch.manual_seed(42)

def get_dataset():
    dataset_dict = np.load("/cluster/project/sachan/zhiheng/CI_Train/dataset/simulated/causal_simulated_1k_linear.npy", allow_pickle = True).tolist()
    from datasets import Dataset
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

def get_simulatedDataset_IRM(n = 100000, group_number = 1, new_data = False, remove_columns = True):
    from sample_datapoint import generate_datapoint_IRM
    dataset_path = f"./dataset/simulatedIRMs_newData_v2/{n}_{group_number}"
    # if (os.path.exists(dataset_path)):
    if not new_data and os.path.exists(dataset_path):
        # load huggingface dataset
        dataset = datasets.load_from_disk(dataset_path)
        print("Dataset loaded from disk")
    else:
        print("Dataset not found, generating new dataset")
        dataset_dict = {"y": [], "env": [], "input": [], "x_1": [], "x_2": []}
        for i in range(n):
            curr_env = 0
            if group_number == 1:
                # generate a Gaussian distribution with mean 0 and std 1
                curr_env = np.random.normal()
            elif group_number == 2:
                # generate a Gaussian distribution with mean 1 and std 1
                curr_env = np.random.normal() + 1
            else:
                raise "Invalid group number"
            x_1, x_2, y, env = generate_datapoint_IRM(env = curr_env, d = 1)
            dataset_dict["x_1"].append(x_1)
            dataset_dict["x_2"].append(x_2)
            dataset_dict["y"].append(y)
            dataset_dict["env"].append(env)
            # concat x_1, x_2, np.array(torch.randn(6))
            dataset_dict["input"].append(np.concatenate([x_1, x_2]))
        from datasets import Dataset
        dataset = Dataset.from_dict(dataset_dict)
        dataset.save_to_disk(dataset_path)
    # delete x_1, x_2 columns from dataset
    if remove_columns:
        dataset = dataset.remove_columns(["x_1", "x_2"])
    return dataset

def get_simulatedDataset_ST(n = 100000, group_number = 1, new_data = False, remove_columns = True, causal = True):
    pass

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
    datset = get_simulatedDataset_IRM(n=128000, group_number=1)
    ic(datset)
    ic(datset[0])
    datset = get_simulatedDataset_IRM(n=128000, group_number=2)
    ic(datset)
    ic(datset[0])