from CIT_warper import run_CIT
from icecream import ic

def test_all_pairs(dataset_dict):
    keys = dataset_dict.keys()
    for i in keys:
        for j in keys:
            for k in keys:
                if (i!=j and j!=k and i!=k):
                    print(f"p-value of {i} \ind {j} | {k} is {run_CIT(dataset_dict[i][:1000], dataset_dict[j][:1000], dataset_dict[k][:1000])}")

if __name__ == '__main__':
    import numpy as np
    dataset_dict = np.load("/cluster/project/sachan/zhiheng/CI_Train/dataset/simulated/anticausal_simulated_100k.npy", allow_pickle = True).tolist()
    #dataset = convert_dict_to_array(dataset_dict)
    ic(dataset_dict['Y'][0:10])
    ic(dataset_dict['X_indZ'][0:10])
    ic(dataset_dict['X_YnZ'][0:10])
    dataset_dict['None'] = []
    for i in range(len(dataset_dict['Y'])):
        dataset_dict['None'].append([0])
    test_all_pairs(dataset_dict)
