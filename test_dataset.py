from CIT_warpper import run_CIT

from icecream import ic
import numpy as np

"""
# for old stress test dataset

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

"""

# used for IRM dataset
def test_data(dataset, X, Y, Z):
    batch_size = 256
    from tqdm import trange
    loss_list = []
    p_value_list = []
    for i in trange(0, len(dataset), batch_size):
        if (i+batch_size > len(dataset)):
            break
        batch = dataset[i:i+batch_size]
        X_batch = batch[X]
        Y_batch = batch[Y]
        Z_batch = batch[Z]
        p_value, stat, loss = run_CIT(X_batch, Y_batch, Z_batch)
        #print(f"p-value of {X} \ind {Y} | {Z} is {p_value}")
        p_value_list.append(p_value)
        loss_list.append(loss.item())
    print(f"average loss is {np.mean(loss_list)}")
    print(f"average p-value is {np.mean(p_value_list)}")

def test_bias_estimator(dataset):
    # test x_1+1 \ind y | input

    batch_size = 256
    from tqdm import trange
    loss_list = []
    p_value_list = []
    for i in trange(0, len(dataset), batch_size):
        if (i+batch_size > len(dataset)):
            break
        batch = dataset[i:i+batch_size]
        X_batch = np.array(batch['x_1'])*0.8+0.2
        Y_batch = np.array(batch['env'])
        Z_batch = np.array(batch['input'])
        p_value, stat, loss = run_CIT(X_batch, Y_batch, Z_batch)
        p_value_list.append(p_value)
        loss_list.append(loss.item())
    print(f"average loss is {np.mean(loss_list)}")
    print(f"average p-value is {np.mean(p_value_list)}")

if __name__ == '__main__':
    from data import get_simulatedDataset_IRM
    dataset = get_simulatedDataset_IRM(n = 25600, group_number = 2, new_data = True, remove_columns = False)
    dataset = dataset.train_test_split(test_size=0.2)
    print(dataset['train'])
    #test_data(dataset['test'], 'x_1', 'x_2', 'y')# should be independent
    #test_data(dataset['test'], 'env', 'x_2', 'y')# should be independent
    #test_data(dataset['test'], 'env', 'x_1', 'y')# should be dependent
    test_bias_estimator(dataset['test'])

