import os
import datasets
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from icecream import ic

from utils import *
from trainer import Trainer

def eval_MI(pred, gt):
    # calculate the mutual information between pred and gt
    # return: a float number
    pred = to_numpy(pred)
    gt = to_numpy(gt)
    n = len(pred)
    pred = pred.reshape(n, -1)
    gt = gt.reshape(n, -1)
    ic(pred.shape, gt.shape)
    pred = to_list(pred)
    gt = to_list(gt)
    from npeet import entropy_estimators as ee
    mi = ee.mi(pred, gt)
    return mi

lambda_CI_list = ["0.0", "0.001", "0.005", "0.01", "0.05", "0.1", "0.3", "1"]

def load_trainer(output_path, dataset):
    # Always load checkpoint4, which is the last checkpoint
    trainer = Trainer(dataset.remove_columns(["x_1", "x_2"]), ['x_1', 'x_2', 'env'],
                      [['input', 'x_1'], ['input', 'x_2'], ['x_1', 'env']],
                      [['x_1', 'x_2', 'y'], ['env', 'x_2', 'y']],
                      lambda_ci=0, lambda_R=1, output_dir=output_path)
    if (torch.cuda.is_available()):
        trainer.cuda()
    ic(output_path + 'epoch_4.pth')
    trainer.load_checkpoint(output_path + 'epoch_4.pth')
    return trainer

if __name__ == '__main__':
    """
    # Unit test begins
    import numpy as np
    n = np.random.rand(100000, 2)
    m = n
    ic(eval_MI(n, m))
    n = np.random.rand(100000, 2)
    m = np.random.rand(100000, 2)
    ic(eval_MI(n, m))
    n = np.random.rand(100000, 2)
    m = n+np.random.rand(100000, 2)
    ic(eval_MI(n, m))
    # Unit test ends
    exit(0)
    """
    dataset_path_1 = f"./dataset/simulatedIRMs_newData/128000_1"
    dataset_path_2 = f"./dataset/simulatedIRMs_newData/128000_2"
    dataset_1 = datasets.load_from_disk(dataset_path_1)
    dataset_2 = datasets.load_from_disk(dataset_path_2)
    ic(dataset_1, dataset_2)

    # eval IRM data

    #ic(dataset_1[0:10])
    trainer = load_trainer('/cluster/project/sachan/zhiheng/CI_Train/method/CIT/result/simulated_IRM_newData/128000_1_0.01_1/', dataset_1)
    ic(trainer.infernece_one_sample(dataset_1[0:10]))
    torch.save(trainer.models.state_dict(), "./1.pth")

    trainer = load_trainer('/cluster/project/sachan/zhiheng/CI_Train/method/CIT/result/simulated_IRM_newData/128000_1_0.0_1/', dataset_1)
    ic(trainer.infernece_one_sample(dataset_1[0:10]))
    trainer.load_checkpoint("./1.pth")
    ic(trainer.infernece_one_sample(dataset_1[0:10]))
    ic(dataset_1[0:10])
    trainer = load_trainer('/cluster/project/sachan/zhiheng/CI_Train/method/CIT/result/simulated_IRM_newData/128000_1_0.01_1/',
                           dataset_1)
    exit(0)
    

    dict_task1 = {"lambda_CI": [], "ID": [], "OOD": []}
    dict_task2 = {"lambda_CI": [], "MI_x1": [], "MI_x2": [], "MI_y": [], "MI_env": [],
                  "OODMI_x1": [], "OODMI_x2": [], "OODMI_y": [], "OODMI_env": []}
    for lambda_ci in lambda_CI_list:
        output_path = './result/simulated_IRM_newData' + '/' + str(128000) + '_' \
                  + str(1) + '_' + str(lambda_ci) + '_' + str(1) + '/'
        # check if there is result.csv in the output_path
        if not os.path.exists(output_path + 'result.csv'):
            print('No result.csv in ' + output_path)
            continue
        else:
            print('Found result.csv in ' + output_path)
        # load the trainer
        trainer = load_trainer(output_path, dataset_1)
        ic(trainer)

        ID_result = trainer.inference(dataset = dataset_1)
        ic(np.mean(ID_result['loss']))
        ic(trainer.evaluate(16, 128))
        OOD_result = trainer.inference(dataset = dataset_2)

        # eval task 1: OOD performance
        # | lambda_CI | ID performance | OOD performance |
        dict_task1["lambda_CI"].append(lambda_ci)
        dict_task1["ID"].append(np.mean(ID_result['loss']))
        # in this case, we simply use the loss as the performance metric
        dict_task1["OOD"].append(np.mean(OOD_result['loss']))
        ic(dict_task1)
        # eval task 2: MI
        # | lambda_CI | MI of x_1 | MI of x_2 | MI of y | MI of env | OODMI of x_1 | OODMI of x_2 | OODMI of y | OODMI of env |
        dict_task2["lambda_CI"].append(lambda_ci)
        train_size = int(len(dataset_1['input'])*0.8)

        x_1 = dataset_1['x_1'][train_size:]
        x_2 = dataset_1['x_2'][train_size:]
        env = dataset_1['env'][train_size:]
        y = dataset_1['y'][train_size:]
        dict_task2["MI_x1"].append(eval_MI(ID_result['x_1'], x_1))
        dict_task2["MI_x2"].append(eval_MI(ID_result['x_2'], x_2))
        dict_task2["MI_y"].append(eval_MI(ID_result['y'], y))
        dict_task2["MI_env"].append(eval_MI(ID_result['env'], env))

        x_1 = dataset_2['x_1'][train_size:]
        x_2 = dataset_2['x_2'][train_size:]
        env = dataset_2['env'][train_size:]
        y = dataset_2['y'][train_size:]
        dict_task2["OODMI_x1"].append(eval_MI(OOD_result['x_1'], x_1))
        dict_task2["OODMI_x2"].append(eval_MI(OOD_result['x_2'], x_2))
        dict_task2["OODMI_y"].append(eval_MI(OOD_result['y'], y))
        dict_task2["OODMI_env"].append(eval_MI(OOD_result['env'], env))

        ic(dict_task2)

    # save the result
    df_task1 = pd.DataFrame(dict_task1)
    df_task1.to_csv('./result/simulated_IRM/eval_task1.csv', index=False)
    df_task2 = pd.DataFrame(dict_task2)
    df_task2.to_csv('./result/simulated_IRM/eval_task2.csv', index=False)
