import os
import datasets
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_regression
from icecream import ic

from utils import *
#from trainer import Trainer
from data import get_simulatedDataset_IRM

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
"""
def old_IRM_eval_code():
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
    dataset_path_1 = f"./dataset/simulatedIRMs_newData/128000_1"
    dataset_path_2 = f"./dataset/simulatedIRMs_newData/128000_2"
    dataset_1 = datasets.load_from_disk(dataset_path_1)
    dataset_2 = datasets.load_from_disk(dataset_path_2)
    ic(dataset_1, dataset_2)

    # eval IRM data

    # ic(dataset_1[0:10])
    trainer = load_trainer(
        '/cluster/project/sachan/zhiheng/CI_Train/method/CIT/result/simulated_IRM_newData/128000_1_0.01_1/', dataset_1)
    ic(trainer.infernece_one_sample(dataset_1[0:10]))
    torch.save(trainer.models.state_dict(), "./1.pth")

    trainer = load_trainer(
        '/cluster/project/sachan/zhiheng/CI_Train/method/CIT/result/simulated_IRM_newData/128000_1_0.0_1/', dataset_1)
    ic(trainer.infernece_one_sample(dataset_1[0:10]))
    trainer.load_checkpoint("./1.pth")
    ic(trainer.infernece_one_sample(dataset_1[0:10]))
    ic(dataset_1[0:10])
    trainer = load_trainer(
        '/cluster/project/sachan/zhiheng/CI_Train/method/CIT/result/simulated_IRM_newData/128000_1_0.01_1/',
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

        ID_result = trainer.inference(dataset=dataset_1)
        ic(np.mean(ID_result['loss']))
        ic(trainer.evaluate(16, 128))
        OOD_result = trainer.inference(dataset=dataset_2)

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
        train_size = int(len(dataset_1['input']) * 0.8)

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

def IRM_toy_model(model_path):
    # In-domain data
    from data import get_simulatedDataset_IRM
    dataset = get_simulatedDataset_IRM(n=256000, group_number=1)
    dataset = dataset.train_test_split(test_size=0.2)

    # 1. create the model class
    from toy_trainer_v4 import Trainer, estimate_upperBound_performance
    trainer = Trainer(dataset, lambda_ci = 1, lambda_R = 1)
    # 2. eval the model upper bound
    #estimate_upperBound_performance(256000, 1)
    # 3. load the model parameters
    path = model_path
    trainer.load_state_dict(torch.load(path))
    #ic(trainer.evaluate())
    trainer.show_params()
    # 4. use out of domain data to eval the env
    dataset_ood = get_simulatedDataset_IRM(n=256000, group_number=2)
    dataset_ood = dataset_ood.train_test_split(test_size=0.2)

    x1_list, x2_list, y_list, env_list = trainer.inference(dataset_ood['train'])
    input = x1_list.detach().numpy().reshape(-1, 1)
    output = np.array(dataset_ood['train']['env']).reshape(-1, 1)
    from sklearn.linear_model import LinearRegression
    model_env = LinearRegression().fit(input, output)
    ic(model_env.coef_, model_env.intercept_)
    env_pred = model_env.predict(input)
    ic(np.mean((env_pred - output) ** 2))
    #exit(0)

    # 5. load the out-of-domain data
    #dataset_ood = get_simulatedDataset_IRM(n=256000, group_number=2)
    #dataset_ood = dataset_ood.train_test_split(test_size=0.2)
    #ic(dataset, dataset_ood)
    #ic(dataset['train']['env'][:10])
    #ic(dataset_ood['train']['env'][:10])
    #ic(np.mean(dataset['train']['env']), np.mean(dataset_ood['train']['env']))
    #ic(np.var(dataset['train']['env']), np.var(dataset_ood['train']['env']))
    #exit(0)
    # 6. eval the model using linear regression
    from sklearn.linear_model import LinearRegression

    # 7. eval the model using linear regression in few-shot setting
    # the baseline is simply use all the features as the input
    # the improved version is to use feature x_1 to estimate env, and use all the features to estimate y

    few_shot_data = dataset_ood['train'][:16]

    test_x1 = np.array(dataset_ood['test']['input'])[:, 0].reshape(-1, 1)
    test_x2 = np.array(dataset_ood['test']['input'])[:, 1].reshape(-1, 1)
    test_env = np.array(dataset_ood['test']['env']).reshape(-1, 1)

    # 1. baseline
    train_x1 = np.array(few_shot_data['input'])[:, 0].reshape(-1, 1)
    train_x2 = np.array(few_shot_data['input'])[:, 1].reshape(-1, 1)
    input = np.concatenate((train_x1, train_x2), axis=1)
    output = few_shot_data['y']
    model = LinearRegression().fit(input, output)
    pred_y = model.predict(input)
    print('baseline train: ', np.mean((pred_y - output)**2))
    # eval
    input = np.concatenate((test_x1, test_x2), axis=1)
    # print(input.shape)
    output = dataset_ood['test']['y']
    pred_y = model.predict(input)
    # ic(pred_y[:10], output[:10])
    print('baseline test: ', np.mean((pred_y - output)**2))
    ic(model.coef_, model.intercept_)

    # 2. improved version
    # first estimate env using in-domain data
    print('estimate env using in-domain data')
    x1_list, x2_list, y_list, env_list = trainer.inference(dataset['train'])
    x1_list, x2_list, y_list, env_list = x1_list.detach().numpy(), x2_list.detach().numpy(), y_list.detach().numpy(), env_list.detach().numpy()

    input = np.array(x1_list).reshape(-1, 1)
    output = dataset['train']['env']
    model_env = LinearRegression().fit(input, output)
    ic(model_env.coef_, model_env.intercept_)
    env_pred = model_env.predict(input)
    ic(np.mean((env_pred - output) ** 2))
    # then estimate y using in-domain data
    print('estimate y using in-domain data qwqqqqqqqqqqq')
    env_gt = output
    # mix the env_pred and env_input with prob 0.5
    env_input = torch.where(torch.rand(env_pred.shape) > 0.5, torch.tensor(env_pred), torch.tensor(env_gt)).reshape(-1, 1)
    env_input = env_input.detach().numpy()
    input = np.concatenate((x1_list, x2_list, env_gt), axis=1)
    output = dataset['train']['y']
    model_y = LinearRegression().fit(input, output)
    ic(model_y.coef_, model_y.intercept_)
    y_pred = model_y.predict(input)
    ic(np.mean((y_pred - output) ** 2))
    # then estimate env using out-of-domain data
    print('estimate env using out-of-domain data')
    x1_list, x2_list, y_list, env_list = trainer.compute_batch(few_shot_data, train = False)
    x1_list, x2_list, y_list, env_list = x1_list.detach().numpy(), x2_list.detach().numpy(), y_list.detach().numpy(), env_list.detach().numpy()
    input = np.array(x1_list).reshape(-1, 1)
    output = few_shot_data['env'].detach().numpy()
    model_env_test = LinearRegression().fit(input, output)
    ic(model_env_test.coef_, model_env_test.intercept_)
    env_pred = model_env_test.predict(input)
    ic(np.mean((env_pred - output) ** 2))
    # then we have an in-domain y model and an out-of-domain env model
    # use both of them to estimate ood y
    print('estimate y using out-of-domain data')
    x1_list, x2_list, y_list, env_list = trainer.inference(dataset_ood['test'])
    x1_list, x2_list, y_list, env_list = x1_list.detach().numpy(), x2_list.detach().numpy(), y_list.detach().numpy(), env_list.detach().numpy()
    input = np.array(x1_list).reshape(-1, 1)
    output = dataset_ood['test']['y']
    env_pred = model_env_test.predict(input)
    ic(np.mean((env_pred - dataset_ood['test']['env']) ** 2))
    input = np.concatenate((x1_list, x2_list, env_pred), axis=1)
    y_pred = model_y.predict(input)
    ic(np.mean((y_pred - output) ** 2))

    exit(0)
    x1_list, x2_list, y_list, env_list = trainer.compute_batch(few_shot_data, train = False)
    x1_list, x2_list, y_list, env_list = x1_list.detach().numpy(), x2_list.detach().numpy(), y_list.detach().numpy(), env_list.detach().numpy()

    # first estimate env
    input = np.array(x1_list).reshape(-1, 1)
    output = few_shot_data['env']
    model_env_test = LinearRegression().fit(input, output)
    ic(model_env_test.coef_, model_env_test.intercept_)
    ic(model_env_test.score(input, output))
    # then estimate y
    # predict env
    env_list_est = model_env_test.predict(input)
    input = np.concatenate((x1_list, x2_list, env_list_est), axis=1)
    #ic(np.mean(x1_list), np.mean(x2_list))
    #ic(np.var(x1_list), np.var(x2_list))
    output = few_shot_data['y'].detach().numpy()
    model_y = LinearRegression().fit(input, output)
    print("improved train: ", np.mean((model_y.predict(input) - output)**2))
    # eval
    # use model to inference all test data to get x1, x2, env
    x1_list, x2_list, y_list, env_list = trainer.inference(dataset_ood['test'])
    x1_list, x2_list, y_list, env_list = x1_list.detach().numpy(), x2_list.detach().numpy(), y_list.detach().numpy(), env_list.detach().numpy()

    input = np.array(x1_list).reshape(-1, 1)
    env_gt = dataset_ood['test']['env']
    env_test_est = model_env_test.predict(input)
    print("improved env test:", np.mean((env_gt - env_test_est)**2))

    #input = np.concatenate((x1_list, x2_list), axis=1)
    input = np.concatenate((x1_list, x2_list, env_test_est), axis=1)
    output = dataset_ood['test']['y']
    #model_y = LinearRegression().fit(input, output)
    pred_y = model_y.predict(input)
    #print(pred_y[:10], output[:10])
    ic(model_y.coef_, model_y.intercept_)
    print("improved test: ", np.mean((pred_y - output)**2))
"""
def test_IRM_toy_model(model_path):
    # load in-domain data and pretrained model
    from data import get_simulatedDataset_IRM
    dataset = get_simulatedDataset_IRM(n=256000, group_number=1)
    dataset = dataset.train_test_split(test_size=0.2)
    dataset_ood = get_simulatedDataset_IRM(n=256000, group_number=2)
    dataset_ood = dataset_ood.train_test_split(test_size=0.2)
    few_shot_data = dataset_ood['train'][0:16]

    # load the model
    from toy_trainer_v4 import Trainer, estimate_upperBound_performance
    trainer = Trainer(dataset, lambda_ci = 1, lambda_R = 1)
    path = model_path
    trainer.load_state_dict(torch.load(path))
    trainer.show_params()
    from sklearn.linear_model import LinearRegression

    print('estimate env using in-domain data')
    x1_list, x2_list, y_list, env_list = trainer.inference(dataset['train'])
    x1_list, x2_list, y_list, env_list = x1_list.detach().numpy(), x2_list.detach().numpy(), y_list.detach().numpy(), env_list.detach().numpy()

    input = np.array(x1_list).reshape(-1, 1)
    output = dataset['train']['env']
    model_env = LinearRegression().fit(input, output)
    ic(model_env.coef_, model_env.intercept_)
    env_pred = model_env.predict(input)
    ic(np.mean((env_pred - output) ** 2))
    # then estimate y using in-domain data
    #print('estimate y using in-domain data')
    #input = np.concatenate((x1_list, x2_list, env_pred), axis=1)
    #output = dataset['train']['y']
    #from sklearn.linear_model import Ridge
    #model_y = Ridge(alpha=1, fit_intercept=True).fit(input, output)

    print('estimate y using in-domain data qwqqqqqqqqqqq')
    env_gt = output
    # mix the env_pred and env_input with prob 0.5
    env_input = torch.where(torch.rand(env_pred.shape) > 0.01, torch.tensor(env_pred), torch.tensor(env_gt)).reshape(-1, 1)
    env_input = env_input.detach().numpy()
    input = np.concatenate((x1_list, x2_list, env_input), axis=1)
    output = dataset['train']['y']
    model_y = LinearRegression().fit(input, output)

    ic(model_y.coef_, model_y.intercept_)
    y_pred = model_y.predict(input)
    ic(np.mean((y_pred - output) ** 2))
    in_domain_MSE = np.mean((y_pred - output) ** 2)

    # then estimate env using out-of-domain data
    print('estimate env using out-of-domain data')
    x1_list, x2_list, y_list, env_list = trainer.compute_batch(few_shot_data, train=False)
    x1_list, x2_list, y_list, env_list = x1_list.detach().numpy(), x2_list.detach().numpy(), y_list.detach().numpy(), env_list.detach().numpy()
    input = np.array(x1_list).reshape(-1, 1)
    output = few_shot_data['env'].detach().numpy()
    model_env_test = model_env
    # manually calculate the intercept
    model_env_test.intercept_ = np.mean(output-input*model_env_test.coef_)
    #model_env_test = model_env_test.fit(input, output)
    ic(model_env_test.coef_, model_env_test.intercept_)
    env_pred = model_env_test.predict(input)
    ic(np.mean((env_pred - output) ** 2))
    #exit(0)
    # then we have an in-domain y model and an out-of-domain env model
    # use both of them to estimate ood y
    print('estimate y using out-of-domain data')
    x1_list, x2_list, y_list, env_list = trainer.inference(dataset_ood['test'])
    x1_list, x2_list, y_list, env_list = x1_list.detach().numpy(), x2_list.detach().numpy(), y_list.detach().numpy(), env_list.detach().numpy()
    input = np.array(x1_list).reshape(-1, 1)
    output = dataset_ood['test']['y']
    env_pred = model_env_test.predict(input)
    ic(np.mean((env_pred - dataset_ood['test']['env']) ** 2))
    input = np.concatenate((x1_list, x2_list, env_pred), axis=1)
    y_pred = model_y.predict(input)
    ic(np.mean((y_pred - output) ** 2))
    out_domain_MSE = np.mean((y_pred - output) ** 2)

    x1_gt = np.array(dataset_ood['test']['input'])[:,0]
    x2_gt = np.array(dataset_ood['test']['input'])[:,1]
    MI_x1 = eval_MI(x1_gt, x1_list)
    MI_x2 = eval_MI(x2_gt, x2_list)
    ic(MI_x1, MI_x2)
    # then compute the CI loss
    print('compute the CI loss')
    from tqdm import trange
    total_CIT_loss = 0
    length = len(dataset_ood['test'])
    for i in trange(0, length, 256):
        x1, x2, y, env = x1_list[i:i+256], x2_list[i:i+256], dataset_ood['test']['y'][i:i+256], dataset_ood['test']['env'][i:i+256]
        from CIT_warpper import run_CIT
        total_CIT_loss += run_CIT(x1, x2, y)[2] + run_CIT(env, x2, y)[2]
    average_CI_loss = total_CIT_loss / (length / 256)
    ic(average_CI_loss)
    # then compute the MI
    #exit(0)
    return in_domain_MSE, out_domain_MSE, average_CI_loss, MI_x1, MI_x2


def test_IRM_toy_model_withoutDomainShift(model_path):
    # load in-domain data and pretrained model
    from data import get_simulatedDataset_IRM
    dataset = get_simulatedDataset_IRM(n=256000, group_number=1)
    dataset = dataset.train_test_split(test_size=0.2)
    dataset_ood = get_simulatedDataset_IRM(n=256000, group_number=2)
    dataset_ood = dataset_ood.train_test_split(test_size=0.2)
    few_shot_data = dataset_ood['train'][0:16]

    # load the model
    from toy_trainer_v4 import Trainer, estimate_upperBound_performance
    trainer = Trainer(dataset, lambda_ci = 1, lambda_R = 1)
    path = model_path
    trainer.load_state_dict(torch.load(path))
    trainer.show_params()
    from sklearn.linear_model import LinearRegression

    print('estimate env using in-domain data')
    x1_list, x2_list, y_list, env_list = trainer.inference(dataset['train'])
    x1_list, x2_list, y_list, env_list = x1_list.detach().numpy(), x2_list.detach().numpy(), y_list.detach().numpy(), env_list.detach().numpy()

    #input = np.array(x1_list).reshape(-1, 1)
    input = np.array(np.concatenate((x1_list, x2_list), axis=1))
    output = dataset['train']['env']
    model_env = LinearRegression().fit(input, output)
    ic(model_env.coef_, model_env.intercept_)
    env_pred = model_env.predict(input)
    ic(np.mean((env_pred - output) ** 2))
    # then estimate y using in-domain data
    #print('estimate y using in-domain data')
    #input = np.concatenate((x1_list, x2_list, env_pred), axis=1)
    #output = dataset['train']['y']
    #from sklearn.linear_model import Ridge
    #model_y = Ridge(alpha=1, fit_intercept=True).fit(input, output)

    print('estimate y using in-domain data qwqqqqqqqqqqq')
    env_gt = output
    # mix the env_pred and env_input with prob 0.5
    env_input = torch.where(torch.rand(env_pred.shape) > 0.01, torch.tensor(env_pred), torch.tensor(env_gt)).reshape(-1, 1)
    env_input = env_input.detach().numpy()
    input = np.concatenate((x1_list, x2_list, env_input), axis=1)
    output = dataset['train']['y']
    model_y = LinearRegression().fit(input, output)

    ic(model_y.coef_, model_y.intercept_)
    y_pred = model_y.predict(input)
    ic(np.mean((y_pred - output) ** 2))
    in_domain_MSE = np.mean((y_pred - output) ** 2)

    # then estimate env using out-of-domain data
    print('estimate env using out-of-domain data')
    x1_list, x2_list, y_list, env_list = trainer.compute_batch(few_shot_data, train=False)
    x1_list, x2_list, y_list, env_list = x1_list.detach().numpy(), x2_list.detach().numpy(), y_list.detach().numpy(), env_list.detach().numpy()
    #input = np.array(x1_list).reshape(-1, 1)
    input = np.array(np.concatenate((x1_list, x2_list), axis=1))
    output = few_shot_data['env'].detach().numpy()
    model_env_test = model_env
    # manually calculate the intercept
    model_env_test.intercept_ = np.mean(output-input*model_env_test.coef_)/2
    #model_env_test = model_env_test.fit(input, output)
    #ic(model_env_test.coef_, model_env_test.intercept_)
    # fit the model
    #model_env_test = model_env_test.fit(input, output)
    ic(model_env_test.coef_, model_env_test.intercept_)
    env_pred = model_env_test.predict(input)
    ic(np.mean((env_pred - output) ** 2))
    #exit(0)
    # then we have an in-domain y model and an out-of-domain env model
    # use both of them to estimate ood y
    print('estimate y using out-of-domain data')
    x1_list, x2_list, y_list, env_list = trainer.inference(dataset_ood['test'])
    x1_list, x2_list, y_list, env_list = x1_list.detach().numpy(), x2_list.detach().numpy(), y_list.detach().numpy(), env_list.detach().numpy()
    input = np.concatenate((x1_list, x2_list), axis=1)
    output = dataset_ood['test']['y']
    env_pred = model_env_test.predict(input)
    ic(np.mean((env_pred - dataset_ood['test']['env']) ** 2))
    input = np.concatenate((x1_list, x2_list, env_pred), axis=1)
    y_pred = model_y.predict(input)
    ic(np.mean((y_pred - output) ** 2))
    ic(np.mean((y_pred - output)))
    out_domain_MSE = np.mean((y_pred - output) ** 2)

    x1_gt = np.array(dataset_ood['test']['input'])[:,0]
    x2_gt = np.array(dataset_ood['test']['input'])[:,1]
    MI_x1 = eval_MI(x1_gt, x1_list)
    MI_x2 = eval_MI(x2_gt, x2_list)
    ic(MI_x1, MI_x2)
    # then compute the CI loss
    exit(0)
    print('compute the CI loss')
    from tqdm import trange
    total_CIT_loss = 0
    length = len(dataset_ood['test'])
    for i in trange(0, length, 256):
        x1, x2, y, env = x1_list[i:i+256], x2_list[i:i+256], dataset_ood['test']['y'][i:i+256], dataset_ood['test']['env'][i:i+256]
        from CIT_warpper import run_CIT
        total_CIT_loss += run_CIT(x1, x2, y)[2] + run_CIT(env, x2, y)[2]
    average_CI_loss = total_CIT_loss / (length / 256)
    ic(average_CI_loss)
    # then compute the MI
    #exit(0)
    return in_domain_MSE, out_domain_MSE, average_CI_loss, MI_x1, MI_x2

def iterate_model_path(reuslt_dir):
    # find all dir name in the result dir
    dir_list = os.listdir(reuslt_dir)
    # find all model path
    model_path_list = []
    for dir in dir_list:
        print(dir)
        value = dir[9:-4]
        model_path = os.path.join(reuslt_dir, dir, "best.pt")
        if os.path.exists(model_path):
            model_path_list.append((value, model_path))
    return model_path_list

def generate_OOD_baseline():
    from sklearn.linear_model import LinearRegression

    # 7. eval the model using linear regression in few-shot setting
    # the baseline is simply use all the features as the input
    # the improved version is to use feature x_1 to estimate env, and use all the features to estimate y

    dataset_ood = get_simulatedDataset_IRM(n=256000, group_number=2)
    dataset_ood = dataset_ood.train_test_split(test_size=0.2)
    few_shot_data = dataset_ood['train'][:16]

    test_x1 = np.array(dataset_ood['test']['input'])[:, 0].reshape(-1, 1)
    test_x2 = np.array(dataset_ood['test']['input'])[:, 1].reshape(-1, 1)
    test_env = np.array(dataset_ood['test']['env']).reshape(-1, 1)

    # 1. baseline
    train_x1 = np.array(few_shot_data['input'])[:, 0].reshape(-1, 1)
    train_x2 = np.array(few_shot_data['input'])[:, 1].reshape(-1, 1)
    input = np.concatenate((train_x1, train_x2), axis=1)
    output = few_shot_data['y']
    model = LinearRegression().fit(input, output)
    pred_y = model.predict(input)
    print('baseline train: ', np.mean((pred_y - output) ** 2))
    # eval
    input = np.concatenate((test_x1, test_x2), axis=1)
    # print(input.shape)
    output = dataset_ood['test']['y']
    pred_y = model.predict(input)
    # ic(pred_y[:10], output[:10])
    print('baseline test: ', np.mean((pred_y - output) ** 2))
    ic(model.coef_, model.intercept_)

if __name__ == '__main__':
    #IRM_toy_model("./result/simulated_IRM_toyModel/256000_1_0.0_0.3/best.pt")
    #IRM_toy_model("./result/simulated_IRM_toyModel_v4_newData_v2/256000_1_1.0_0.3/best.pt")
    #generate_OOD_baseline()
    #test_IRM_toy_model_withoutDomainShift("./result/simulated_IRM_toyModel_v4.1_newData_v2/25600_1_1_0.3/best.pt")
    test_IRM_toy_model_withoutDomainShift("./result/simulated_IRM_toyModel_v4_newData_v2/256000_1_1.0_0.3/best.pt")
    exit(0)
    model_path_list = iterate_model_path("./result/simulated_IRM_toyModel_v4_newData_v2")
    result_dict = {"lambda_CI_value": [],
                     "in_domain_MSE": [],
                        "out_domain_MSE": [],
                        "average_CI_loss": [],
                        "MI_x1": [],
                        "MI_x2": []}
    for value, model_path in model_path_list:
        in_domain_MSE, out_domain_MSE, average_CI_loss, MI_x1, MI_x2 = test_IRM_toy_model(model_path)
        from utils import to_numpy
        in_domain_MSE = to_numpy(in_domain_MSE)
        out_domain_MSE = to_numpy(out_domain_MSE)
        average_CI_loss = to_numpy(average_CI_loss)
        MI_x1 = to_numpy(MI_x1)
        MI_x2 = to_numpy(MI_x2)
        result_dict["lambda_CI_value"].append(value)
        result_dict["in_domain_MSE"].append(in_domain_MSE)
        result_dict["out_domain_MSE"].append(out_domain_MSE)
        result_dict["average_CI_loss"].append(average_CI_loss)
        result_dict["MI_x1"].append(MI_x1)
        result_dict["MI_x2"].append(MI_x2)
    df = pd.DataFrame(result_dict)
    df.to_csv("./result/simulated_IRM_toyModel_v4_newData_v2/result.csv")
