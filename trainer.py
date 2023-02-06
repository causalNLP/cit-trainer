# pipeline: dataset_dict; graph;
# mixed feature -> high-dim point -> result point
from model import ToyEstimator, ToyRegresser
from CIT_warpper import run_CIT

import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import torch.optim as optim
import numpy as np
from tqdm import trange
from icecream import ic

from utils import *

def preprocessing_dataset(dataset_dict, batch_size, keys_list = ['X_indY', 'X_indZ', 'X_YnZ']):
    # input: dict of 2-d list/torch.tensor; output: torch.tensor, split_sign
    # 1. X_indY, X_indZ, X_YnZ -> mixed_X
    # 2. None = [[0], [0], ..., [0]]'
    # used to mix the disentangled dataset to a single dataset
    # 相当于强行把simulated dataset混合在一起
    total_list = []
    split_sign = []
    for i in keys_list:
        if i == 'None':
            total_list.append(torch.tensor(np.zeros((batch_size, 1))))
        else:
            total_list.append(torch.tensor(dataset_dict[i]))
        split_sign.append(total_list[-1].shape[-1])
    return torch.cat(tuple(total_list), 1), split_sign

"""
def preprocessing_dataset_nlp(dataset_dict, batch_size, key_list = ['text']):
    # Just do the tokenlize and align step
    # 对于nlp dataset来说，bert embedding（或者直接token list）就直接是语义了，所以不需要再做什么了
    # 可以都试试
    pass
"""

class Trainer(nn.Module):
    #随机选择80%的作为training; 20%的作为test: randomshuffle+traintest split
    def __init__(self, dataset, direct_causes, model_pairs, CI_pairs, lambda_ci = 0.01, lambda_R = 1, output_dir = './output/'):
        # Input:
        # dataset: huggingface dataset
        # direct_causes: list of str, the list of direct causes of the target, using all of them to inference the target
        # model_pairs: [x_key, y_key]; x‘->model->y, please use topology order
        # CI_pairs: [x_key, y_key, z_key]; constraint x \bot y | z
        # lambda_ci: the weight of CI loss
        super().__init__()
        self.device = 'cpu'

        self.dataset = dataset
        # self.input_columns = input_columns
        self.dataset_size = len(dataset)
        self.trainset_size = int(self.dataset_size*0.8)
        self.models = []

        self.estimator_model_dict = {}
        # 记录这条边的起点

        self.estimator_batch_temp = {}
        #self.modelpair_batch_temp = []
        self.result_temp = {}

        self.model_pairs = model_pairs
        # 这里的格式是从[x_key, y_key]; 意思是从 x->model->y
        # In IRM model example, the model pair is [input, x_1], [input, x_2], [input, env]
        self.invisible_variables = []

        for i in range(len(self.model_pairs)):
            model_pair = self.model_pairs[i]
            if (self.estimator_model_dict.get(model_pair[1])==None):

                if (model_pair[0] in self.invisible_variables):
                    input_dim = 2
                else:
                    input_dim = len(dataset[0][model_pair[0]])

                output_dim = 2
                if (model_pair[1] in dataset[0].keys()):
                    output_dim = len(dataset[0][model_pair[1]])

                elif (not (model_pair[1] in self.invisible_variables)):
                    self.invisible_variables.append(model_pair[1])
                self.estimator_model_dict[model_pair[1]] = ToyEstimator(input_dim = input_dim, output_dim = output_dim)
                self.models.append(self.estimator_model_dict[model_pair[1]])
            # In our simplified model, we assume that the estimator only have one causes,
            # It also can have many causes, the code should be improved

        self.direct_causes = direct_causes
        sample_dict = {}
        for i in direct_causes:
            # check if there is i column in dataset, if it is, use it, else use [0, 0]
            if (i in dataset[0].keys()):
                sample_dict[i] = dataset[i][0]
            else:
                sample_dict[i] = [0, 0]
                # in case of the direct cause is not in the dataset (which means it is invisible)
        self.regressor = ToyRegresser(sample_dict)
        self.models.append(self.regressor)
        # regresser不唯一，每一条model pair都是, 后续代码里需要修改

        self.models = nn.ModuleList(self.models)
        #ic(self.models)
        #exit(0)
        self.CI_pairs = CI_pairs # n*3*i
        self.lambda_ci = lambda_ci
        self.lambda_R = lambda_R
        self.loss = None
        self.CI_loss = None
        self.optimizer = optim.Adam(self.parameters())
        self.MSELoss = MSELoss()
        self.output_dir = output_dir # Use this path to save the model checkpoint
        if (not os.path.exists(self.output_dir)):
            os.mkdir(self.output_dir)
        self.debug_flag = False

    def infernece_one_sample(self, dataset_dict):
        # infrence single will defined datapoint
        # interface to just simply load the model
        evaluated_dict = {}
        for i in range(len(self.model_pairs)):
            model_pair = self.model_pairs[i]
            if (model_pair[0] in dataset_dict.keys()):
                input_datapoint = torch.tensor(dataset_dict[model_pair[0]])
            else:
                input_datapoint = torch.tensor(evaluated_dict[model_pair[0]])
            evaluated_dict[model_pair[1]] = self.estimator_model_dict[model_pair[1]](input_datapoint)
        result = self.regressor(evaluated_dict, dataset_dict['y'])
        return evaluated_dict, result[0], result[1] # evaluation of all variable, regression result, loss

    def inference(self, dataset = None, testSet = True, batch_size = 32):
        # inference the whole dataset
        # Output:
        # result_dict: dict of list of tensor, the evaluated value of each variable,
        if (dataset != None):
            self.dataset = dataset
            # it is a risky operation, must guarantee that the dataset has the same size as the one used to train the model

        result_dict = {}
        for model_pair in self.model_pairs:
            result_dict[model_pair[1]] = []
        result_dict['y'] = []
        result_dict['loss'] = []

        if (testSet):
            begin, end = self.trainset_size, self.dataset_size
        else:
            begin, end = 0, self.trainset_size
        for i in trange(begin, end, batch_size):
            if (i+batch_size>end):
                batch_size = end-i
            batch_dict = self.dataset[i:i+batch_size]
            evaluated_dict_batch, result_batch, loss_batch = self.infernece_one_sample(batch_dict)
            for key in evaluated_dict_batch.keys():
                if (result_dict.get(key)==None):
                    result_dict[key] = []
                result_dict[key].append(evaluated_dict_batch[key])
            result_dict['y'].append(result_batch)
            result_dict['loss'].append(loss_batch)
        for key in result_dict.keys():
            if (key=='loss'):
                result_dict[key] = [to_list(i) for i in result_dict[key]]
            else:
                result_dict[key] = to_list(torch.cat(result_dict[key], dim = 0))
        return result_dict

    # Too large batch size CIT?
    # What if only allow the gradient flow through 8 datapoint?
    # First inference 512 without gradient flow, then train 8 with gradient flow
    def train_epoch(self, batch_size = 8, batch_size_CIT = 512):
        idx_shuffle = np.arange(self.trainset_size)
        np.random.shuffle(idx_shuffle)
        running_loss = 0
        runing_CI_loss = 0
        ic(0, self.trainset_size, batch_size_CIT)
        for base in trange(0, self.trainset_size, batch_size_CIT):
            # First inference 512 with no grad
            if (base+batch_size_CIT>self.trainset_size):
                #ic(i, i+batch_size_CIT, self.trainset_size)
                # if remain datapoint is less than 512, then drop them
                break
            #print(f"Now is {base}, first do the inference without gradient flow")
            self.batch_idx = idx_shuffle[base:base+batch_size_CIT]
            self.estimator_batch_temp = {}
            for model_name in self.estimator_model_dict:
                self.estimator_batch_temp[model_name] = []
            with torch.no_grad():
                for i in range(0, batch_size_CIT, batch_size):
                    batch_data = self.dataset[self.batch_idx[i:i+batch_size]]
                    # print("we calculate the batch_data_idx is ", self.batch_idx[i:i+batch_size])
                    #ic(self.batch_idx[i:i+batch_size], batch_data)

                    for model_pair in self.model_pairs:
                        if (model_pair[0] in batch_data.keys()):
                            input_datapoint = torch.tensor(batch_data[model_pair[0]])
                        else:
                            input_datapoint = self.estimator_batch_temp[model_pair[0]][-1]
                        self.estimator_batch_temp[model_pair[1]].append(self.estimator_model_dict[model_pair[1]](input_datapoint))

            # Then inference 8*64 with grad and loss backpropagation
            if (base != 0):
                print(f"Now is {base}, finish the inference without gradient flow, start the inference with gradient flow, running loss is {running_loss/base}, running CI loss is {runing_CI_loss/base}")
            # exit(0)
            for i in range(0, batch_size_CIT, batch_size):
                self.optimizer.zero_grad()

                batch_data = self.dataset[self.batch_idx[i:i+batch_size]]
                #print("we calculate the batch_data_idx is ", self.batch_idx[i:i+batch_size])

                for model_pair in self.model_pairs:
                    if (model_pair[0] in batch_data.keys()):
                        input_datapoint = torch.tensor(batch_data[model_pair[0]])
                    else:
                        input_datapoint = self.estimator_batch_temp[model_pair[0]][i//batch_size]
                    self.estimator_batch_temp[model_pair[1]][i // batch_size] = self.estimator_model_dict[model_pair[1]](input_datapoint)
                    #ic(model_pair[1], self.estimator_batch_temp[model_pair[1]][i // batch_size])
                """
                input_batch_data, _ = preprocessing_dataset(batch_data, batch_size)
                for model_name in self.estimator_model_dict:
                    self.estimator_batch_temp[model_name][i//batch_size] = self.estimator_model_dict[model_name](input_batch_data)
                """

                data = {}
                loss_list = []
                for model_name in self.direct_causes:
                    if model_name in self.invisible_variables:
                        # recalculate the data with the gradient if it is invisible
                        data[model_name] = self.estimator_batch_temp[model_name][i//batch_size]
                    else:
                        data[model_name] = batch_data[model_name]
                        # use the real data if it is visible
                        # calculate MSE loss between real data and estimated data
                        target = torch.tensor(batch_data[model_name])
                        if (torch.cuda.is_available()):
                            target = target.cuda()
                        #ic(self.estimator_batch_temp[model_name][i//batch_size], target)
                        loss_list.append(self.MSELoss(self.estimator_batch_temp[model_name][i//batch_size], target)*self.lambda_R)
                        # Regression loss coefficient

                #ic(data, batch_data['y'])
                #exit(0)
                result = self.regressor(data, batch_data['y'])
                self.loss = result[1]

                #ic(loss_list)
                self.loss += sum(loss_list)
                self.CI_loss = None
                for CI_pair in self.CI_pairs:
                    self.run_CI(CI_pair[0], CI_pair[1], CI_pair[2])
                running_loss += self.loss.item()
                if (self.CI_loss is not None):
                    runing_CI_loss += self.CI_loss.item()
                    self.loss += self.CI_loss * self.lambda_ci
                #ic(self.loss)

                if (self.loss != 0):
                    #ic(self.loss, self.CI_loss, loss_list)
                    self.loss.backward()
                    self.optimizer.step()
                    # Clean the gradient
                    self.optimizer.zero_grad()

                for model_name in self.estimator_model_dict:
                    self.estimator_batch_temp[model_name][i // batch_size].detach_()

                self.loss, self.CI_loss = None, None
        running_loss /= self.trainset_size//batch_size_CIT*batch_size_CIT
        runing_CI_loss /= self.trainset_size//batch_size_CIT*batch_size_CIT
        ic(running_loss, runing_CI_loss)
        return running_loss, runing_CI_loss

    def train_epochs(self, epoch_number, batch_size = 8, batch_size_CIT = 512):
        epoch_number_list, loss_list, R_loss_list, CI_loss_list, pvalue_list = [], [], [], [], []
        loss, R_loss, CI_loss, pvalue, pred, target = self.evaluate(batch_size, batch_size_CIT)
        epoch_number_list.append(0)
        loss_list.append(loss)
        R_loss_list.append(R_loss)
        CI_loss_list.append(CI_loss)
        pvalue_list.append(pvalue)

        for i in range(epoch_number):
            print("Epoch: ", i)
            self.train_epoch(batch_size, batch_size_CIT)
            loss, R_loss, CI_loss, pvalue, pred, target = self.evaluate(batch_size, batch_size_CIT)
            epoch_number_list.append(i+1)
            loss_list.append(loss)
            R_loss_list.append(R_loss)
            CI_loss_list.append(CI_loss)
            pvalue_list.append(pvalue)

            ic(loss, R_loss, CI_loss, pvalue)
            #ic(self.estimator_batch_temp)
            # TODO: Save the checkpoint after each epoch
            model_save_name = f"epoch_{i}.pth"
            model_save_name = os.path.join(self.output_dir, model_save_name)
            torch.save(self.models.state_dict(), model_save_name)
        # save the result into a csv file
        result = pd.DataFrame({'epoch': epoch_number_list, 'loss': loss_list, 'R_loss': R_loss_list,
                              'CI_loss': CI_loss_list, 'pvalue': pvalue_list})
        result.to_csv(os.path.join(self.output_dir, "result.csv"), index = False)
        return loss_list, CI_loss_list, pvalue_list

    def load_checkpoint(self, checkpoint_path):
        self.models.load_state_dict(torch.load(checkpoint_path))
        self.regressor = self.models[-1]
        # self.models.eval()

    # this function is used to evaluate the model, which is the performance on the inference mode on the test set
    def evaluate(self, batch_size = 8, batch_size_CIT = 512, testSet = True):
        # Average MSE loss; Average CI loss; Total CIT P-Value; pred; target;
        idx_shuffle = np.arange(self.trainset_size, self.dataset_size)
        #ic(idx_shuffle)
        loss_list, CI_loss_list, avePV_list , pred_list, target_list = [], [], [], [], []
        R_loss_list = []

        begin, end = self.trainset_size, self.dataset_size
        if (not testSet):
            begin, end = 0, self.trainset_size
            idx_shuffle = np.arange(self.trainset_size)
        for i in range(begin, end, batch_size_CIT):
            if (i+batch_size_CIT>self.dataset_size):
                # if remain datapoint is less than 512, then drop them
                break
            self.batch_idx = idx_shuffle[i-begin:i+batch_size_CIT-begin]
            self.estimator_batch_temp = {}
            for model_name in self.estimator_model_dict:
                self.estimator_batch_temp[model_name] = []
            with torch.no_grad():
                for j in range(0, batch_size_CIT, batch_size):
                    batch_data = self.dataset[self.batch_idx[j:j+batch_size]]
                    #ic(batch_idx[i:i+batch_size], batch_data)
                    """
                    input_batch_data, _ = preprocessing_dataset(batch_data, batch_size)
                    for model_name in self.estimator_model_dict:
                        self.estimator_batch_temp[model_name].append(self.estimator_model_dict[model_name](input_batch_data))
                    """

                    for model_pair in self.model_pairs:
                        #ic(batch_data, model_pair)

                        if (model_pair[0] in self.invisible_variables):
                            input_datapoint = self.estimator_batch_temp[model_pair[0]][-1]
                        else:
                            input_datapoint = torch.tensor(batch_data[model_pair[0]])
                        self.estimator_batch_temp[model_pair[1]].append(self.estimator_model_dict[model_pair[1]](input_datapoint))

                    data = {}
                    loss_list_sample = []
                    for model_name in self.direct_causes:
                        if (model_name in self.invisible_variables):
                            data[model_name] = self.estimator_batch_temp[model_name][j//batch_size]
                        else:
                            data[model_name] = self.estimator_batch_temp[model_name][j//batch_size]
                            # use the real data if it is visible
                            # calculate MSE loss between real data and estimated data
                            target = torch.tensor(batch_data[model_name])
                            if (torch.cuda.is_available()):
                                target = target.cuda()
                            loss_list_sample.append(self.MSELoss(self.estimator_batch_temp[model_name][j//batch_size], target))

                    result = self.regressor(data, batch_data['y'])
                    pred_list.append(to_list(result[0]))
                    R_loss_list.append(to_list(sum(loss_list_sample)))
                    loss_list.append(to_list(result[1]))
                    target_list.append(to_list(batch_data['y']))

                self.CI_loss = None
                for CI_pair in self.CI_pairs:
                    # ic(CI_pair)
                    #if (i==0):
                    #    self.debug_flag = True
                    avePV_list.append(self.run_CI(CI_pair[0], CI_pair[1], CI_pair[2]))
                    self.debug_flag = False
                CI_loss_list.append(to_list(self.CI_loss))

        #ic(len(loss_list), CI_loss_list)
        return np.mean(loss_list)/batch_size, np.mean(R_loss_list)/batch_size, np.mean(CI_loss_list)/batch_size_CIT, np.mean(avePV_list), np.concatenate(pred_list), np.concatenate(target_list)
        # return average_loss, total_pvalue, total_pred, total_target

    def run_CI(self, X, Y, Z):
        # ic(X, Y, Z)
        X_embedding = self.get_subset_embedding(X)
        Y_embedding = self.get_subset_embedding(Y)
        Z_embedding = self.get_subset_embedding(Z)
        if (self.debug_flag):
            ic(X_embedding[:10], Y_embedding[:10], Z_embedding[:10])
        #print("Now we calculate the CI")
        #ic(X_embedding.shape, Y_embedding.shape, Z_embedding.shape)
        pvalue, test_stat, loss = run_CIT(X_embedding, Y_embedding, Z_embedding)
        #ic(pvalue, test_stat, loss)
        if (type(loss) != torch.Tensor):
            return pvalue
        #ic(loss)
        if self.CI_loss == None:
            self.CI_loss = loss
        else:
            self.CI_loss += loss
        return pvalue

    def get_subset_embedding(self, model_name):
        # We always use gt data if we can use it to estimate the embedding
        if (self.debug_flag):
            ic(model_name, self.dataset[0].get(model_name) != None)
        if (self.dataset[0].get(model_name) != None):
            return preprocessing_dataset(self.dataset[self.batch_idx], len(self.dataset), [model_name])[0]
        else:
            return torch.cat(self.estimator_batch_temp[model_name], dim=0)

def test_simulated_dataset():
    from data import get_dataset
    dataset_dict = np.load("/cluster/project/sachan/zhiheng/CI_Train/dataset/simulated/causal_simulated_1k_linear.npy",
                           allow_pickle = True).tolist()
    # change dataset_dict to a dict of torch.tensor
    for key in dataset_dict:
        dataset_dict[key] = torch.tensor(dataset_dict[key])
    # change dataset_dict to a huggingface dataset
    from datasets import Dataset
    dataset = Dataset.from_dict(dataset_dict)
    ic(dataset)
    ic(dataset['Z'][0:10])
    trainer = Trainer(dataset, ['X_indY', 'X_indZ', 'X_YnZ'], [], [])
    trainer.train_epochs(10, 8, 512)
    loss, pvalue, pred, target = trainer.evaluate(8, 512)
    print(loss, pvalue, pred, target)

def test_CIT_trainer():
    from data import get_simulatedDataset_IRM
    dataset = get_simulatedDataset_IRM(n = 10000)
    ic(dataset)
    from trainer import Trainer
    trainer = Trainer(dataset, ['x_1', 'x_2', 'env'],
                      [['input', 'x_1'], ['input', 'x_2'], ['input', 'env']],
                      [['x_1', 'x_2', 'y'], ['env', 'x_2', 'y']])
    if (torch.cuda.is_available()):
        trainer.cuda()
    #trainer.train_epochs(10, 16, 128)
    loss, R_loss, CI_loss, pvalue, pred, target = trainer.evaluate(16, 128)
    ic(loss, CI_loss, pvalue)
    trainer.train_epochs(10, 16, 128)
    loss, R_loss, CI_loss, pvalue, pred, target = trainer.evaluate(16, 128)
    ic(loss, CI_loss, pvalue)
# Use
if __name__ == '__main__':
    test_CIT_trainer()