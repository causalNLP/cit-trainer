# pipeline: dataset_dict; graph;
# mixed feature -> high-dim point -> result point
from model import ToyEstimator, ToyRegresser
from CIT_warpper import run_CIT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import trange
from icecream import ic

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

def preprocessing_dataset_nlp(dataset_dict, batch_size, key_list = ['text']):
    # Just do the tokenlize and align step
    # 对于nlp dataset来说，bert embedding（或者直接token list）就直接是语义了，所以不需要再做什么了
    # 可以都试试
    pass

class Trainer(nn.Module):
    #随机选择80%的作为training; 20%的作为test: randomshuffle+traintest split
    def __init__(self, dataset, input_columns, model_pairs, CI_pairs, lambda_i = 0.01):
        super().__init__()
        self.device = 'cpu'

        self.dataset = dataset
        self.input_columns = input_columns
        self.dataset_size = len(dataset)
        self.trainset_size = int(self.dataset_size*0.8)
        self.models = []

        self.estimator_model_dict = {}
        # 记录这条边的起点

        self.estimator_batch_temp = {}
        #self.modelpair_batch_temp = []
        self.result_temp = {}

        self.model_pairs = model_pairs
        # 这里的格式是从[x_key, y_key, model]; 意思是从estimator x->model->y

        for i in range(len(self.model_pairs)):
            model_pair = self.model_pairs[i]
            if (self.estimator_model_dict.get(model_pair[0])==None):
                self.estimator_model_dict[model_pair[0]] = Estimator()
                self.models.append(self.estimator_model_dict[model_pair[0]])
            # estimator是个dict 因为是唯一的
            self.model_pairs[i].append(Regresser())
            self.models.append(self.model_pairs[i][-1])
            # regresser不唯一，每一条model pair都是, 后续代码里需要修改

        self.models = nn.ModuleList(self.models)
        self.CI_pairs = CI_pairs # n*3*i
        self.lambda_i = lambda_i
        self.loss = None
        self.CI_loss = None
        self.optimizer = optim.Adam(self.parameters())

    def infernece_one_sample(self, dataset_dict, batch_size):
        # infrence single will defined datapoint
        # interface to just simply load the model
        input_datapoint, _ = preprocessing_dataset(dataset_dict, batch_size, self.input_columns)
        evaluated_dict = {}
        for estimator_name in self.estimator_model_dict:
            evaluated_dict[estimator_name] = self.estimator_model_dict[estimator_name](input_datapoint)
        result_list = []
        loss_list = []
        for model_pair in self.model_pairs:
            result = model_pair[-1](evaluated_dict[model_pair[0]], dataset_dict[model_pair[1]])
            result_list.append(result[0])
            loss_list.append(result[1])
        return evaluated_dict, result_list, loss_list

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
                    input_batch_data, _ = preprocessing_dataset(batch_data, batch_size, self.input_columns)
                    for model_name in self.estimator_model_dict:
                        self.estimator_batch_temp[model_name].append(self.estimator_model_dict[model_name](input_batch_data))
                '''
                for model_name in estimator_model_dict:
                    model = estimator_model_dict[model_name]
                    self.estimator_result_temp[model_name] = []
                    for i in range(0, batch_size_CIT, batch_size):
                        data = self.dataset[i:i+batch_size][self.input_column]
                        self.estimator_result_temp[model_name].append(model(data))

                for model_name in regresser_model_dict:
                    model = regresser_model_dict[model_name]
                    self.regresser_result_temp[model_name] = []
                    for i in range(batch_size_CIT//batch_size):
                        data = self.estimator_result_temp[self.regresser_source_dict[model_name]][i]
                        self.regresser_result_temp[model_name].append(model(data))
                '''
            # Then inference 8*64 with grad and loss backpropagation
            print(f"Now is {base}, finish the inference without gradient flow, start the inference with gradient flow")
            # exit(0)
            for i in range(0, batch_size_CIT, batch_size):
                self.optimizer.zero_grad()
                loss_list = []

                batch_data = self.dataset[self.batch_idx[i:i+batch_size]]
                #print("we calculate the batch_data_idx is ", self.batch_idx[i:i+batch_size])
                input_batch_data, _ = preprocessing_dataset(batch_data, batch_size, self.input_columns)
                for model_name in self.estimator_model_dict:
                    self.estimator_batch_temp[model_name][i//batch_size] = self.estimator_model_dict[model_name](input_batch_data)

                for model_pair_num in range(len(self.model_pairs)):
                    data = self.estimator_batch_temp[self.model_pairs[model_pair_num][0]][i//batch_size]
                    result = self.model_pairs[model_pair_num][-1](data, batch_data[self.model_pairs[model_pair_num][1]])
                    loss_list.append(result[1])

                #ic(loss_list)
                self.loss = sum(loss_list)
                self.CI_loss = None
                for CI_pair in self.CI_pairs:
                    self.run_CI(CI_pair[0], CI_pair[1], CI_pair[2])
                running_loss += self.loss.item()
                runing_CI_loss += self.CI_loss.item()

                self.loss += self.CI_loss * self.lambda_i
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
        for i in range(epoch_number):
            print("Epoch: ", i)
            self.train_epoch(batch_size, batch_size_CIT)
            loss, CI_loss, pvalue, pred, target = self.evaluate(8, 128)
            ic(loss, CI_loss, pvalue)

    def evaluate(self, batch_size = 8, batch_size_CIT = 512):
        # Average MSE loss; Average CI loss; Total CIT P-Value; pred; target;
        idx_shuffle = np.arange(self.trainset_size, self.dataset_size)
        #ic(idx_shuffle)
        loss_list, CI_loss_list, avePV_list , pred_list, target_list = [], [], [], [], []
        for i in range(self.trainset_size, self.dataset_size, batch_size_CIT):
            if (i+batch_size_CIT>self.dataset_size):
                # if remain datapoint is less than 512, then drop them
                break
            self.batch_idx = idx_shuffle[i-self.trainset_size:i+batch_size_CIT-self.trainset_size]
            self.estimator_batch_temp = {}
            for model_name in self.estimator_model_dict:
                self.estimator_batch_temp[model_name] = []
            with torch.no_grad():
                for i in range(0, batch_size_CIT, batch_size):
                    batch_data = self.dataset[self.batch_idx[i:i+batch_size]]
                    #ic(batch_idx[i:i+batch_size], batch_data)
                    input_batch_data, _ = preprocessing_dataset(batch_data, batch_size, self.input_columns)
                    for model_name in self.estimator_model_dict:
                        self.estimator_batch_temp[model_name].append(self.estimator_model_dict[model_name](input_batch_data))
                    for model_pair_num in range(len(self.model_pairs)):
                        data = self.estimator_batch_temp[self.model_pairs[model_pair_num][0]][i // batch_size]
                        result = self.model_pairs[model_pair_num][-1](data, batch_data[self.model_pairs[model_pair_num][1]])
                        pred_list.append(result[0])
                        loss_list.append(result[1])
                        target_list.append(batch_data[self.model_pairs[model_pair_num][1]])
                self.CI_loss = None
                for CI_pair in self.CI_pairs:
                    avePV_list.append(self.run_CI(CI_pair[0], CI_pair[1], CI_pair[2]))
                CI_loss_list.append(self.CI_loss)

        return np.mean(loss_list), np.mean(CI_loss_list), np.mean(avePV_list), np.concatenate(pred_list), np.concatenate(target_list)
        # return average_loss, total_pvalue, total_pred, total_target

    def run_CI(self, X, Y, Z):
        X_embedding = self.get_subset_embedding(X)
        Y_embedding = self.get_subset_embedding(Y)
        Z_embedding = self.get_subset_embedding(Z)
        #print("Now we calculate the CI")
        #ic(X_embedding.shape, Y_embedding.shape, Z_embedding.shape)
        pvalue, test_stat, loss = run_CIT(X_embedding, Y_embedding, Z_embedding)
        if self.CI_loss == None:
            self.CI_loss = loss
        else:
            self.CI_loss += loss
        return pvalue

    def get_subset_embedding(self, model_name):
        if (self.estimator_batch_temp.get(model_name) == None):
            return preprocessing_dataset(self.dataset[self.batch_idx], len(self.batch_idx), [model_name])[0]
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
    from data import get_dataset
    dataset = get_dataset()
    ic(dataset)
    from trainer import Trainer
    trainer = Trainer(dataset, ['X_indY', 'X_indZ', 'X_YnZ'],
                      [['X_indY', 'Z'], ['X_YnZ', 'Z'], ['X_indZ', 'Y'], ['X_YnZ', 'Y']],
                      [['X_indY', 'X_YnZ', 'Z'], ['X_indZ', 'X_YnZ', 'None']])
    trainer.train_epochs(10, 8, 128)
    loss, CI_loss, pvalue, pred, target = trainer.evaluate(8, 128)
    ic(loss, CI_loss, pvalue)

# Use
if __name__ == '__main__':
    test_CIT_trainer()