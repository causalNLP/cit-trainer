"""
A more general version for gradient update
1. for each model with ground truth, we use the ground truth to train the model(without update other models)
2. for each model without ground truth, we use the loss_task and loss_CIT to train the model(the gradient can be updated by other models)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from icecream import ic

from utils import *


class Trainer(nn.Module):
    def __init__(self, dataset, lambda_ci=0.05, lambda_R=0.3, gt_rate = 0.1, output_dir='./output/'):
        super(Trainer, self).__init__()
        self.dataset = dataset['train']
        self.test_dataset = dataset['test']
        self.lambda_ci = lambda_ci
        self.lambda_R = lambda_R
        self.output_dir = output_dir
        self.env_model = nn.Linear(2, 1)
        self.x1_model = nn.Linear(2, 2)
        self.x2_model = nn.Linear(2, 2)
        self.y_model = nn.Linear(5, 1)
        self.gt_rate = gt_rate # the rate of ground truth of env during training

        #self.optimizer = optim.Adam(self.parameters(), lr=0.1)
        # Define the optimizer for each model
        self.optimizer_dict = {
            'env': optim.Adam(self.env_model.parameters(), lr=0.1),
            'x1': optim.Adam(self.x1_model.parameters(), lr=0.1),
            'x2': optim.Adam(self.x2_model.parameters(), lr=0.1),
            'y': optim.Adam(self.y_model.parameters(), lr=0.1)
        }
        self.lr_scheduler_dict = {
            'env': optim.lr_scheduler.StepLR(self.optimizer_dict['env'], step_size=20, gamma=0.01),
            'x1': optim.lr_scheduler.StepLR(self.optimizer_dict['x1'], step_size=20, gamma=0.01),
            'x2': optim.lr_scheduler.StepLR(self.optimizer_dict['x2'], step_size=20, gamma=0.01),
            'y': optim.lr_scheduler.StepLR(self.optimizer_dict['y'], step_size=20, gamma=0.01)
        }
        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        self.loss_fn = nn.MSELoss()
        if torch.cuda.is_available():
            self.cuda()
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        # add log.txt to record the training process
        self.log_file = open(os.path.join(self.output_dir, 'log.txt'), 'w')

    def compute_batch(self, batch_datapoint, train=True):
        batch_datapoint['env'] = torch.tensor(batch_datapoint['env']).reshape(-1, 1)
        batch_datapoint['input'] = torch.tensor(batch_datapoint['input']).reshape(-1, 2)
        batch_datapoint['y'] = torch.tensor(batch_datapoint['y']).reshape(-1, 1)
        # change the data from double to float
        batch_datapoint['env'] = batch_datapoint['env'].float()
        batch_datapoint['input'] = batch_datapoint['input'].float()
        batch_datapoint['y'] = batch_datapoint['y'].float()
        if torch.cuda.is_available():
            batch_datapoint['env'] = batch_datapoint['env'].cuda()
            batch_datapoint['input'] = batch_datapoint['input'].cuda()
            batch_datapoint['y'] = batch_datapoint['y'].cuda()
        x1 = self.x1_model(batch_datapoint['input'])
        x2 = self.x2_model(batch_datapoint['input'])
        env = self.env_model(x1)
        env_gt = batch_datapoint['env']
        # mix the ground truth and the predicted env by the probability of self.gt_rate
        env_toTrain = torch.where(torch.rand_like(env) < self.gt_rate, env_gt, env)
        if train:
            y = self.y_model(torch.cat([x1, x2, batch_datapoint['env']], dim=1))
            #print("qwqqqqq")
        else:
            y = self.y_model(torch.cat([x1, x2, env], dim=1))
        return x1, x2, y, env

    # compute the loss for each batch, if optimizer_dict is not None, then update the model
    def compute_loss(self, batch_datapoint, optimizer_dict = None):
        from CIT_warpper import run_CIT
        x1, x2, y, env = self.compute_batch(batch_datapoint)
        loss = self.loss_fn(y, batch_datapoint['y'])
        # compute the conditional independence test loss
        # print(x1.shape, x2.shape, batch_datapoint['y'].shape, batch_datapoint['env'].shape)
        # print(run_CIT(x1, x2, batch_datapoint['y']))
        CIT_loss = run_CIT(x1, x2, batch_datapoint['y'])[2] * self.lambda_ci \
                   + run_CIT(batch_datapoint['env'], x2, batch_datapoint['y'])[2] * self.lambda_ci
        # compute the regression loss
        R_loss = self.lambda_R * (self.loss_fn(env, batch_datapoint['env']))
        loss_value, CIT_loss_value, R_loss_value = loss.item(), CIT_loss.item(), R_loss.item()
        if optimizer_dict is not None:
            # first update the model with ground truth, using R_loss
            optimizer_dict['env'].zero_grad()
            R_loss.backward(retain_graph=True)
            optimizer_dict['env'].step()
            # then update the model without ground truth and y, using loss and CIT_loss
            loss = loss + CIT_loss
            optimizer_dict['x1'].zero_grad()
            optimizer_dict['x2'].zero_grad()
            optimizer_dict['y'].zero_grad()
            loss.backward()
            optimizer_dict['x1'].step()
            optimizer_dict['x2'].step()
            optimizer_dict['y'].step()
        return loss_value, CIT_loss_value, R_loss_value

    def train_epochs(self, num_epochs = 60):
        loss_list, CIT_loss_list, R_loss_list = [], [], []
        best_eval_loss = 100000
        for epoch in range(num_epochs):
            print('epoch: {}'.format(epoch))
            # dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=256, shuffle=True)
            from tqdm import trange
            length = len(self.dataset)
            shuffle_idx = np.random.permutation(length)
            for i in trange(0, length, 256):
                if (i + 256 > length):
                    break
                batch_datapoint = self.dataset[shuffle_idx[i:i + 256]]
                loss, CIT_loss, R_loss = self.compute_loss(batch_datapoint, optimizer_dict=self.optimizer_dict)
                loss_list.append(loss)
                CIT_loss_list.append(CIT_loss)
                R_loss_list.append(R_loss)
                #self.optimizer.zero_grad()
                #loss += CIT_loss + R_loss
                #loss.backward()
                #self.optimizer.step()
            # print average loss
            # self.lr_scheduler.step()
            print('epoch: {}, loss: {}, CIT_loss: {}, R_loss: {}'.format(epoch, np.mean(loss_list),
                                                                         np.mean(CIT_loss_list), np.mean(R_loss_list)))
            # write the message to the log file
            with open(self.output_dir + 'log.txt', 'a') as f:
                f.write('epoch: {}, loss: {}, CIT_loss: {}, R_loss: {}'.format(epoch, np.mean(loss_list),
                                                                               np.mean(CIT_loss_list), np.mean(R_loss_list)))
            if (epoch % 1 == 0):
                self.show_params()
                eval_loss = self.evaluate()
                # save the model every 5 epochs on self.output_dir/{epoch}.pt
                torch.save(self.state_dict(), self.output_dir + '{}.pt'.format(epoch))
                print('model saved to {}'.format(self.output_dir + '{}.pt'.format(epoch)))
                # save the best model on self.output_dir/best.pt
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    torch.save(self.state_dict(), self.output_dir + 'best.pt')
                    print('model saved to {}'.format(self.output_dir + 'best.pt'))
        self.show_params()

    def load(self):
        self.load_state_dict(torch.load(self.output_dir + 'model.pt'))
        print('model loaded from {}'.format(self.output_dir + 'model.pt'))

    def show_params(self):  # print out the parameters of the model
        for name, param in self.state_dict().items():
            print(name, param)

    def evaluate(self):
        # self.mode_switch(2)
        loss_list, CIT_loss_list, R_loss_list = [], [], []
        from tqdm import trange
        length = len(self.test_dataset)
        for i in trange(0, length, 256):
            batch_datapoint = self.test_dataset[i:i + 256]
            loss, CIT_loss, R_loss = self.compute_loss(batch_datapoint)
            loss_list.append(loss)
            CIT_loss_list.append(CIT_loss)
            R_loss_list.append(R_loss)
        print('test loss: {}, test CIT_loss: {}, test R_loss: {}'.format(np.mean(loss_list), np.mean(CIT_loss_list),
                                                                         np.mean(R_loss_list)))
        with open(self.output_dir + 'log.txt', 'a') as f:
            f.write('test loss: {}, test CIT_loss: {}, test R_loss: {}'.format(np.mean(loss_list),
                                                                               np.mean(CIT_loss_list),
                                                                               np.mean(R_loss_list)))
        return np.mean(loss_list) + np.mean(CIT_loss_list) + np.mean(R_loss_list)
        # self.mode_switch(0)

    def inference(self, dataset):
        x1_list, x2_list, y_list, env_list = [], [], [], []
        length = len(dataset)
        for i in range(0, length, 256):
            batch_datapoint = dataset[i:i + 256]
            x1, x2, y, env = self.compute_batch(batch_datapoint, train=False)
            x1, x2, y, env = x1.detach().cpu(), x2.detach().cpu(), y.detach().cpu(), env.detach().cpu()
            x1_list.append(x1)
            x2_list.append(x2)
            y_list.append(y)
            env_list.append(env)
        x1_list = torch.cat(x1_list, dim=0)
        x2_list = torch.cat(x2_list, dim=0)
        y_list = torch.cat(y_list, dim=0)
        env_list = torch.cat(env_list, dim=0)
        return x1_list, x2_list, y_list, env_list

def load_target_state(trainer):
    state_dict = trainer.state_dict()
    state_dict['env_model.weight'] = torch.tensor([[1, 0]], dtype=torch.float32)
    state_dict['env_model.bias'] = torch.tensor([0], dtype=torch.float32)
    state_dict['x1_model.weight'] = torch.tensor([[1, 0]], dtype=torch.float32)
    state_dict['x1_model.bias'] = torch.tensor([0], dtype=torch.float32)
    state_dict['x2_model.weight'] = torch.tensor([[0, 1]], dtype=torch.float32)
    state_dict['x2_model.bias'] = torch.tensor([0, 0], dtype=torch.float32)
    state_dict['y_model.weight'] = torch.tensor([[0.5, 0, 0.5, 0, 0.5]], dtype=torch.float32)
    state_dict['y_model.bias'] = torch.tensor([0], dtype=torch.float32)
    trainer.load_state_dict(state_dict)


def do_exp():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='simulated_IRM_toyModel_v5_newData_v2')
    # TODO, other experiments
    parser.add_argument('--lambda_ci', type=float, default=0.05)
    parser.add_argument('--lambda_r', type=float, default=0.3)
    parser.add_argument('--dataset_size', type=int, default=128000)
    parser.add_argument('--dataset_group', type=int, default=1)  # using different subgroup in generating dataset
    parser.add_argument('--output_path', type=str, default='./result/')
    parser.add_argument('--num_epochs', type=int, default=10)  # using different subgroup in generating dataset
    args = parser.parse_args()
    output_path = args.output_path + args.exp_name + '/' + str(args.dataset_size) + '_' \
                  + str(args.dataset_group) + '_' + str(args.lambda_ci) + '_' + str(args.lambda_r) + '/'
    ic(output_path)

    from data import get_simulatedDataset_IRM
    dataset = get_simulatedDataset_IRM(n=args.dataset_size, group_number=args.dataset_group)

    # split the huggingface dataset into train and test
    dataset = dataset.train_test_split(test_size=0.2)

    trainer = Trainer(dataset, output_dir=output_path, lambda_ci=args.lambda_ci, lambda_R=args.lambda_r)
    trainer.train_epochs(args.num_epochs)

if __name__ == '__main__':
    do_exp()