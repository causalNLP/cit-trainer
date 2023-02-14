import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from utils import *

class Trainer(nn.Module):
    def __init__(self, dataset, lambda_ci = 1, lambda_R = 0.3, output_dir = './output/'):
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
        self.optimizer = optim.Adam(self.parameters(), lr=0.1)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.loss_fn = nn.MSELoss()
        if torch.cuda.is_available():
            self.cuda()
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def compute_loss(self, batch_datapoint):
        from CIT_warpper import run_CIT
        # cat the data from list of tensors to a tensor
        batch_datapoint['env'] = torch.cat(batch_datapoint['env']).reshape(-1, 1)
        batch_datapoint['input'] = torch.cat(batch_datapoint['input']).reshape(-1, 2)
        batch_datapoint['y'] = torch.cat(batch_datapoint['y']).reshape(-1, 1)
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
        y = self.y_model(torch.cat([x1, x2, env], dim = 1))
        loss = self.loss_fn(y, batch_datapoint['y'])
        # compute the conditional independence test loss
        #print(x1.shape, x2.shape, batch_datapoint['y'].shape, batch_datapoint['env'].shape)
        #print(run_CIT(x1, x2, batch_datapoint['y']))
        CIT_loss = run_CIT(x1, x2, batch_datapoint['y'])[2] * self.lambda_ci\
                   +run_CIT(batch_datapoint['env'], x2, batch_datapoint['y'])[2] * self.lambda_ci
        CIT_loss = self.lambda_ci * CIT_loss
        # compute the regression loss
        R_loss = self.lambda_R * (self.loss_fn(env, batch_datapoint['env']))
        return loss, CIT_loss, R_loss

    def train_epochs(self, num_epochs = 50):
        loss_list, CIT_loss_list, R_loss_list = [], [], []
        for epoch in range(num_epochs):
            print('epoch: {}'.format(epoch))
            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=256, shuffle=True)
            from tqdm import tqdm
            for batch_datapoint in tqdm(dataloader):
                loss, CIT_loss, R_loss = self.compute_loss(batch_datapoint)
                self.optimizer.zero_grad()
                loss_list.append(loss.item())
                loss += CIT_loss + R_loss
                loss.backward()
                self.optimizer.step()
                if (CIT_loss == 0):
                    CIT_loss_list.append(CIT_loss)
                else:
                    CIT_loss_list.append(CIT_loss.item())
                R_loss_list.append(R_loss.item())
            # print average loss
            self.lr_scheduler.step()
            print('epoch: {}, loss: {}, CIT_loss: {}, R_loss: {}'.format(epoch, np.mean(loss_list), np.mean(CIT_loss_list), np.mean(R_loss_list)))
            if (epoch % 5 == 0):
                self.show_params()
                #self.evalute()
        self.save()
        self.show_params()

    def save(self):
        torch.save(self.state_dict(), self.output_dir + 'model.pt')
        print('model saved to {}'.format(self.output_dir + 'model.pt'))

    def load(self):
        self.load_state_dict(torch.load(self.output_dir + 'model.pt'))
        print('model loaded from {}'.format(self.output_dir + 'model.pt'))

    def show_params(self): # print out the parameters of the model
        for name, param in self.state_dict().items():
            print(name, param)

    def mode_switch(self, mode):
        if mode == 0: # train mode
            self.env_model.train()
            self.x1_model.train()
            self.x2_model.train()
            self.y_model.train()
            self.optimizer = optim.Adam(self.parameters(), lr = 0.1)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        elif mode == 1: # fine tune mode
            self.y_model.train()
            self.env_model.eval()
            self.x1_model.eval()
            self.x2_model.eval()
            self.optimizer = optim.Adam(self.y_model.parameters(), lr = 0.1)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        elif mode == 2: # test mode
            self.env_model.eval()
            self.x1_model.eval()
            self.x2_model.eval()
            self.y_model.eval()


    def fine_tune(self, dataset, num_epochs = 100):
        self.dataset = dataset
        # only train the y_model
        self.mode_switch(1)
        self.train_epochs(num_epochs)
        self.mode_switch(0)

    def evalute(self):
        #self.mode_switch(2)
        dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=256, shuffle=True)
        loss_list, CIT_loss_list, R_loss_list = [], [], []
        for batch_datapoint in dataloader:
            loss, CIT_loss, R_loss = self.compute_loss(batch_datapoint)
            loss_list.append(loss.item())
            CIT_loss_list.append(CIT_loss.item())
            R_loss_list.append(R_loss.item())
        print('test loss: {}, test CIT_loss: {}, test R_loss: {}'.format(np.mean(loss_list), np.mean(CIT_loss_list), np.mean(R_loss_list)))
        #self.mode_switch(0)

if __name__ == '__main__':
    from data import get_simulatedDataset_IRM
    dataset = get_simulatedDataset_IRM(n = 25600, group_number = 1)
    # split the huggingface dataset into train and test
    dataset = dataset.train_test_split(test_size=0.3)
    # train the model
    trainer = Trainer(dataset)
    trainer.train_epochs()
    trainer.evalute()
    OOD_dataset = get_simulatedDataset_IRM(n = 25600, group_number = 2)
    trainer.fine_tune(OOD_dataset)