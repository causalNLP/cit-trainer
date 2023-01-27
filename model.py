from icecream import ic
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss

input_dim = 10
output_dim = 2

# Input: 32*3 dim
# Output: 32 dim
class ToyEstimator(nn.Module):
    def __init__(self, input_dim = input_dim, output_dim = output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 120)
        self.fc2 = nn.Linear(120, output_dim)

    def forward(self, x):
        #ic(x.shape)
        #ic("################################")
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ToyRegresser(nn.Module):
    def __init__(self, sample_dataDict):
        # The sample_dataDict is a dictionary of a sample data
        # e.g. {"x_1":[0, 0], "x_2":[0, 0], "env":[0]}
        super().__init__()
        self.fc1_dict = {}
        self.lambda_dict = {}
        for i in sample_dataDict.keys():
            self.fc1_dict[i] = nn.Linear(len(sample_dataDict[i]), 120)
            self.lambda_dict[i] = nn.Parameter(torch.tensor(1.0))
            if (torch.cuda.is_available()):
                self.lambda_dict[i] = self.lambda_dict[i].cuda()
                self.fc1_dict[i] = self.fc1_dict[i].cuda()

        self.fc2 = nn.Linear(120, 1)
        self.loss = MSELoss()

    def forward(self, x, output = None):
        ans = None
        for i in x.keys():
            if (ans is None):
                ans = self.fc1_dict[i](x[i])
            else:
                ans += self.fc1_dict[i](x[i])
        ans = self.fc2(ans)
        if (output == None):
            return ans
        else:
            output = torch.tensor(output, dtype = torch.float32)
            if (torch.cuda.is_available()):
                output = output.cuda()
            return ans, self.loss(ans, output)

#TODO
class BertEstimator(nn.Module):
    def __init__(self, class_num = 2):
        super().__init__()
        pass

    def forward(self, x, output = None):
        pass

class BertRegresser(nn.Module):
    def __init__(self, class_num = 2):
        super().__init__()
        pass

    def forward(self, x, output = None):
        pass

if __name__ == '__main__':
    data = torch.ones(1, input_dim).cuda()
    estimator = ToyEstimator()
    regresser = ToyRegresser({"x_1":[0, 0], "x_2":[0, 0], "env":[0]})
    estimator.cuda()
    regresser.cuda()
    ic(data)
    ic(estimator(data))
    medium_data = estimator(data)
    ic(regresser({"x_1":medium_data, "x_2":medium_data, "env":torch.ones(1, 1).cuda()}, output = 1))


