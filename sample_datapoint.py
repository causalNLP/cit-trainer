import numpy as np
#from generate_dataset import simple_SCM, noise

from icecream import ic
from causallearn.utils.cit import CIT
from tqdm import trange
import torch
from CIT_warpper import run_CIT

# The follow two function is generate a simulated dataset X->Y->Z (is_chain = 1) or X->Y<-Z (is_chain = 0)
# For X->Y->Z, it have the property that X and Z are independent given Y, but X and Z are dependent given None
# For X->Y<-Z, it have the property that X and Z are dependent given Y, but X and Z are independent given None
def generate_datapoint(Z_dim, is_chain = 0):
    if (is_chain == 0):
        mu_x = np.random.random()+1
        mu_z = np.random.random()+1
        Y = np.random.random()+mu_x+mu_z
    else:
        mu_x = np.random.random()+1
        Y = np.random.random()+mu_x
        mu_z = np.random.random()+Y

    X = []
    for i in range(Z_dim):
        X.append((np.random.random() + 0.5) * mu_x)

    Z = []
    for i in range(Z_dim):
        Z.append((np.random.random() + 0.5) * mu_z)

    return X, [Y], Z

def generate_dataset(data_function = generate_datapoint, sample_dim = 32, is_chain = 0, sample_size = 32):
    X, Y, Z = [],[],[]
    for i in range(sample_size):
        x_point, y_point, z_point = data_function(sample_dim, is_chain)
        X.append(x_point)
        Y.append(y_point)
        Z.append(z_point)
    return X, Y, Z

# The follow two function is generate a simulated dataset which have the common OOD problem in IRM
# The causal relationship is X_1->Y->X_2, while X_1 and Y are the effect by the environment
# See https://drive.google.com/file/d/1PHdeO-gMLSgYDXGEwx72sSpCb6H2deXX/view?usp=share_link
def generate_datapoint_IRM(env = 1, d = 2):
    x_1 = torch.randn(d)*0.5 + env
    y = x_1 + env + torch.randn(d)*0.2
    x_2 = y + torch.randn(d)*0.7
    #x_1 = (torch.randn(d) + 1) * env
    #y = x_1 + (torch.randn(d) + 1) * env
    #x_2 = y + torch.randn(d)
    #ic(x_1, x_2, y.sum(0, keepdim=True), env)
    return np.array(x_1), np.array(x_2), np.array(y.sum(0, keepdim=True)), np.array([env])

def generate_dataset_IRM(sample_size = 512):
    x_1, x_2, y, env = [],[],[],[]
    for i in range(sample_size):
        x_1_point, x_2_point, y_point, env_point = generate_datapoint_IRM(env = np.random.random())
        x_1.append(x_1_point)
        x_2.append(x_2_point)
        y.append(y_point)
        env.append(env_point)
    # transform to a numpy array
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    y = np.array(y)
    env = np.array(env)
    ic(x_1.shape, x_2.shape, y.shape, env.shape)
    return x_1, x_2, y, env


def generate_datapoint_ST(causal = True, d = 2):

    pass

def generate_dataset_ST(sample_size = 512, causal = True):
    pass


if __name__ == '__main__':
    x_1, x_2, y, env = generate_dataset_IRM()
    # X_1 \corr env | Y
    ic(run_CIT(x_1, env, y))
    # X_2 \bot env | Y
    ic(run_CIT(x_2, env, y))
    # X_1 \bot X_2 | Y
    ic(run_CIT(x_1, x_2, y))
