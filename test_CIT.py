import numpy as np
#from generate_dataset import simple_SCM, noise

from icecream import ic
from causallearn.utils.cit import CIT
from tqdm import trange
import torch

from CIT_warpper import run_CIT
from sample_datapoint import generate_dataset

np.random.seed(42)
def test_CIT_gradient():
    X, Y, Z = generate_dataset(is_chain = 0)
    X = torch.tensor(X, requires_grad = True)
    Y = torch.tensor(Y, requires_grad = True)
    Z = torch.tensor(Z, requires_grad = True)
    ic(run_CIT(X,Y,Z))
    for i in range(1000):
        X = torch.tensor(X, requires_grad = True)
        Y = torch.tensor(Y, requires_grad = True)
        Z = torch.tensor(Z, requires_grad = True)
        p_value, stat, loss = run_CIT(X, Y, Z, tester_type = 1)
        #if (loss<0):
        #    loss = torch.tensor(0, requires_grad = True)
        #loss = loss*loss
        ic(p_value, stat, loss)
        #ic(X, X.detach())
        #ic(X, X.grad)
        if (type(loss)==int):
            break
        loss.backward()
        X = X.detach().numpy()-X.grad.numpy()*1e-3
        Y = Y.detach().numpy()-Y.grad.numpy()*1e-3
        Z = Z.detach().numpy()-Z.grad.numpy()*1e-3
        print("")
    #X, Y, Z = generate_dataset(is_chain = 1)
    #ic(run_CIT(X,Y,Z, tester_type = 1))

def test_CIT_trainer():
    from data import get_dataset
    dataset = get_dataset()
    from trainer import Trainer
    trainer = Trainer(dataset, ['X_indY', 'X_indZ', 'X_YnZ'],
                      [['X_indY', 'Z'], ['X_YnZ', 'Z'], ['X_indZ', 'Y'], ['X_YnZ', 'Y']],
                      [['X_indY', 'X_YnZ', 'Z'], ['X_indZ', 'X_YnZ', 'None']])
    trainer.train_epochs(10, 8, 512)
    loss, CI_loss, pvalue, pred, target = trainer.evaluate(8, 128)
    ic(loss, CI_loss, pvalue)

if __name__ == '__main__':
    test_CIT_gradient()