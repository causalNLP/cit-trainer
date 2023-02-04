from causallearn.utils.KCI.KCI import KCI_CInd
from calc_CI import KCI_CInd_Gradient
from icecream import ic
import numpy as np
import torch

def run_CIT(X, Y, Z, tester_type = 1):
    # This function is used to generate the causal effect of Y on X given Z
    # Output: the p-value of Y \bot X | Z
    if type(X) != torch.Tensor:
        X = torch.tensor(X)
    if type(Y) != torch.Tensor:
        Y = torch.tensor(Y)
    if type(Z) != torch.Tensor:
        Z = torch.tensor(Z)
    if torch.cuda.is_available():
        X = X.cuda()
        Y = Y.cuda()
        Z = Z.cuda()

    if (tester_type==0):
        ci_tester = KCI_CInd()
    else:
        ci_tester = KCI_CInd_Gradient()
    return ci_tester.compute_pvalue(X, Y, Z)
