#from datasets import load_from_disk, Dataset
from icecream import ic
import numpy as np
from tqdm import trange

np.random.seed(42)
from numpy.linalg import norm

class simple_SCM(object):
    def __init__(self, input_dim = 1, output_dim = 1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.f_function = np.random.randint(4, size = (input_dim, output_dim))
        self.b_constant = np.random.random_sample((input_dim, output_dim))*2+0.5

    def relation_function(self, input, func_number):
        input = np.array(input)
        #return input
        if (func_number==0):
            return input*input
        elif (func_number==1):
            return input*input*input
        elif (func_number==2):
            return np.tanh(input)
        else:
            return np.sinc(input)

    def sampling(self, input_vec):
        output_vec = np.array([0]*self.output_dim)
        for i in range(len(input_vec)):
            for j in range(len(output_vec)):
                output_vec[j] += self.relation_function(input_vec[i], self.f_function[i][j])*self.b_constant[i][j]
        return output_vec

    def __str__(self):
        function_name = ["x^2", "x^3", "tanh(x)", "sinc(x)"]
        output = ""
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                output += str(self.b_constant[i][j]) + " * " + function_name[self.f_function[i][j]]
                if (j!=self.output_dim-1):
                    output += " + "
            if (i!=self.input_dim-1):
                output += "\n"
        return output


class noise(object):
    def __init__(self, output_dim = 1):
        self.output_dim = output_dim
        self.eps_function = np.random.randint(2, size = (output_dim))
        self.sigma_constant = np.random.random_sample((output_dim))*2+1

    def noise_function(self, func_number):
        # return np.random.normal()
        if (func_number==0):
            return np.random.normal()
        else:
            return np.random.random_sample()-0.5

    def generate(self):
        output_vec = []
        for i in range(self.output_dim):
            output_vec.append(self.sigma_constant[i]*self.noise_function(self.eps_function[i]))
        return np.array(output_vec)

    def __str__(self):
        function_name = ["normal", "uniform"]
        output = ""
        for i in range(self.output_dim):
            output += str(self.sigma_constant[i]) + " * " + function_name[self.eps_function[i]]
            if (i!=self.output_dim-1):
                output += " | "
        return output

"""
def generate_causal_dataset(size, dim = 32):
    Y_noise, Z_noise = noise(dim), noise(dim)
    XindY_noise, XindZ_noise, XYnZ_noise = noise(dim), noise(dim), noise(dim)
    Z_YnZ_SCM, Z_indY_SCM, indZ_Y_SCM, YnZ_Y_SCM = simple_SCM(dim, dim), simple_SCM(dim, dim), simple_SCM(dim, dim), simple_SCM(dim, dim)
    dataset = {
        "X_indY":[],
        "X_indZ":[],
        "X_YnZ":[],
        "Y":[],
        "Z":[],
    }
    for i in trange(size):
        Z = Z_noise.generate()
        #Z = Z/norm(Z)
        X_indY = XindY_noise.generate()+Z_indY_SCM.sampling(Z)
        #X_indY = X_indY/norm(X_indY)
        X_YnZ = XYnZ_noise.generate()+Z_YnZ_SCM.sampling(Z)
        #X_YnZ = X_YnZ/norm(X_YnZ)
        X_indZ = XindZ_noise.generate()
        #X_indZ = X_indZ/norm(X_indZ)
        Y = Y_noise.generate() + indZ_Y_SCM.sampling(X_indZ) + YnZ_Y_SCM.sampling(X_YnZ)
        #Y = Y/norm(Y)
        dataset['Z'].append(Z)
        dataset['Y'].append(Y)
        dataset['X_indY'].append(X_indY)
        dataset['X_indZ'].append(X_indZ)
        dataset['X_YnZ'].append(X_YnZ)
    #ic(dataset)
    return dataset

def generate_anticausal_dataset(size, dim = 32):
    Y_noise, Z_noise = noise(dim), noise(dim)
    XindY_noise, XindZ_noise, XYnZ_noise = noise(dim), noise(dim), noise(dim)
    Z_YnZ_SCM, Z_indY_SCM, Y_indZ_SCM, Y_YnZ_SCM = simple_SCM(dim, dim), simple_SCM(dim, dim), simple_SCM(dim, dim), simple_SCM(dim, dim)
    dataset = {
        "X_indY":[],
        "X_indZ":[],
        "X_YnZ":[],
        "Y":[],
        "Z":[],
    }
    for i in trange(size):
        Z = Z_noise.generate()
        Y = Y_noise.generate()

        X_indY = XindY_noise.generate()+Z_indY_SCM.sampling(Z)
        X_YnZ = XYnZ_noise.generate()+Z_YnZ_SCM.sampling(Z)+Y_YnZ_SCM.sampling(Y)
        X_indZ = XindZ_noise.generate()+Y_indZ_SCM.sampling(Y)
        dataset['Z'].append(Z)
        dataset['Y'].append(Y)
        dataset['X_indY'].append(X_indY)
        dataset['X_indZ'].append(X_indZ)
        dataset['X_YnZ'].append(X_YnZ)
    #ic(dataset[0])
    return dataset

def resampling_dataset(dataset, ratio):
    def cos_sim(x,y):
        if (len(x)>len(y)):
            x,y = y,x
        if (len(x)<len(y)):
            y = y[:len(x)]
        from numpy import dot
        from numpy.linalg import norm
        result = dot(x, y) / (norm(x) * norm(y))
        return result

    n = len(dataset['Z'])
    nn = int(n*ratio)
    similarity_list = []
    for i in range(n):
        similarity_list.append([cos_sim(dataset['Y'][i], dataset['Z'][i]),i])
    similarity_list = sorted(similarity_list)
    ic(similarity_list[:10])
    ic(similarity_list[-10:])
    lowsim_dataset, highsim_dataset = {}, {}
    keys = ["Y", "Z", "X_indY", "X_indZ", "X_YnZ"]
    for i in range(nn):
        for key in keys:
            if (lowsim_dataset.get(key)==None):
                lowsim_dataset[key] = []
            lowsim_dataset[key].append(dataset[key][similarity_list[i][1]])
    for i in range(n-nn, n):
        for key in keys:
            if (highsim_dataset.get(key)==None):
                highsim_dataset[key] = []
            highsim_dataset[key].append(dataset[key][similarity_list[i][1]])

    return lowsim_dataset, highsim_dataset
"""
def generate_causal_dataset(size, dim = 32):
    dataset = {
        "hidden_X_indY": [],
        "hidden_X_indZ": [],
        "hidden_X_YnZ": [],
        "X_indY": [],
        "X_indZ": [],
        "X_YnZ": [],
        "Y": [],
        "Z": [],
    }

    from icecream import ic
    Y_noise, Z_noise = noise(1), noise(1)
    ic(Y_noise, Z_noise)
    XindY_noise, XindZ_noise, XYnZ_noise = noise(1), noise(1), noise(1)
    ic(XindY_noise, XindZ_noise, XYnZ_noise)
    Z_YnZ_SCM, Z_indY_SCM, indZ_Y_SCM, YnZ_Y_SCM = simple_SCM(1, 1), simple_SCM(1, 1), simple_SCM(1, 1), simple_SCM(1, 1)
    ic(Z_YnZ_SCM, Z_indY_SCM, indZ_Y_SCM, YnZ_Y_SCM)
    XindY_noise_highdim, XindZ_noise_highdim, XYnZ_noise_highdim = noise(dim), noise(dim), noise(dim)#可能需要?
    ic(XindY_noise_highdim, XindZ_noise_highdim, XYnZ_noise_highdim)
    XindY_SCM_highdim, XindZ_SCM_highdim, XYnZ_SCM_highdim = simple_SCM(1, dim), simple_SCM(1, dim), simple_SCM(1, dim)
    ic(XindY_SCM_highdim, XindZ_SCM_highdim, XYnZ_SCM_highdim)

    for i in trange(size):
        Z = Z_noise.generate()
        hidden_X_indY = XindY_noise.generate()+Z_indY_SCM.sampling(Z)
        hidden_X_YnZ = XYnZ_noise.generate()+Z_YnZ_SCM.sampling(Z)
        hidden_X_indZ = XindZ_noise.generate()
        Y = Y_noise.generate() + indZ_Y_SCM.sampling(hidden_X_indZ) + YnZ_Y_SCM.sampling(hidden_X_YnZ)

        X_indY = XindY_noise_highdim.generate()+XindY_SCM_highdim.sampling(hidden_X_indY)
        X_indZ = XindZ_noise_highdim.generate()+XindZ_SCM_highdim.sampling(hidden_X_indZ)
        X_YnZ = XYnZ_noise_highdim.generate()+XYnZ_SCM_highdim.sampling(hidden_X_YnZ)

        dataset['Z'].append(Z)
        dataset['hidden_X_indY'].append(hidden_X_indY)
        dataset['hidden_X_YnZ'].append(hidden_X_YnZ)
        dataset['hidden_X_indZ'].append(hidden_X_indZ)
        dataset['Y'].append(Y)
        dataset['X_indY'].append(X_indY)
        dataset['X_indZ'].append(X_indZ)
        dataset['X_YnZ'].append(X_YnZ)

    return dataset

def generate_anticausal_dataset(size, dim = 32):

    dataset = {
        "hidden_X_indY": [],
        "hidden_X_indZ": [],
        "hidden_X_YnZ": [],
        "X_indY": [],
        "X_indZ": [],
        "X_YnZ": [],
        "Y": [],
        "Z": [],
    }

    Y_noise, Z_noise = noise(1), noise(1)
    XindY_noise, XindZ_noise, XYnZ_noise = noise(1), noise(1), noise(1)
    Z_YnZ_SCM, Z_indY_SCM, Y_YnZ_SCM, Y_indZ_SCM = simple_SCM(1, 1), simple_SCM(1, 1), simple_SCM(1, 1), simple_SCM(1, 1)
    XindY_noise_highdim, XindZ_noise_highdim, XYnZ_noise_highdim = noise(dim), noise(dim), noise(dim)#可能需要?
    XindY_SCM_highdim, XindZ_SCM_highdim, XYnZ_SCM_highdim = simple_SCM(1, dim), simple_SCM(1, dim), simple_SCM(1, dim)

    for i in trange(size):

        Z = Z_noise.generate()
        Y = Y_noise.generate()
        hidden_X_indY = XindY_noise.generate() + Z_indY_SCM.sampling(Z)
        hidden_X_YnZ = XYnZ_noise.generate() + Z_YnZ_SCM.sampling(Z) + Y_YnZ_SCM.sampling(Y)
        hidden_X_indZ = XindZ_noise.generate() + Y_indZ_SCM.sampling(Y)

        X_indY = XindY_noise_highdim.generate()+XindY_SCM_highdim.sampling(hidden_X_indY)
        X_indZ = XindZ_noise_highdim.generate()+XindZ_SCM_highdim.sampling(hidden_X_indZ)
        X_YnZ = XYnZ_noise_highdim.generate()+XYnZ_SCM_highdim.sampling(hidden_X_YnZ)

        dataset['Z'].append(Z)
        dataset['hidden_X_indY'].append(hidden_X_indY)
        dataset['hidden_X_YnZ'].append(hidden_X_YnZ)
        dataset['hidden_X_indZ'].append(hidden_X_indZ)
        dataset['Y'].append(Y)
        dataset['X_indY'].append(X_indY)
        dataset['X_indZ'].append(X_indZ)
        dataset['X_YnZ'].append(X_YnZ)

    return dataset

def resampling_dataset(dataset):
    normalized_Y = np.array(dataset['Y'])/np.average(dataset['Y'])
    normalized_Z = np.array(dataset['Z'])/np.average(dataset['Z'])
    dot_YZ = normalized_Y*normalized_Z
    idx_dotYZ = []
    n = len(dot_YZ)
    for i in range(n):
        idx_dotYZ.append([dot_YZ[i],i])
    idx_dotYZ = sorted(idx_dotYZ)
    keys = ['hidden_X_indY', 'hidden_X_indZ', 'hidden_X_YnZ', 'X_indY', 'X_indZ', 'X_YnZ', 'Y', 'Z']
    dataset1 = {
        "hidden_X_indY": [],
        "hidden_X_indZ": [],
        "hidden_X_YnZ": [],
        "X_indY": [],
        "X_indZ": [],
        "X_YnZ": [],
        "Y": [],
        "Z": [],
    }
    dataset2 = {
        "hidden_X_indY": [],
        "hidden_X_indZ": [],
        "hidden_X_YnZ": [],
        "X_indY": [],
        "X_indZ": [],
        "X_YnZ": [],
        "Y": [],
        "Z": [],
    }
    for i in range(n):
        if (np.random.random()<(i+1)/(n+1)):
            for key in keys:
                dataset1[key].append(dataset[key][i])
        else:
            for key in keys:
                dataset2[key].append(dataset[key][i])
    return dataset1, dataset2

def save_dataset(dataset, output_dir):
    a = np.array(dataset)
    np.save(output_dir, a)


if __name__ == '__main__':
    pass