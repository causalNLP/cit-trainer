import torch
import math
import numpy as np
from numpy import sqrt
from torch import tensor
from numpy.linalg import eigh, eigvalsh
from numpy import shape, ndarray
from scipy import stats
from icecream import ic
from sklearn.gaussian_process import GaussianProcessRegressor
from numpy.linalg import pinv
from numpy import eye
from torch.linalg import pinv as pinv_torch
from torch import eye as eye_torch

#from causallearn.utils.KCI.GaussianKernel import GaussianKernel
from causallearn.utils.KCI.Kernel import Kernel
#from causallearn.utils.KCI.LinearKernel import LinearKernel
#from causallearn.utils.KCI.PolynomialKernel import PolynomialKernel
from numpy import exp, median, shape, sqrt, ndarray
from numpy.random import permutation
from scipy.spatial.distance import cdist, pdist, squareform

def center_kernel_matrix(K):
    """
    Centers the kernel matrix via a centering matrix H=I-1/n and returns HKH
    [Updated @Haoyue 06/24/2022]
    equivalent to:
        H = eye(n) - 1.0 / n
        return H.dot(K.dot(H))
    since n is always big, we can save time on the dot product by plugging H into dot and expand as sum.
    time complexity is reduced from O(n^3) (matrix dot) to O(n^2) (traverse each element).
    Also, consider the fact that here K (both Kx and Ky) are symmetric matrices, so K_colsums == K_rowsums
    """
    # assert np.all(K == K.T), 'K should be symmetric'
    n = shape(K)[0]
    K_colsums = K.sum(axis=0)
    K_allsum = K_colsums.sum()
    return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n ** 2)

def center_kernel_matrix_regression(K, Kz, epsilon):
    """
    Centers the kernel matrix via a centering matrix R=I-Kz(Kz+\epsilonI)^{-1} and returns RKR
    """
    n = shape(K)[0]
    eye = eye_torch(n)
    if torch.cuda.is_available():
        eye = eye.cuda()
    try:
        Rz_torch = epsilon * pinv_torch(Kz + epsilon * eye)
    except:
        ic(epsilon, Kz, epsilon)
        ic(eye_torch(n))
        raise Exception('pinv_torch failed')
    return Rz_torch.mm(K.mm(Rz_torch)), Rz_torch


class GaussianKernel(Kernel):
    def __init__(self, width=1.0):
        Kernel.__init__(self)
        self.width: float = 1.0 / width ** 2

    def kernel(self, X:tensor):
        """
        Computes the Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)=exp(-0.5* ||x-y||**2 *self.width)
        """
        #ic(squareform(pdist(X, 'sqeuclidean')))
        pdist_gradient = torch.nn.functional.pdist(X, p=2)
        pdist_gradient = pdist_gradient * pdist_gradient
        pdist_padding = torch.tensor([0.0], requires_grad = True)
        if (torch.cuda.is_available()):
            pdist_padding = pdist_padding.cuda()
        sq_dists_gradient = None
        idx = 0
        idx_line = 0
        n = X.shape[0]

        # TODO: To be improved
        for i in range(X.shape[0]):
            if (idx_line>0):
                #ic(idx_line, sq_dists_gradient)
                #ic(sq_dists_gradient[:, idx_line].shape, pdist_padding.shape, pdist_gradient[idx: idx+n-idx_line-1].shape)
                sq_line_gradient = torch.cat((sq_dists_gradient[:, idx_line], pdist_padding, pdist_gradient[idx: idx+n-idx_line-1]), 0)
            else:
                sq_line_gradient = torch.cat((pdist_padding, pdist_gradient[idx: idx+n-idx_line-1]), 0)

            sq_line_gradient = sq_line_gradient.reshape((1, n))
            #ic(sq_line_gradient.shape)
            idx = idx+n-idx_line-1
            idx_line += 1

            if (i == 0):
                sq_dists_gradient = sq_line_gradient
            else:
                sq_dists_gradient = torch.cat((sq_dists_gradient, sq_line_gradient), 0)
            #ic(sq_dists_gradient.shape)


        #ic(sq_dists_gradient)
        #ic(squareform(pdist(X, 'sqeuclidean')))
        K_gredient = torch.exp(-0.5 * sq_dists_gradient * self.width)
        #print("")
        #sq_dists = squareform(pdist(X, 'sqeuclidean'))
        #K = exp(-0.5 * sq_dists * self.width)
        #ic(K, K_gredient)
        return K_gredient

    # use empirical kernel width instead of the median
    def set_width_empirical_kci(self, X: ndarray):
        n = shape(X)[0]
        if n < 200:
            width = 1.2
        elif n < 1200:
            width = 0.7
        else:
            width = 0.4
        theta = 1.0 / (width ** 2)
        self.width = theta / X.shape[1]

class KCI_CInd_Gradient(object):
    """
    Python implementation of Kernel-based Conditional Independence (KCI) test. Conditional version.
    The original Matlab implementation can be found in http://people.tuebingen.mpg.de/kzhang/KCI-test.zip

    References
    ----------
    [1] K. Zhang, J. Peters, D. Janzing, and B. Schölkopf, "A kernel-based conditional independence test and application in causal discovery," In UAI 2011.
    """

    def __init__(self, kernelX='Gaussian', kernelY='Gaussian', kernelZ='Gaussian', nullss=5000, est_width='empirical',
                 use_gp=False, approx=True, polyd=2, kwidthx=None, kwidthy=None, kwidthz=None):
        """
        Construct the KCI_CInd model.
        Parameters
        ----------
        kernelX: kernel function for input data x
            'Gaussian': Gaussian kernel
            'Polynomial': Polynomial kernel
            'Linear': Linear kernel
        kernelY: kernel function for input data y
        kernelZ: kernel function for input data z (conditional variable)
        est_width: set kernel width for Gaussian kernels
            'empirical': set kernel width using empirical rules
            'median': set kernel width using the median trick
            'manual': set by users
        null_ss: sample size in simulating the null distribution
        use_gp: whether use gaussian process to determine kernel width for z
        approx: whether to use gamma approximation (default=True)
        polyd: polynomial kernel degrees (default=1)
        kwidthx: kernel width for data x (standard deviation sigma, default None)
        kwidthy: kernel width for data y (standard deviation sigma)
        kwidthz: kernel width for data z (standard deviation sigma)
        """
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.kernelZ = kernelZ
        self.est_width = est_width
        self.polyd = polyd
        self.kwidthx = kwidthx
        self.kwidthy = kwidthy
        self.kwidthz = kwidthz
        self.nullss = nullss
        self.epsilon_x = 0.01
        self.epsilon_y = 0.01
        self.use_gp = use_gp
        self.thresh = 1e-5
        self.approx = approx
        self.epsilon = 1e-5

    def compute_pvalue(self, data_x=None, data_y=None, data_z=None):
        """
        Main function: compute the p value and return it together with the test statistic
        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)
        data_z: input data for z (nxd3 array)

        Returns
        _________
        pvalue: p value
        test_stat: test statistic
        """
        #ic(data_x, data_y, data_z)
        #data_x = torch.tensor(data_x, requires_grad = True)
        #data_y = torch.tensor(data_y, requires_grad = True)
        #data_z = torch.tensor(data_z, requires_grad = True)
        #ic(self.kwidthx, self.kwidthy, self.kwidthz)
        Kx, Ky, Kzx, Kzy = self.kernel_matrix(data_x, data_y, data_z)
        # All kernel matrix need gradient

        #ic(Kx.shape, Ky.shape, Kzx.shape, Kzy.shape)
        #ic(Kx, Ky, Kzx, Kzy)
        test_stat, KxR, KyR = self.KCI_V_statistic(Kx, Ky, Kzx, Kzy)
        # test_stat need gradient

        KxR = KxR.cpu().detach().numpy()
        KyR = KyR.cpu().detach().numpy()
        #ic(test_stat.shape, KxR.shape, KyR.shape)
        uu_prod, size_u = self.get_uuprod(KxR, KyR)
        loss = None
        #ic(uu_prod.shape, size_u)
        if self.approx:
            k_appr, theta_appr, mean_appr, var_appr = self.get_kappa(uu_prod)
            pvalue = 1 - stats.gamma.cdf(test_stat.cpu().detach().numpy(), k_appr, 0, theta_appr)
            std_appr = math.sqrt(var_appr)
            loss = (test_stat-mean_appr-std_appr)/(std_appr+self.epsilon)
            if (loss<0):
                loss = 0
            # loss = loss*loss
        else:
            null_samples = self.null_sample_spectral(uu_prod, size_u, Kx.shape[0])
            pvalue = sum(null_samples > test_stat) / float(self.nullss)

        return pvalue, test_stat, loss

    def kernel_matrix(self, data_x, data_y, data_z):
        """
        Compute kernel matrix for data x, data y, and data_z
        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)
        data_z: input data for z (nxd3 array)

        Returns
        _________
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)
        Kzx: centering kernel matrix for data_x (nxn)
        kzy: centering kernel matrix for data_y (nxn)
        """
        # normalize the data
        #ic(data_x,data_y,data_z)
        #print(torch.mean(data_x, 1, True))
        def zscore_tensor(data):
            data = data - torch.mean(data, 0)
            # convert data to numpy array
            data_array = data.cpu().detach().numpy()
            #ic(np.maximum(np.std(data_array, axis = 0), np.array([1e-12]*len(data_array[0]))))
            #ic(data_array.shape, np.std(data_array, axis = 0).shape, np.array([1e-12]*len(data_array[0])).shape)
            data_var = np.maximum(np.std(data_array, axis = 0), np.array([1e-12]*data_array.shape[1])).reshape((1,-1))
            if (torch.cuda.is_available()):
                data_var = torch.tensor(data_var).cuda()
            #ic(data, data_var)
            try:
                data = data/data_var
            except:
                ic(data, data_var)
                raise Exception("Error in zscore_tensor")
            return data

        data_x = zscore_tensor(data_x)
        data_y = zscore_tensor(data_y)
        data_z = zscore_tensor(data_z)
        #ic(data_x[0])


        # concatenate x and z
        data_x = torch.cat((data_x, 0.5 * data_z), 1)

        #data_x = data_x.detach().numpy()
        #data_y = data_y.detach().numpy()
        #data_z = data_z.detach().numpy()

        if self.kernelX == 'Gaussian':
            kernelX = GaussianKernel()
            if self.est_width == 'empirical':
                kernelX.set_width_empirical_kci(data_x)
            else:
                raise Exception('Undefined kernel width estimation method')
        else:
            raise Exception('Undefined kernel function')

        if self.kernelY == 'Gaussian':
            kernelY = GaussianKernel()
            if self.est_width == 'empirical':
                kernelY.set_width_empirical_kci(data_y)
            else:
                raise Exception('Undefined kernel width estimation method')
        else:
            raise Exception('Undefined kernel function')

        Kx = kernelX.kernel(data_x)
        Ky = kernelY.kernel(data_y)

        #Kx = Kx.detach().numpy()
        #Ky = Ky.detach().numpy()

        # centering kernel matrix
        Kx = center_kernel_matrix(Kx)
        Ky = center_kernel_matrix(Ky)
        #ic(Kx, Ky)
        if self.kernelZ == 'Gaussian':
            if not self.use_gp:
                kernelZ = GaussianKernel()
                if self.est_width == 'empirical':
                    kernelZ.set_width_empirical_kci(data_z)
                Kzx = kernelZ.kernel(data_z)
                Kzy = Kzx
            else:
                raise Exception('Undefined kernel estimation method')
        else:
            raise Exception('Undefined kernel function')

        #Kzx = Kzx.detach().numpy()
        #Kzy = Kzy.detach().numpy()
        return Kx, Ky, Kzx, Kzy

    def KCI_V_statistic(self, Kx, Ky, Kzx, Kzy):
        """
        Compute V test statistic from kernel matrices Kx and Ky
        Parameters
        ----------
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)
        Kzx: centering kernel matrix for data_x (nxn)
        kzy: centering kernel matrix for data_y (nxn)

        Returns
        _________
        Vstat: KCI v statistics
        KxR: centralized kernel matrix for data_x (nxn)
        KyR: centralized kernel matrix for data_y (nxn)
        """
        KxR = center_kernel_matrix_regression(Kx, Kzx, self.epsilon_x)[0]
        KyR = center_kernel_matrix_regression(Ky, Kzy, self.epsilon_y)[0]
        Vstat = torch.sum(KxR * KyR)
        return Vstat, KxR, KyR

    def get_uuprod(self, Kx, Ky):
        """
        Compute eigenvalues for null distribution estimation

        Parameters
        ----------
        Kx: centralized kernel matrix for data_x (nxn)
        Ky: centralized kernel matrix for data_y (nxn)

        Returns
        _________
        uu_prod: product of the eigenvectors of Kx and Ky
        size_u: number of producted eigenvectors

        """
        wx, vx = eigh(0.5 * (Kx + Kx.T))
        wy, vy = eigh(0.5 * (Ky + Ky.T))
        idx = np.argsort(-wx)
        idy = np.argsort(-wy)
        wx = wx[idx]
        vx = vx[:, idx]
        wy = wy[idy]
        vy = vy[:, idy]
        vx = vx[:, wx > np.max(wx) * self.thresh]
        wx = wx[wx > np.max(wx) * self.thresh]
        vy = vy[:, wy > np.max(wy) * self.thresh]
        wy = wy[wy > np.max(wy) * self.thresh]
        vx = vx.dot(np.diag(np.sqrt(wx)))
        vy = vy.dot(np.diag(np.sqrt(wy)))

        # calculate their product
        T = Kx.shape[0]
        num_eigx = vx.shape[1]
        num_eigy = vy.shape[1]
        size_u = num_eigx * num_eigy
        uu = np.zeros((T, size_u))
        for i in range(0, num_eigx):
            for j in range(0, num_eigy):
                uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

        if size_u > T:
            uu_prod = uu.dot(uu.T)
        else:
            uu_prod = uu.T.dot(uu)

        return uu_prod, size_u

    def null_sample_spectral(self, uu_prod, size_u, T):
        """
        Simulate data from null distribution

        Parameters
        ----------
        uu_prod: product of the eigenvectors of Kx and Ky
        size_u: number of producted eigenvectors
        T: sample size

        Returns
        _________
        null_dstr: samples from the null distribution

        """
        eig_uu = eigvalsh(uu_prod)
        eig_uu = -np.sort(-eig_uu)
        eig_uu = eig_uu[0:np.min((T, size_u))]
        eig_uu = eig_uu[eig_uu > np.max(eig_uu) * self.thresh]

        f_rand = np.random.chisquare(1, (eig_uu.shape[0], self.nullss))
        null_dstr = eig_uu.T.dot(f_rand)
        return null_dstr

    def get_kappa(self, uu_prod):
        """
        Get parameters for the approximated gamma distribution
        Parameters
        ----------
        uu_prod: product of the eigenvectors of Kx and Ky

        Returns
        ----------
        k_appr, theta_appr: approximated parameters of the gamma distribution

        """
        mean_appr = np.trace(uu_prod)
        var_appr = 2 * np.trace(uu_prod.dot(uu_prod))
        #ic(mean_appr, var_appr)
        #ic(uu_prod)
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        #ic(k_appr, theta_appr)
        return k_appr, theta_appr, mean_appr, var_appr
