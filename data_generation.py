import numpy as np
from scipy.stats import invwishart, dirichlet
import argparse

seed = 0
numpy_randomGen = np.random.default_rng(seed)

def fiducial_values(alpha, mu_0, lambda_, S, nu):
    Pi=dirichlet.rvs(alpha, random_state=numpy_randomGen)
    Sigmas=[]
    mus=[]
    for i in range(len(alpha)):
        Sigmas.append(invwishart.rvs(nu, S, random_state=numpy_randomGen))
        mus.append(numpy_randomGen.multivariate_normal(mu_0, 1/lambda_*Sigmas[-1]))
    return Pi, mus, Sigmas

def generate_data(mus, Sigmas, K=3, assignements=[0]*300+[1]*300+[2]*300):   
    """
    Given the number of clusters and the number of points per clusters, as well as the
    fiducial values of the means and covariance matices, it returns a dataset and its labels
    """    
    
    for i in range(len(assignements)):
        samples=numpy_randomGen.multivariate_normal(mus[assignements[i]], Sigmas[assignements[i]])
        if i!=0:     
            Data=np.vstack([Data,samples])
        else:
            Data=samples
    return Data

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic data.')
    parser.add_argument('--alpha', nargs='+', type=float, default=[1, 2, 1.5], help='Dirichlet distribution parameters')
    parser.add_argument('--mu_0', nargs='+', type=float, default=[0, 0], help='Mean of the multivariate normal distribution')
    parser.add_argument('--lambda_', type=float, default=0.025, help='Precision parameter for the multivariate normal distribution')
    parser.add_argument('--S', type=float, default=1, help='Scale matrix for the inverse Wishart distribution (identity matrix scale)')
    parser.add_argument('--nu', type=int, default=5, help='Degrees of freedom for the inverse Wishart distribution')
    parser.add_argument('--datafile', type=str, default='data/data', help='Path to save the data file')
    args = parser.parse_args()

    alpha = args.alpha
    mu_0 = args.mu_0
    lambda_ = args.lambda_
    S = np.eye(len(mu_0)) * args.S
    nu = args.nu
    Pi, mus, Sigmas = fiducial_values(alpha, mu_0, lambda_, S, nu)
    assignements = numpy_randomGen.choice(len(alpha), 1000, p=Pi[0])
    Data = generate_data(mus, Sigmas, assignements=assignements)
    np.save(args.datafile,np.hstack([Data,assignements.reshape(-1,1)]), allow_pickle=True)