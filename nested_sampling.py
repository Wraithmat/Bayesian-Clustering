import numpy as np
from scipy.stats import invwishart, dirichlet, multivariate_normal
from joblib import Parallel, delayed
import argparse
import mpmath as mp

null_Gen = np.random.default_rng(0)

def sample_prior(alpha, mu_0, lambda_, S, nu, N_points, random_generator=null_Gen):
    """
    Returns N_points sample from the prior distribution of the parameters of the model.
    The results are stored in a matrix of shape (N_points, N_clusters*N_dim+N_clusters*N_dim**2+N_clusters+N_dataset), where:
    - the first N_clusters*N_dim columns contain the means of the clusters
    - the next N_clusters*N_dim**2 columns contain the covariance matrices of the clusters
    - the next N_clusters columns contain the mixing coefficients of the clusters
    """
    
    N_clusters=len(alpha)
    N_dim=len(mu_0)
    result=np.empty((N_points, N_clusters*N_dim+N_clusters*N_dim**2+N_clusters), dtype=float)
    result[:,N_clusters*N_dim+N_clusters*N_dim**2:N_clusters*N_dim+N_clusters*N_dim**2+N_clusters]=dirichlet.rvs(alpha, size=N_points, random_state=random_generator)
    
    sample_mus = lambda x: random_generator.multivariate_normal(mu_0, x.reshape((N_dim,N_dim)))
    sample_mus = np.vectorize(sample_mus, signature='(k)->(n)')
    

    for i in range(len(alpha)):
        result[:,N_clusters*N_dim+i*N_dim**2:N_clusters*N_dim+(i+1)*N_dim**2]=(invwishart.rvs(nu, S, size=N_points, random_state=random_generator)).reshape((N_points, N_dim**2))
        result[:,N_dim*i:N_dim*(i+1)]=sample_mus(1/lambda_*result[:,N_clusters*N_dim+i*N_dim**2:N_clusters*N_dim+(i+1)*N_dim**2])
    return result

def compute_likelihood(data, N_clusters, N_dim, params):
    """
    Returns the log likelihood for you dataset.
    """
    pi=params[-N_clusters:]
    mus=params[:N_clusters*N_dim].reshape(N_clusters,N_dim)
    Sigmas=params[N_clusters*N_dim:N_clusters*N_dim+N_clusters*N_dim**2].reshape(N_clusters,N_dim,N_dim)
    probabilities = np.array([multivariate_normal.pdf(data, mean=mus[i], cov=Sigmas[i]) for i in range(len(mus))]).T*pi
    vec_log = lambda x: mp.log(x)
    vec_log = np.vectorize(vec_log)
    log_likelihood = mp.fsum(vec_log([mp.fsum(row) for row in probabilities]))

    return log_likelihood

def _update(L, alpha, mu_0, lambda_, S, nu, N_clusters, N_dim, data, max_rep=600, random_generator=null_Gen):
    L_=-np.inf
    rep=0
    L_tot_exp=0
    while L_<=L:
        rep+=1
        if rep>max_rep:
            print(f'{rep} samples were performed but the likelihood is still smaller')
            raise ValueError('Too many samples needed')
        trial=sample_prior(alpha, mu_0, lambda_, S, nu, 1, random_generator)[0]
        L_=compute_likelihood(data, N_clusters, N_dim, trial)
        L_tot_exp+=mp.exp(L_)
    return L_, trial, L_tot_exp, rep


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--N_live', type=int, default=200, help='Number of points to sample')
    parser.add_argument('--alpha', type=float, nargs='+', default=[1, 2, 1.5, 1.5, 1.5], help='Alpha values for Dirichlet distribution')
    parser.add_argument('--mu_0', type=float, nargs='+', default=[0, 0], help='Mean vector for the normal distribution')
    parser.add_argument('--lambda_', type=float, default=0.025, help='Lambda value')
    parser.add_argument('--S', type=float, nargs='+', default=[1, 0, 0, 1], help='Scale matrix for the inverse Wishart distribution')
    parser.add_argument('--nu', type=int, default=5, help='Degrees of freedom for the inverse Wishart distribution')
    parser.add_argument('--steps', type=int, default=1500, help='Degrees of freedom for the inverse Wishart distribution')
    parser.add_argument('--data', type=str, default='./data/data_new.npy', help='Path to the data file')
    parser.add_argument('--outputfile', type=str, default='./data/evidences/output.txt', help='Output file to write results')
    parser.add_argument('--n_clusters', type=int, default=1, help='Output file to write results')

    args = parser.parse_args()

    vec_exp = lambda x: mp.exp(x)
    vec_exp = np.vectorize(vec_exp)

    vec_log = lambda x: mp.log(x)
    vec_log = np.vectorize(vec_log)
    
    #algorithm paraters
    seed=args.seed
    N=args.N_live
    steps=args.steps
    
    #model parameters
    alpha_ = args.alpha
    mu_0 = np.array(args.mu_0)
    lambda_ = args.lambda_
    S = np.array(args.S).reshape(2, 2)
    nu = args.nu
    N_clusters=args.n_clusters

    assert(len(alpha_)>=N_clusters)
    alpha = alpha_[:N_clusters]

    #data parameters
    data = np.load(args.data)
    outputfile = args.outputfile
    N_data=len(data)

    numpy_randomGen = np.random.default_rng(seed)

    points=sample_prior(alpha, mu_0, lambda_, S, nu, N, numpy_randomGen)
    X_i=[1]
    L_i=[-np.inf]
    Z=0
    Ls=Parallel(n_jobs=-1)(delayed(compute_likelihood)(data[:,:-1],N_clusters,2,i) for i in points)
    Ls=np.array(Ls)
    update_MC=mp.fsum(vec_exp(Ls))
    total_points=N
    MC_est=mp.log(update_MC)-mp.log(total_points)
    print('MC estimate: ', MC_est, 'Log_likelihood: (', np.mean(Ls), '+/-', np.std(Ls), f'), {update_MC}')
    with open(outputfile, 'w') as f:
        f.write('iteration,nested_log_evidence,max_log_likelihood,ratio_latest_loglikelihood,MC_estimate,total_points\n')
    
    for i in range(steps):
        Ls=Ls[np.argsort(-Ls)]
        points=points[np.argsort(-Ls)]
        
        L_i.append(Ls[-1])
        X_i.append(mp.exp(-(i+1)/N))
        Z+=(X_i[-2]-X_i[-1])*mp.exp(L_i[-1])
        if (i+1)%5==0:
            with open(outputfile, 'a') as f:
                f.write(f'{i+1},{mp.log(Z)},{Ls[0]},{L_i[-1]-L_i[-2]},{mp.log(update_MC)-mp.log(total_points)},{mp.log(L_tot_exp)},{total_points}\n')
    
        if mp.log(Z+mp.exp(Ls[0])*X_i[-1])-mp.log(Z)<0.5:
            print('Converged, with the criterion 0', mp.log(Z+mp.exp(Ls[0])*X_i[-1])-mp.log(Z))
            with open(outputfile, 'a') as f:
                f.write(f'{i+1},{mp.log(Z)},{Ls[0]},{L_i[-1]-L_i[-2]},{mp.log(update_MC)-mp.log(total_points)},{mp.log(L_tot_exp)},{total_points}\n0,0,0,0,0,0,0\n')
            break
        if L_i[-1]-L_i[-2]<1/N:
            print('Converged, with the criterion 1', L_i[-1]-L_i[-2])
            with open(outputfile, 'a') as f:
                f.write(f'{i+1},{mp.log(Z)},{Ls[0]},{L_i[-1]-L_i[-2]},{mp.log(update_MC)-mp.log(total_points)},{mp.log(L_tot_exp)},{total_points}\n1,1,1,1,1,1,1\n')
            break
        Ls[-1], points[-1], L_tot_exp, n=_update(L_i[-1], alpha, mu_0, lambda_, S, nu, N_clusters, 2, data[:,:-1], max_rep=10000, random_generator=numpy_randomGen)  
        update_MC+=L_tot_exp
        total_points+=n
        L_i=[L_i[-1]]
    Z+=(X_i[-2]-X_i[-1])*mp.exp(L_i[-1])
    print(f'Iteration {i+1}: {mp.log(Z)}, {Ls[0]}, \t the MC estimate is {mp.log(update_MC)-mp.log(total_points)}, with {total_points} points')