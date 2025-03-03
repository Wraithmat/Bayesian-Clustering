import numpy as np
from scipy.stats import invwishart, dirichlet, multivariate_normal

null_Gen = np.random.default_rng(0)

def sample_prior(alpha, mu_0, lambda_, S, nu, N_points, N_dataset, random_generator=null_Gen):
    """
    Returns N_points sample from the prior distribution of the parameters of the model.
    The results are stored in a matrix of shape (N_points, N_clusters*N_dim+N_clusters*N_dim**2+N_clusters+N_dataset), where:
    - the first N_clusters*N_dim columns contain the means of the clusters
    - the next N_clusters*N_dim**2 columns contain the covariance matrices of the clusters
    - the next N_clusters columns contain the mixing coefficients of the clusters
    - the last N_dataset columns contain the cluster assignments of the data points
    """
    
    N_clusters=len(alpha)
    N_dim=len(mu_0)
    result=np.empty((N_points, N_clusters*N_dim+N_clusters*N_dim**2+N_clusters+N_dataset), dtype=float)
    result[:,N_clusters*N_dim+N_clusters*N_dim**2:N_clusters*N_dim+N_clusters*N_dim**2+N_clusters]=dirichlet.rvs(alpha, size=N_points, random_state=random_generator)

    result[:,-N_dataset:]=(result[:,None,:N_clusters].cumsum(2)>numpy_randomGen.random((N_points,N_dataset))[:,:,None]).argmax(2)
    
    sample_mus = lambda x: random_generator.multivariate_normal(mu_0, x.reshape((N_dim,N_dim)))
    sample_mus = np.vectorize(sample_mus, signature='(k)->(n)')
    

    for i in range(len(alpha)):
        result[:,N_clusters*N_dim+i*N_dim**2:N_clusters*N_dim+(i+1)*N_dim**2]=(invwishart.rvs(nu, S, size=N_points, random_state=random_generator)).reshape((N_points, N_dim**2))
        result[:,N_dim*i:N_dim*(i+1)]=sample_mus(1/lambda_*result[:,N_clusters*N_dim+i*N_dim**2:N_clusters*N_dim+(i+1)*N_dim**2])
    return result

def compute_likelihood(data, N_clusters, N_dim, params):
    mus=params[:N_clusters*N_dim].reshape(N_clusters,N_dim)
    Sigmas=params[N_clusters*N_dim:N_clusters*N_dim+N_clusters*N_dim**2].reshape(N_clusters,N_dim,N_dim)
    zi=params[N_clusters*N_dim+N_clusters*N_dim**2+N_clusters:].astype(int)
    likelihood=0
    for i in range(len(data)):
        likelihood+=np.log(multivariate_normal.pdf(data[i], mean=mus[zi[i]], cov=Sigmas[zi[i]]))
    return likelihood

def _update(L, alpha, mu_0, lambda_, S, nu, N_clusters, N_dim, data, max_rep=100, random_generator=null_Gen):
    L_=-np.inf
    rep=0
    while L_<L:
        rep+=1
        if rep>max_rep:
            print(f'{rep} samples were performed but the likelihood is still smaller')
            break
        trial=sample_prior(alpha, mu_0, lambda_, S, nu, 1, len(data), random_generator)[0]
        L_=compute_likelihood(data, N_clusters, N_dim, trial)
    return L_, trial

if __name__=='__main__':
    seed=0
    N=100
    steps=20
    data=np.load('./data/data.npy')
    numpy_randomGen = np.random.default_rng(seed)

    alpha = [1,1.5,2]
    mu_0 = [0,0]
    lambda_=0.025
    S=np.eye(2)
    nu=5
    N_data=len(data)

    points=sample_prior(alpha, mu_0, lambda_, S, nu, N, N_data, numpy_randomGen)
    X_i=[1]
    L_i=[]
    Z=0
    Ls=np.array([compute_likelihood(data[:,:-1],3,2,i) for i in points])
    for i in range(steps):
        Ls=Ls[np.argsort(-Ls)]
        points=points[np.argsort(-Ls)]

        L_i.append(Ls[-1])
        X_i.append(np.exp(-i/N))
        Z+=(X_i[-2]-X_i[-1])*np.exp(L_i[-1])
        print(f'Iteration {i+1}: {Z}, {L_i[-1]}, {Ls[:3]}')
        Ls[-1], points[-1]=_update(L_i[-1], alpha, mu_0, lambda_, S, nu, 3, 2, data[:,:-1], max_rep=100, random_generator=numpy_randomGen)  
    print(Z)