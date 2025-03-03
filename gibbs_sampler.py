import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import invwishart, dirichlet, multivariate_normal
import argparse
from line_profiler import profile

null_gen=np.random.default_rng(0)

def parse_arguments():
    """
    Parses command-line arguments for the Gibbs Sampler for Clustering.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Command-line arguments:
        --seed (int): Random seed (default: 0).
        --data_path (str): Path to the data file (default: 'data.npy').
        --n_clusters (int): Number of clusters (default: 3).
        --hot_start (str): Initialize Gibbs sampling with KMeans (default: '1').
        --n_iter (int): Number of iterations (default: 100).
        --savepath (str): Path to save the results (default: 'data/gibbs_sampler.npy').
    """
    parser = argparse.ArgumentParser(description='Gibbs Sampler for Clustering')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--data_path', type=str, default='data/data.npy', help='Path to the data file')
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters')
    parser.add_argument('--hot_start', type=str, default='1', help='Initialize Gibbs sampling with KMeans')
    parser.add_argument('--n_iter', type=int, default=100, help='Number of iterations')
    parser.add_argument('--savepath', type=str, default='data/gibbs_sampler.npy', help='Path to save the results')
    return parser.parse_args()

@profile
def sample_pi(alpha, z, numpy_randomGen=null_gen):
    for i in range(len(alpha)):
        alpha[i]+=np.count_nonzero(z==i)
    return dirichlet.rvs(alpha, random_state=numpy_randomGen)

@profile
def sample_z_nonvectorized(features, mus, Sigmas, Pi, numpy_randomGen=null_gen):
    z=np.zeros(len(features))
    for i in range(len(features)):
        probs=[]
        for j in range(len(mus)):
            probs.append(Pi[j]*multivariate_normal.pdf(features[i], mean=mus[j], cov=Sigmas[j]))
        z[i]=numpy_randomGen.choice(np.arange(len(mus)), p=probs/np.sum(probs))
    return z

@profile
def sample_z(features, mus, Sigmas, Pi, numpy_randomGen=null_gen):
    p_z=np.zeros((len(features), len(mus)))
    for j in range(len(mus)):
        p_z[:,j]=(Pi[j]*multivariate_normal.pdf(features, mean=mus[j], cov=Sigmas[j]))
    p_z=p_z/np.sum(p_z, axis=1)[:,None]
    z=(p_z.cumsum(1)>numpy_randomGen.random((len(features)))[:,None]).argmax(1)
    return z

@profile
def sample_mus_Sigmas(features, z, mu_0, lambda_, S, nu, numpy_randomGen=null_gen):
    mus=[]
    Sigmas=[]
    for i in np.unique(z):
        n = np.sum(z==i)

        lambda_n=lambda_+n
        nu_n=nu+n
        mu_n=(n*np.mean(features[z==i], axis=0)+lambda_*mu_0)/lambda_n
        X=features[z==i]-np.mean(features[z==i], axis=0)
        vec=np.mean(features[z==i], axis=0)-mu_0
        S_n=S+X.T@X+lambda_*n/(lambda_+n)*vec.T@vec

        Sigmas.append(invwishart.rvs(nu_n, S_n, random_state=numpy_randomGen))
        mus.append(numpy_randomGen.multivariate_normal(mu_n, 1/lambda_n*Sigmas[-1]))
    return mus, Sigmas

if __name__=='__main__':
    args = parse_arguments()

    seed = args.seed
    numpy_randomGen = np.random.default_rng(seed)

    data=np.load(args.data_path)

    n_clusters = args.n_clusters

    features=data[:,:-1]
    labels=data[:,-1]
    if args.hot_start=='1':
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(features)
        zi=kmeans.labels_
        pi=np.array([np.count_nonzero(zi==i) for i in range(n_clusters)])/len(zi)
        mus=kmeans.cluster_centers_
        Sigmas=[]
        for i in range(n_clusters):
            Sigmas.append(np.cov(features[zi==i].T))
    else:
        pi=np.ones(n_clusters)/n_clusters
        zi=numpy_randomGen.choice(np.arange(n_clusters), size=len(features), p=pi)
        mus=[]
        Sigmas=[]
        for i in range(n_clusters):
            mus.append(numpy_randomGen.multivariate_normal(np.mean(features[zi==i], axis=0), np.cov(features[zi==i].T)))
            Sigmas.append(np.cov(features[zi==i].T))
    mus=[mus]
    Sigmas=[Sigmas]
    alpha = [1,1.5,2,2,2]
    alpha = alpha[:n_clusters]
    if args.hot_start=='1':
        alpha=np.array(alpha)[np.argsort([np.count_nonzero(zi==i) for i in range(n_clusters)])]
    for i in range(args.n_iter):
        pi_=sample_pi(alpha, zi[-1], numpy_randomGen)
        pi=np.vstack([pi, pi_])
        zi_=sample_z(features, mus[-1], Sigmas[-1], pi[-1], numpy_randomGen)
        zi=np.vstack([zi, zi_])
        mus_, Sigmas_=sample_mus_Sigmas(features, zi[-1], np.zeros(len(features[0])), 0.025, np.eye(2), 5, numpy_randomGen)
        mus.append(np.array(mus_))
        Sigmas.append(np.array(Sigmas_))
    dictionary={'pi':pi, 'zi':zi, 'mus':mus, 'Sigmas':Sigmas}
    np.save(args.savepath, dictionary, allow_pickle=True)

