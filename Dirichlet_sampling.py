import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import invwishart, dirichlet, multivariate_normal
import argparse
from line_profiler import profile
from scipy.special import gamma
import mpmath as mp
from tqdm import tqdm

null_gen=np.random.default_rng(0)

class DPSampler:
    def __init__(self, data, alpha=1, mu_0=np.zeros(2), lambda_=0.025, S=np.eye(2), nu=5, true_labels=None, seed=0, hot_start=0, n_clusters_0=3, n_iter=100):
        '''
        data: np.array of shape (N, D)
        alpha, mu_0, lambda_, S, nu: hyperparameters
        true_labels: np.array of shape (N,) with true labels
        seed: seed for random number generator
        hot_start: True if using kmeans initialization
        n_clusters_0: number of clusters for initialization
        '''
        
        self.seed=seed
        self.data=data
        self.hot_start=hot_start
        self.n_iter=n_iter
        self.true_labels=true_labels
        self.N=len(data)
        self.D=len(data[0])
        self.n_clusters=n_clusters_0
        self.numpy_randomGen=np.random.default_rng(seed)
        
        #fixing hyperparameters
        self.mu_0=mu_0
        self.lambda_=lambda_
        self.S=S
        self.nu=nu
        self.alpha=alpha
        self.init_params(hot_start)
        self.live_clusters=np.unique(self.zi)
        self.n_clusters=len(self.live_clusters)

    
    def init_params(self,hot_start):
        if hot_start!=0:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed).fit(self.data)
            self.zi=kmeans.labels_
            self.pi=np.array([np.count_nonzero(self.zi==i) for i in range(self.n_clusters)])/self.N
            self.mus=kmeans.cluster_centers_
            self.Sigmas=[]
            for i in range(self.n_clusters):
                self.Sigmas.append(np.cov(self.data[self.zi==i].T))
        else:
            if self.len(self.alpha)==1:
                self.pi=self.numpy_randomGen.dirichlet(self.alpha*np.ones(self.n_clusters)/self.n_clusters)
            else:
                self.pi=self.numpy_randomGen.dirichlet(self.alpha)
            self.zi=self.numpy_randomGen.choice(np.arange(self.n_clusters), size=len(self.data), p=self.pi)
            self.mus=[]
            self.Sigmas=[]
            for i in range(self.n_clusters):
                self.mus.append(null_gen.multivariate_normal(np.mean(self.data[self.zi==i], axis=0), np.cov(self.data[self.zi==i].T)))
                self.Sigmas.append(np.cov(self.data[self.zi==i].T))
            self.mus=np.array(self.mus)

    
    def sample_mus_Sigmas(self):
        assert(np.unique(self.zi).shape[0]==self.n_clusters)
        assert(self.live_clusters.shape[0]==self.n_clusters)
        assert(len(self.mus)==len(self.Sigmas))
        for i in range(self.n_clusters):
            n = np.sum(self.zi==self.live_clusters[i])
            lambda_n=self.lambda_+n
            nu_n=self.nu+n
            mu_n=(n*np.mean(self.data[self.zi==self.live_clusters[i]], axis=0)+self.lambda_*self.mu_0)/lambda_n
            X=self.data[self.zi==self.live_clusters[i]]-np.mean(self.data[self.zi==self.live_clusters[i]], axis=0)
            vec=np.mean(self.data[self.zi==self.live_clusters[i]], axis=0)-self.mu_0
            S_n=self.S+X.T@X+self.lambda_*n/(self.lambda_+n)*vec.T@vec

            self.Sigmas[self.live_clusters[i]]=(invwishart.rvs(nu_n, S_n, random_state=self.numpy_randomGen))
            self.mus[self.live_clusters[i]]=(self.numpy_randomGen.multivariate_normal(mu_n, 1/lambda_n*self.Sigmas[self.live_clusters[i]]))
    

    def sample_z_i_given_ms(self):
        '''
        Iterates over the data and samples z_i given the other parameters
        '''
        n_i=[np.count_nonzero(self.zi==j) for j in range(max(self.live_clusters)+1)]
        for i in range(self.N):
            #compute p(z_i|z_{-i}, x, m, S)
            p_z_i=np.zeros(self.n_clusters+1)
            n_i[self.zi[i]]-=1

            #compute p(z_i=j|z_{-i}, x, m, S) for each cluster
            for j in range(self.n_clusters):
                p_z_i[j]=n_i[self.live_clusters[j]]*multivariate_normal.pdf(self.data[i], mean=self.mus[self.live_clusters[j]], cov=self.Sigmas[self.live_clusters[j]])
            #compute the probability of z_i being in new cluster
            p_z_i[-1]=self.alpha*mp.exp(self.log_likelihood_model(self.data[i], self.lambda_, self.mu_0, self.nu, self.S))
            #normalize
            p_z_i=p_z_i/np.sum(p_z_i)
            
            #sample z_i
            new_point=self.numpy_randomGen.choice(np.arange(self.n_clusters+1), p=p_z_i)
            if new_point==self.n_clusters:
                self.zi[i]=self.n_clusters
                self.mus=np.vstack([self.mus,self.data[i]])
                self.Sigmas.append(self.S)
                n_i.append(0)
            else:
                self.zi[i]=self.live_clusters[new_point]
            n_i[self.zi[i]]+=1
            self.live_clusters=np.unique(self.zi)
            self.n_clusters=len(self.live_clusters)
            


    def log_likelihood_model(self, data, lambda_, mu_0, nu, S):
        
        prefactor=(self.D/2*(-np.log(np.pi)+np.log(lambda_/(1+lambda_)))-np.log(gamma(nu/2+(1-self.D)/2))+np.log(gamma(nu/2+1/2))+nu/2*np.log(np.linalg.det(S)))
        S_tot=S-1/(1+lambda_)*np.outer(data+lambda_*mu_0,data+lambda_*mu_0)+lambda_*np.outer(mu_0,mu_0)+np.outer(data,data)
        likelihood_=np.linalg.det(S_tot)**(-(nu+1)/2)
        log_likelihood=mp.log(likelihood_)+prefactor
        return log_likelihood
    
    def run(self, filename=None):
        groups={}
        groups["assignements"]=self.zi
        for i in self.live_clusters:
            groups[f"mu_{i}"]=self.mus[i]
            groups[f"Sigma_{i}"]=self.Sigmas[i].reshape(1,2,2)

        for iter in tqdm(range(self.n_iter)):
            self.sample_mus_Sigmas()
            self.sample_z_i_given_ms()
            groups["assignements"]=np.vstack([groups["assignements"],self.zi])
            for i in self.live_clusters:
                if f"mu_{i}" not in groups.keys():
                    groups[f"mu_{i}"]=self.mus[i]
                    groups[f"Sigma_{i}"]=self.Sigmas[i].reshape(1,2,2)
                else:
                    groups[f"mu_{i}"]=np.vstack([groups[f"mu_{i}"],self.mus[i]])
                    groups[f"Sigma_{i}"]=np.concatenate([groups[f"Sigma_{i}"],self.Sigmas[i].reshape(1,2,2)], axis=0)
            
        if filename!=None:
            np.save(filename, groups)