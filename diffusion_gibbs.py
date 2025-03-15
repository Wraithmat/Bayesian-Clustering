import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import invwishart, dirichlet, multivariate_normal
import argparse
import mpmath as mp
from tqdm import tqdm
from scipy.integrate import simpson, trapz
from joblib import Parallel, delayed

null_gen=np.random.default_rng(0)

def arithmethic_average(log_likelihoods):
    vec_exp = lambda x: mp.exp(x)
    vec_exp = np.vectorize(vec_exp)
    avg = mp.fsum(vec_exp(log_likelihoods))/len(log_likelihoods)
    return avg, mp.log(avg)

def harmonic_mean(loglikelihoods):
    neg_vec_exp = lambda x: mp.exp(-x)
    neg_vec_exp = np.vectorize(neg_vec_exp)
    h_mean = len(loglikelihoods)/mp.fsum(neg_vec_exp(loglikelihoods))
    return h_mean, mp.log(h_mean)

class DiffusionGibbs:
    def __init__(self, data, alpha=[1], mu_0=np.zeros(2), lambda_=0.025, S=np.eye(2), nu=5, true_labels=None, seed=0, hot_start=0, n_clusters_0=3, n_iter=100):
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
        self.MH_acceptance=[0,0]

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
            if len(self.alpha)==1:
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


    def sample_pi(self, alpha, z, numpy_randomGen=None):
        """
        Sample the probability of each cluster. To be used in the gibbs step.
        """
        if numpy_randomGen is None:
            numpy_randomGen=self.numpy_randomGen
        for i in range(len(alpha)):
            alpha[i]+=np.count_nonzero(z==i)
        return dirichlet.rvs(alpha, random_state=numpy_randomGen)


    def sample_z(self, features, mus, Sigmas, Pi, beta=1, numpy_randomGen=None):
        """
        Sample the cluster assignments for each data point. To be used in the gibbs step.
        """
        if numpy_randomGen is None:
            numpy_randomGen=self.numpy_randomGen
        p_z=np.zeros((len(features), len(mus)))
        for j in range(len(mus)):
            p_z[:,j]=(Pi[j]*multivariate_normal.pdf(features, mean=mus[j], cov=Sigmas[j])**beta)
        p_z=p_z/np.sum(p_z, axis=1)[:,None]
        z=(p_z.cumsum(1)>numpy_randomGen.random((len(features)))[:,None]).argmax(1)
        return z


    def sample_mus_Sigmas(self, features, z, mu_0, lambda_, S, nu, beta=1, numpy_randomGen=None):
        """
        Sample the means and covariances for each cluster in the gibbs step.
        mus and sigmas are a list of the means and covariances for each cluster.
        """
        if numpy_randomGen is None:
            numpy_randomGen=self.numpy_randomGen
        mus=[]
        Sigmas=[]
        for i in np.unique(z):
            n = np.sum(z==i)

            lambda_n=lambda_+beta*n
            nu_n=nu+beta*n
            mu_n=(beta*n*np.mean(features[z==i], axis=0)+lambda_*mu_0)/lambda_n
            vector=lambda_*mu_0+beta*n*np.mean(features[z==i], axis=0)
            S_n= S + np.outer(vector, vector)/lambda_n + lambda_*np.outer(mu_0,mu_0) + beta*features.T@features

            Sigmas.append(invwishart.rvs(nu_n, S_n, random_state=numpy_randomGen))
            mus.append(numpy_randomGen.multivariate_normal(mu_n, 1/lambda_n*Sigmas[-1]))
        return mus, Sigmas
    
    def compute_loglikelihood_(self, zi, mus, Sigmas):
        """
        Returns the log likelihood for you dataset.
        The temperature beta should be considered in a later step.
        """
        l_likelihood=0
        for i in range(self.n_clusters):
            l_likelihood+=np.sum(np.log(multivariate_normal.pdf(self.data[zi==i], mean=mus[i], cov=Sigmas[i])))
        return l_likelihood

    
    def gibbs_step(self, beta=1):
        self.pi=self.sample_pi(self.alpha, self.zi, self.numpy_randomGen)
        self.zi=self.sample_z(self.data, self.mus, self.Sigmas, self.pi, beta, self.numpy_randomGen)
        self.mus, self.Sigmas=self.sample_mus_Sigmas(self.data, self.zi, self.mu_0, self.lambda_, self.S, self.nu, beta, self.numpy_randomGen)

    
    def MH_acceptance(self, z1, z2, mu_1, mu_2, Sigmas_1, Sigmas_2, beta1, beta2):
        """
        Metropolis-Hastings step to accept or reject the new parameters.
        """
        l_likelihood_1=self.compute_loglikelihood_(z1, mu_1, Sigmas_1)
        l_likelihood_2=self.compute_loglikelihood_(z2, mu_2, Sigmas_2)
        p=min(1, np.exp((beta1-beta2)*(l_likelihood_2-l_likelihood_1)))
        if self.numpy_randomGen.random()<p:
            self.MH_acceptance[0]+=1
            self.MH_acceptance[1]+=1
            return True
        else:
            self.MH_acceptance[1]+=1
            return False 
            
    
    def gibbs_sampler(self):
        dictionary={'pi':self.pi, 'zi':self.zi, 'mus':self.mus, 'Sigmas':self.Sigmas}

        for _ in tqdm(range(self.n_iter)):
            self.gibbs_step()
            dictionary['pi']=np.concatenate([dictionary['pi'],self.pi], axis=0)
            dictionary['zi']=np.concatenate([dictionary['zi'],self.zi], axis=0)
            dictionary['mus']=np.concatenate([dictionary['mus'],self.mus], axis=0)
            dictionary['Sigmas']=np.concatenate([dictionary['Sigmas'],self.Sigmas], axis=0)
        
        return dictionary

    
    def diffusion_gibbs(self, betas=None):
        """"
        Performs the gibbs sampler with a diffusion step, allowing for interchange between temperature levels.
        If the temperatures are not fixed, the strategy by Wangang Xie, et al. (https://doi.org/10.1093/sysbio/syq085) is implemented.
        """

        if betas is None:
            #Obtain the temperatures which equally space the quantiles of the beta distribution
            betas=np.linspace(0,1,10)**(1/0.3)
        
        dictionary={}
        for i in betas:
            self.init_params(0)
            dictionary.update({'pi_%.4f'%i:self.pi, 'zi_%.4f'%i:self.zi, 'mus_%.4f'%i:self.mus, 'Sigmas_%.4f'%i:self.Sigmas, 'loglikelihood_%.4f'%i:[]})
        
        for iter in tqdm(range(self.n_iter)):
            for i in betas:
                pi_=self.sample_pi(self.alpha, dictionary['zi_%.4f'%i][-1], self.numpy_randomGen)
                zi_=self.sample_z(self.data, dictionary['mus_%.4f'%i][-1], dictionary['Sigmas_%.4f'%i][-1], dictionary['pi_%.4f'%i][-1], i, self.numpy_randomGen)
                mus_, Sigmas_=self.sample_mus_Sigmas(self.data, dictionary['zi_%.4f'%i][-1], self.mu_0, self.lambda_, self.S, self.nu, i, self.numpy_randomGen)

                dictionary['pi_%.4f'%i]=np.concatenate(dictionary['pi_%.4f'%i],pi_)
                dictionary['zi_%.4f'%i]=np.concatenate(dictionary['zi_%.4f'%i],zi_)
                dictionary['mus_%.4f'%i]=np.concatenate(dictionary['mus_%.4f'%i],mus_)
                dictionary['Sigmas_%.4f'%i]=np.concatenate(dictionary['Sigmas_%.4f'%i],Sigmas_)
                dictionary['loglikelihood_%.4f'%i].append(self.compute_loglikelihood_(zi_, mus_, Sigmas_))
            
            if iter%2==0:
                for i in range(0,len(betas),2):
                    acc=self.MH_acceptance(dictionary['zi_%.4f'%betas[i]], dictionary['zi_%.4f'%betas[i+1]], dictionary['mus_%.4f'%betas[i]], dictionary['mus_%.4f'%betas[i+1]], dictionary['Sigmas_%.4f'%betas[i]], dictionary['Sigmas_%.4f'%betas[i+1]], betas[i], betas[i+1])
                    if acc:
                        dictionary['zi_%.4f'%betas[i]]=np.concatenate(dictionary['zi_%.4f'%betas[i]],dictionary['zi_%.4f'%betas[i+1]][-1])
                        dictionary['zi_%.4f'%betas[i+1]]=np.concatenate(dictionary['zi_%.4f'%betas[i+1]],dictionary['zi_%.4f'%betas[i]][-2])
                        dictionary['mus_%.4f'%betas[i]]=np.concatenate(dictionary['mus_%.4f'%betas[i]],dictionary['mus_%.4f'%betas[i+1]][-1])
                        dictionary['mus_%.4f'%betas[i+1]]=np.concatenate(dictionary['mus_%.4f'%betas[i+1]],dictionary['mus_%.4f'%betas[i]][-2])
                        dictionary['Sigmas_%.4f'%betas[i]]=np.concatenate(dictionary['Sigmas_%.4f'%betas[i]],dictionary['Sigmas_%.4f'%betas[i+1]][-1])
                        dictionary['Sigmas_%.4f'%betas[i+1]]=np.concatenate(dictionary['Sigmas_%.4f'%betas[i+1]],dictionary['Sigmas_%.4f'%betas[i]][-2])
                        dictionary['loglikelihood_%.4f'%betas[i]]=dictionary['loglikelihood_%.4f'%betas[i]]+dictionary['loglikelihood_%.4f'%betas[i+1]][-1]
                        dictionary['loglikelihood_%.4f'%betas[i+1]]=dictionary['loglikelihood_%.4f'%betas[i]]+dictionary['loglikelihood_%.4f'%betas[i+1]][-2]

            else:
                for i in range(1,len(betas),2):
                    acc=self.MH_acceptance(dictionary['zi_%.4f'%betas[i]], dictionary['zi_%.4f'%betas[i+1]], dictionary['mus_%.4f'%betas[i]], dictionary['mus_%.4f'%betas[i+1]], dictionary['Sigmas_%.4f'%betas[i]], dictionary['Sigmas_%.4f'%betas[i+1]], betas[i], betas[i+1])       
                    if acc:
                        dictionary['zi_%.4f'%betas[i]]=np.concatenate(dictionary['zi_%.4f'%betas[i]],dictionary['zi_%.4f'%betas[i+1]][-1])
                        dictionary['zi_%.4f'%betas[i+1]]=np.concatenate(dictionary['zi_%.4f'%betas[i+1]],dictionary['zi_%.4f'%betas[i]][-2])
                        dictionary['mus_%.4f'%betas[i]]=np.concatenate(dictionary['mus_%.4f'%betas[i]],dictionary['mus_%.4f'%betas[i+1]][-1])
                        dictionary['mus_%.4f'%betas[i+1]]=np.concatenate(dictionary['mus_%.4f'%betas[i+1]],dictionary['mus_%.4f'%betas[i]][-2])
                        dictionary['Sigmas_%.4f'%betas[i]]=np.concatenate(dictionary['Sigmas_%.4f'%betas[i]],dictionary['Sigmas_%.4f'%betas[i+1]][-1])
                        dictionary['Sigmas_%.4f'%betas[i+1]]=np.concatenate(dictionary['Sigmas_%.4f'%betas[i+1]],dictionary['Sigmas_%.4f'%betas[i]][-2])
                        dictionary['loglikelihood_%.4f'%betas[i]]=dictionary['loglikelihood_%.4f'%betas[i]]+dictionary['loglikelihood_%.4f'%betas[i+1]][-1]
                        dictionary['loglikelihood_%.4f'%betas[i+1]]=dictionary['loglikelihood_%.4f'%betas[i]]+dictionary['loglikelihood_%.4f'%betas[i+1]][-2]


        return dictionary
    
    def process_beta(self, i, dictionary, n_iter, randomGen):
        """
        Process the diffusion step for a given temperature i.
        """
        pi_=dictionary['zi_%.4f' % i][-1].reshape(1,-1)
        zi_=dictionary['zi_%.4f' % i][-1].reshape(1,-1)
        mus_=[dictionary['mus_%.4f' % i][-1]]
        Sigmas_=[dictionary['Sigmas_%.4f' % i][-1]]
        loglikelihood=[]

        for _ in range(n_iter):
            _pi_ = self.sample_pi(self.alpha, pi_[-1], randomGen)
            _zi_ = self.sample_z(self.data, mus_[-1], Sigmas_[-1], pi_[-1], i, randomGen)
            _mus_, _Sigmas_ = self.sample_mus_Sigmas(self.data, zi_[-1], self.mu_0, self.lambda_, self.S, self.nu, i, randomGen)
            loglikelihood.append(self.compute_loglikelihood_(zi_[-1], mus_[-1], Sigmas_[-1]))
            pi_=np.concatenate([pi_,_pi_], axis=0)
            zi_=np.concatenate([zi_,_zi_], axis=0)
            mus_=np.concatenate([mus_,_mus_], axis=0)
            Sigmas_=np.concatenate([Sigmas_,_Sigmas_], axis=0)

        return (pi_[1:], zi_[1:], mus_[1:], Sigmas_[1:], loglikelihood)
    

    def parallel_diffusion_gibbs(self, betas=None, n_gibbs_iter=10, n_jobs=4):
        """
        Parallel version of the diffusion gibbs sampler.
        """
        if betas is None:
            betas=np.linspace(0,1,10)**(1/0.3)
        betas=np.sort(betas)

        ss = np.random.SeedSequence(self.seed)
        child_seeds = ss.spawn(len(betas))
        streams = [np.random.default_rng(s) for s in child_seeds]
        
        dictionary={}
        for i in betas:
            self.init_params(0)
            dictionary.update({'pi_%.4f'%i:self.pi, 'zi_%.4f'%i:self.zi, 'mus_%.4f'%i:self.mus, 'Sigmas_%.4f'%i:self.Sigmas, 'loglikelihood_%.4f'%i:[]})
        print(dictionary.values())
        for iter in tqdm(range(self.n_iter)):

            results=Parallel(n_jobs=n_jobs)(delayed(self.process_beta)(betas[i], dictionary, n_gibbs_iter, streams[i]) for i in range(len(betas)))                

            for i in range(len(betas)):
                dictionary['pi_%.4f'%betas[i]]=np.concatenate(dictionary['pi_%.4f'%betas[i]],results[i][0])
                dictionary['zi_%.4f'%betas[i]]=np.concatenate(dictionary['zi_%.4f'%betas[i]],results[i][1])
                dictionary['mus_%.4f'%betas[i]]=np.concatenate(dictionary['mus_%.4f'%betas[i]],results[i][2])
                dictionary['Sigmas_%.4f'%betas[i]]=np.concatenate(dictionary['Sigmas_%.4f'%betas[i]],results[i][3])
                dictionary['loglikelihood_%.4f'%betas[i]]=dictionary['loglikelihood_%.4f'%betas[i]]+results[i][4]
            
            if iter%2==0:
                for i in range(0,len(betas),2):
                    acc=self.MH_acceptance(dictionary['zi_%.4f'%betas[i]], dictionary['zi_%.4f'%betas[i+1]], dictionary['mus_%.4f'%betas[i]], dictionary['mus_%.4f'%betas[i+1]], dictionary['Sigmas_%.4f'%betas[i]], dictionary['Sigmas_%.4f'%betas[i+1]], betas[i], betas[i+1])
                    if acc:
                        dictionary['zi_%.4f'%betas[i]]=np.concatenate(dictionary['zi_%.4f'%betas[i]],dictionary['zi_%.4f'%betas[i+1]][-1])
                        dictionary['zi_%.4f'%betas[i+1]]=np.concatenate(dictionary['zi_%.4f'%betas[i+1]],dictionary['zi_%.4f'%betas[i]][-2])
                        dictionary['mus_%.4f'%betas[i]]=np.concatenate(dictionary['mus_%.4f'%betas[i]],dictionary['mus_%.4f'%betas[i+1]][-1])
                        dictionary['mus_%.4f'%betas[i+1]]=np.concatenate(dictionary['mus_%.4f'%betas[i+1]],dictionary['mus_%.4f'%betas[i]][-2])
                        dictionary['Sigmas_%.4f'%betas[i]]=np.concatenate(dictionary['Sigmas_%.4f'%betas[i]],dictionary['Sigmas_%.4f'%betas[i+1]][-1])
                        dictionary['Sigmas_%.4f'%betas[i+1]]=np.concatenate(dictionary['Sigmas_%.4f'%betas[i+1]],dictionary['Sigmas_%.4f'%betas[i]][-2])
                        dictionary['loglikelihood_%.4f'%betas[i]]=dictionary['loglikelihood_%.4f'%betas[i]]+dictionary['loglikelihood_%.4f'%betas[i+1]][-1]
                        dictionary['loglikelihood_%.4f'%betas[i+1]]=dictionary['loglikelihood_%.4f'%betas[i]]+dictionary['loglikelihood_%.4f'%betas[i+1]][-2]

            else:
                for i in range(1,len(betas),2):
                    acc=self.MH_acceptance(dictionary['zi_%.4f'%betas[i]], dictionary['zi_%.4f'%betas[i+1]], dictionary['mus_%.4f'%betas[i]], dictionary['mus_%.4f'%betas[i+1]], dictionary['Sigmas_%.4f'%betas[i]], dictionary['Sigmas_%.4f'%betas[i+1]], betas[i], betas[i+1])       
                    if acc:
                        dictionary['zi_%.4f'%betas[i]]=np.concatenate(dictionary['zi_%.4f'%betas[i]],dictionary['zi_%.4f'%betas[i+1]][-1])
                        dictionary['zi_%.4f'%betas[i+1]]=np.concatenate(dictionary['zi_%.4f'%betas[i+1]],dictionary['zi_%.4f'%betas[i]][-2])
                        dictionary['mus_%.4f'%betas[i]]=np.concatenate(dictionary['mus_%.4f'%betas[i]],dictionary['mus_%.4f'%betas[i+1]][-1])
                        dictionary['mus_%.4f'%betas[i+1]]=np.concatenate(dictionary['mus_%.4f'%betas[i+1]],dictionary['mus_%.4f'%betas[i]][-2])
                        dictionary['Sigmas_%.4f'%betas[i]]=np.concatenate(dictionary['Sigmas_%.4f'%betas[i]],dictionary['Sigmas_%.4f'%betas[i+1]][-1])
                        dictionary['Sigmas_%.4f'%betas[i+1]]=np.concatenate(dictionary['Sigmas_%.4f'%betas[i+1]],dictionary['Sigmas_%.4f'%betas[i]][-2])
                        dictionary['loglikelihood_%.4f'%betas[i]]=dictionary['loglikelihood_%.4f'%betas[i]]+dictionary['loglikelihood_%.4f'%betas[i+1]][-1]
                        dictionary['loglikelihood_%.4f'%betas[i+1]]=dictionary['loglikelihood_%.4f'%betas[i]]+dictionary['loglikelihood_%.4f'%betas[i+1]][-2]

        n_count=[]
        for i in betas:
            dictionary['loglikelihood_%.4f'%i]=np.array(dictionary['loglikelihood_%.4f'%i])
            n_count.append(len(dictionary['loglikelihood_%.4f'%i]))

        if 0 in betas:
            z_avg, log_z_avg = arithmethic_average(dictionary['loglikelihood_%.4f'%0])
        if 1 in betas:
            z_hmean, log_z_hmean = harmonic_mean(dictionary['loglikelihood_%.4f'%1])

        thermodynamic_averages = [np.average(dictionary['loglikelihood_%.4f'%i]) for i in betas]
        th = (trapz(thermodynamic_averages, betas), simpson(thermodynamic_averages, betas))

        vec_exp = lambda x: mp.exp(x)
        vec_exp = np.vectorize(vec_exp)
        r_k=[1/n_count[i+1]*mp.fsum(vec_exp(dictionary['loglikelihood_%.4f'%betas[i+1]]*(betas[i+1]-betas[i]))) for i in range(len(betas[1:]))]
        stepping_stone = (mp.fprod(r_k), mp.fsum(vec_exp(r_k)))
        
        return dictionary, (z_avg, log_z_avg), (z_hmean, log_z_hmean), th, stepping_stone, r_k, thermodynamic_averages

