import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import invwishart, dirichlet, multivariate_normal
import argparse
import mpmath as mp
from tqdm import tqdm
from scipy.integrate import simpson, trapz
from joblib import Parallel, delayed
from scipy.integrate import newton_cotes


null_gen=np.random.default_rng(0)

vec_log = np.vectorize(mp.log)

vec_exp = np.vectorize(mp.exp)

def autocorr2(x):
    r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    c=(r2/x.shape-np.mean(x)**2)/np.std(x)**2
    return c[:len(x)//2]

def block_error(timeserie, block_sizes, inv_sigma_average=False):
    '''
    Returns a list of standard deviations of the mean of the blocked timeserie and the error on the standard deviation of the mean.
    If inv_sigma_average is True, it returns a biased estimate of average sigma, using as many sigmas as needed to get in the plateau.
    
    The output will contain an estimate of the standard deviation and the standard deviation of the standard deviation.

    The algorithm follows the ideas in https://doi.org/10.1063/1.457480 .
    '''
    sigma_m=[]
    err_sigma=[]
    blocked_chain=timeserie.copy() 

    for i in block_sizes:
        n=len(blocked_chain)
        c=np.var(blocked_chain, ddof=0)
        sigma_m.append(np.sqrt(c/(n-1)))
        err_sigma.append(sigma_m[-1]/np.sqrt(2*(n-1)))
        blocked_chain=np.add.reduceat(timeserie, np.arange(0, len(timeserie)//i*i, i))/i

    if inv_sigma_average:
        for i in range(len(sigma_m)):
            inverse_sigma_average = np.sum(np.array(sigma_m[i:])/np.array(err_sigma[i:])**2)/np.sum(1/np.array(err_sigma[i:])**2)
            err_inverse_sigma_average = 1/np.sqrt(np.sum(1/np.array(err_sigma[i:])**2))

            if (abs((sigma_m[i]-inverse_sigma_average))/(err_sigma[i]**2+err_inverse_sigma_average**2)**0.5)>1.96:
                #The value of the measure is too different from the mean of the following terms, we are not in the plateau yet
                continue
            return sigma_m, err_sigma, inverse_sigma_average, err_inverse_sigma_average
    else:
        return sigma_m, err_sigma
    
def arithmethic_average(log_likelihoods):
    """The average of the exponential of the log likelihoods."""
    avg = mp.fsum(vec_exp(log_likelihoods))/len(log_likelihoods)
    return avg, mp.log(avg)

def harmonic_mean(log_likelihoods):
    neg_vec_exp = lambda x: mp.exp(-x)
    neg_vec_exp = np.vectorize(neg_vec_exp)
    h_mean = len(log_likelihoods)/mp.fsum(neg_vec_exp(log_likelihoods))
    return h_mean, mp.log(h_mean)

def arithmethic_average_error(log_likelihoods, log_avg=False):
    """
    It returns the arithmetic average and an error estimate computed with the use of block analysis.
    Notice: the log results should not be considered as unbiased estimates, they are returned only for computational purposes.
    Notice: we considered autocorrelation times for the log likelihoods, not for the exponential of the log likelihoods.

    Output: (avg, standard deviation of the average), (logavg, logblock_err)
    """
    if log_avg:
        avg = mp.fsum(log_likelihoods)/len(log_likelihoods)
        logavg = mp.log(avg)
    else:
        avg, logavg = arithmethic_average(log_likelihoods)
    autocorr_times = autocorr2(log_likelihoods)
    
    for thr in np.arange(0.1,0.9,0.1):
        try:
            autocorr=np.argmax(autocorr_times<thr)
            block_sizes = np.unique(np.linspace(autocorr,len(log_likelihoods)/5, dtype=int))
            if log_avg:
                _, _, block_err, _ = block_error(log_likelihoods, block_sizes, inv_sigma_average=True)
            else:
                _, _, block_err, _ = block_error(vec_exp(log_likelihoods), block_sizes, inv_sigma_average=True)
            block_err_std = block_err*((len(log_likelihoods)-1)/2)**0.5*mp.exp(mp.loggamma(0.5*(len(log_likelihoods)-1))-mp.loggamma(0.5*len(log_likelihoods)))
            
            return (avg, block_err_std), (logavg, mp.log(block_err_std)), (block_err**2, mp.log(block_err**2)) 
        except:
            continue
    print('The block analysis did not converge')
    return (avg, 0), (logavg, 0), (0, 0)

def harmonic_mean_thinningerror(log_likelihoods, min_pt=20):
    """
    It returns the partition function computed with the harmonic average and an error estimate.
    Notice: the log results should not be considered as unbiased estimates, they are returned only for computational purposes.
    Thinning is used to reduce the autocorrelation time. The thinning is performed by taking one sample every autocorr time, considering
    the log likelihoods.
    Ideas from https://arxiv.org/pdf/2111.12720 , https://www.stat.cmu.edu/~hseltman/files/ratio.pdf

    Inputs:
    log_likelihoods: a one dimensional array of the logarithm of the values of which you want to compute the harmonic mean
    min_pt: is the minimal number of points in the thinned chain

    Outputs:
    (zhm, zhm_std): the harmonic mean and its standard deviation
    (log_zhm, log_zhm_std): the logarithm of the harmonic mean and the logarithm of its standard deviation
    (zhm_var, log_zhm_var): the variance of the harmonic mean and the logarithm of its variance
    (rho, var_rho): the value of rho and its variance estimated with thinnig procedure, these values can be used for the computation of the Bayes factor 
    (log_rho, log_var_rho): the logarithm of the value of rho and the log of its variance estimated with thinnig procedure
    """
    neg_vec_exp = lambda x: mp.exp(-x)
    neg_vec_exp = np.vectorize(neg_vec_exp)

    autocorr_times = autocorr2(vec_log(neg_vec_exp(log_likelihoods)))
    
    for thr in np.arange(0.1,0.9,0.1):
        autocorr=np.argmax(autocorr_times<thr)
        if len(log_likelihoods)/autocorr<min_pt:
            if thr==0.8:
                autocorr=max(1,int(len(log_likelihoods/min_pt)))
            else:    
                continue
        
        rho = mp.fsum(neg_vec_exp(log_likelihoods[::autocorr]))/len(log_likelihoods[::autocorr])
        
        mean = mp.fsum(vec_exp(log_likelihoods[::autocorr]))/len(log_likelihoods[::autocorr])
        var = mp.fsum(vec_exp(log_likelihoods[::autocorr]-mean), squared=True)/len(log_likelihoods[::autocorr]-1)
        
        h_mean_var = mean**-4 * var / len(log_likelihoods[::autocorr]) 
        
        zhm = rho**-1*(1+rho**-2*h_mean_var)
        zhm_std = h_mean_var**0.5 / mean**2 * ((len(log_likelihoods[::autocorr])-1)/2)**0.5*mp.exp(mp.loggamma(0.5*(len(log_likelihoods[::autocorr])-1))-mp.loggamma(0.5*len(log_likelihoods[::autocorr])))
        zhm_var = h_mean_var / mean**4

        return (zhm,zhm_std), (mp.log(zhm), mp.log(zhm_std)), (zhm_var, mp.log(zhm_var)), (rho, h_mean_var), (mp.log(rho), mp.log(h_mean_var))
    
def harmonic_mean_blockerror(log_likelihoods):
    """
    It returns the partition function computed with the harmonic average and an error estimate.
    Notice: the log results should not be considered as unbiased estimates, they are returned only for computational purposes.
    The error is computed with the use of block analysis.
    Notice: we considered autocorrelation times for the log likelihoods, not for the exponential of the log likelihoods.

    Output: (avg, standard deviation of the average), (logavg, logblock_err), (rho, var_rho), (mp.log(rho), mp.log(var_rho)) 
    """

    neg_vec_exp = lambda x: mp.exp(-x)
    neg_vec_exp = np.vectorize(neg_vec_exp)

    rho = mp.fsum(neg_vec_exp(log_likelihoods))/len(log_likelihoods)

    autocorr_times = autocorr2(vec_log(neg_vec_exp(log_likelihoods)))
    
    for thr in np.arange(0.1,0.9,0.1):
        try:
            autocorr=np.argmax(autocorr_times<thr)
            block_sizes = np.unique(np.linspace(autocorr,len(log_likelihoods)/5, dtype=int))
            _, _, block_err, _ = block_error(neg_vec_exp(log_likelihoods), block_sizes, inv_sigma_average=True)
            var_rho=block_err**2
            zhm = rho**-1*(1+var_rho/rho**2)   
            zhm_var = var_rho/rho**4         
            return (zhm, zhm_var), (mp.log(zhm), mp.log(zhm_var)), (rho, var_rho), (mp.log(rho), mp.log(var_rho)) 
        except:
            continue
    
    return (zhm, 0), (mp.log(zhm), 0)

def th_integration_error(log_z_beta, var_log_z, betas):
    """
    Use the trapezoidal rule to evaluate the weigth of each value of the logarithm.
    The weigths are used for the estimation of the variance of the integral.

    Outputs:
    (integral, var): the estimate of the integral and its variance
    """

    dx=np.diff(betas)
    a=np.array([dx[0]]+list(dx[1:]+dx[:-1])+[dx[-1]])/2

    integral=mp.fsum(a*log_z_beta)
    var=mp.fsum(a**2*var_log_z)

    return (integral, var)

def stepping_stone_error(rk, var_rk):
    """
    The function computes the evidence using the stepping stone method.
    The evidence is computed as the product of the ratios of the normalizing constants.
    The variance is computed as the variance of the products.
    """

    z=mp.fprod(rk)
    log_z = mp.fsum(vec_log(rk))

    var_z = mp.fprod(var_rk+rk**2)-mp.fprod(rk)**2

    return (z, var_z), (log_z, mp.log(var_z))

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
                try:
                    self.mus.append(self.numpy_randomGen.multivariate_normal(np.mean(self.data[self.zi==i], axis=0), np.cov(self.data[self.zi==i].T)))
                    self.Sigmas.append(np.cov(self.data[self.zi==i].T))
                except:
                    self.Sigmas.append(invwishart.rvs(self.nu, self.S, random_state=self.numpy_randomGen))
                    self.mus.append(self.numpy_randomGen.multivariate_normal(self.mu_0, 1/self.lambda_*self.Sigmas[-1]))
            self.mus=np.array(self.mus)


    def sample_pi(self, alpha, z, numpy_randomGen=None):
        """
        Sample the probability of each cluster. To be used in the gibbs step.
        """
        if numpy_randomGen is None:
            numpy_randomGen=self.numpy_randomGen
        for i in range(len(alpha)):
            alpha[i]+=np.count_nonzero(z==i)
        return dirichlet.rvs(alpha, random_state=numpy_randomGen)[0]


    def sample_z(self, features, mus, Sigmas, Pi, beta=1, numpy_randomGen=None):
        """
        Sample the cluster assignments for each data point. To be used in the gibbs step.
        """
        if numpy_randomGen is None:
            numpy_randomGen=self.numpy_randomGen
        p_z=np.zeros((len(features), len(mus)))
        for j in range(len(mus)):
            p_z[:,j]=(Pi[j]*(multivariate_normal.pdf(features, mean=mus[j], cov=Sigmas[j])**beta))
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
        for i in range(self.n_clusters):
            n = np.sum(z==i)

            lambda_n=lambda_+beta*n
            nu_n=nu+beta*n
            mu_n=(beta*np.sum(features[z==i], axis=0)+lambda_*mu_0)/lambda_n
            
            vector=lambda_*mu_0+beta*np.sum(features[z==i], axis=0)

            S_n= S - np.outer(vector, vector)/lambda_n + lambda_*np.outer(mu_0,mu_0) + beta*features[z==i].T@features[z==i]

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
            if np.sum(zi==i)!=0:
                l_likelihood+=mp.fsum(vec_log((multivariate_normal.pdf(self.data[zi==i], mean=mus[i], cov=Sigmas[i])).reshape(-1)))
        return l_likelihood

    
    def gibbs_step(self, beta=1):
        self.pi=self.sample_pi(self.alpha, self.zi)
        self.zi=self.sample_z(self.data, self.mus, self.Sigmas, self.pi, beta)
        self.mus, self.Sigmas=self.sample_mus_Sigmas(self.data, self.zi, self.mu_0, self.lambda_, self.S, self.nu, beta)

    
    def _MH_acceptance(self, z1, z2, mu_1, mu_2, Sigmas_1, Sigmas_2, beta1, beta2):
        """
        Metropolis-Hastings step to accept or reject the new parameters.
        """
        l_likelihood_1=self.compute_loglikelihood_(z1, mu_1, Sigmas_1)
        l_likelihood_2=self.compute_loglikelihood_(z2, mu_2, Sigmas_2)
        p=min(1, mp.exp((beta1-beta2)*(l_likelihood_2-l_likelihood_1)))
        if self.numpy_randomGen.random()<p:
            self.MH_acceptance[0]+=1
            self.MH_acceptance[1]+=1
            return True
        else:
            self.MH_acceptance[1]+=1
            return False 
        
    def _MH_step(self, dictionary, betas, starting=0):
        """
        Update the dictionary with the new values after performing a Metropolis-Hastings step.
        The Metropolis-Hastings step is performed between couples of consecutive temperatures.
        """
        for i in range(starting,len(betas)-1,2):
            acc=self._MH_acceptance(dictionary['zi_%.7f'%betas[i]][-1], dictionary['zi_%.7f'%betas[i+1]][-1], dictionary['mus_%.7f'%betas[i]][-1], dictionary['mus_%.7f'%betas[i+1]][-1], dictionary['Sigmas_%.7f'%betas[i]][-1], dictionary['Sigmas_%.7f'%betas[i+1]][-1], betas[i], betas[i+1])
            if acc:
                dictionary['zi_%.7f'%betas[i]]=np.vstack((dictionary['zi_%.7f'%betas[i]],dictionary['zi_%.7f'%betas[i+1]][-1].reshape(1, self.N)))
                dictionary['zi_%.7f'%betas[i+1]]=np.vstack((dictionary['zi_%.7f'%betas[i+1]],dictionary['zi_%.7f'%betas[i]][-2].reshape(1, self.N)))
                dictionary['mus_%.7f'%betas[i]]=np.vstack((dictionary['mus_%.7f'%betas[i]],dictionary['mus_%.7f'%betas[i+1]][-1].reshape(1, self.n_clusters, self.D)))
                dictionary['mus_%.7f'%betas[i+1]]=np.vstack((dictionary['mus_%.7f'%betas[i+1]],dictionary['mus_%.7f'%betas[i]][-2].reshape(1, self.n_clusters, self.D)))
                dictionary['Sigmas_%.7f'%betas[i]]=np.vstack((dictionary['Sigmas_%.7f'%betas[i]],dictionary['Sigmas_%.7f'%betas[i+1]][-1].reshape(1, self.n_clusters, self.D, self.D)))
                dictionary['Sigmas_%.7f'%betas[i+1]]=np.vstack((dictionary['Sigmas_%.7f'%betas[i+1]],dictionary['Sigmas_%.7f'%betas[i]][-2].reshape(1, self.n_clusters, self.D, self.D)))
                dictionary['loglikelihood_%.7f'%betas[i]]=dictionary['loglikelihood_%.7f'%betas[i]]+[dictionary['loglikelihood_%.7f'%betas[i+1]][-1]]
                dictionary['loglikelihood_%.7f'%betas[i+1]]=dictionary['loglikelihood_%.7f'%betas[i+1]]+[dictionary['loglikelihood_%.7f'%betas[i]][-2]]
            else:
                dictionary['zi_%.7f'%betas[i]]=np.vstack((dictionary['zi_%.7f'%betas[i]],dictionary['zi_%.7f'%betas[i]][-1].reshape(1, self.N)))
                dictionary['zi_%.7f'%betas[i+1]]=np.vstack((dictionary['zi_%.7f'%betas[i+1]],dictionary['zi_%.7f'%betas[i+1]][-1].reshape(1, self.N)))
                dictionary['mus_%.7f'%betas[i]]=np.vstack((dictionary['mus_%.7f'%betas[i]],dictionary['mus_%.7f'%betas[i]][-1].reshape(1, self.n_clusters, self.D)))
                dictionary['mus_%.7f'%betas[i+1]]=np.vstack((dictionary['mus_%.7f'%betas[i+1]],dictionary['mus_%.7f'%betas[i+1]][-1].reshape(1, self.n_clusters, self.D)))
                dictionary['Sigmas_%.7f'%betas[i]]=np.vstack((dictionary['Sigmas_%.7f'%betas[i]],dictionary['Sigmas_%.7f'%betas[i]][-1].reshape(1, self.n_clusters, self.D, self.D)))
                dictionary['Sigmas_%.7f'%betas[i+1]]=np.vstack((dictionary['Sigmas_%.7f'%betas[i+1]],dictionary['Sigmas_%.7f'%betas[i+1]][-1].reshape(1, self.n_clusters, self.D, self.D)))
                dictionary['loglikelihood_%.7f'%betas[i]]=dictionary['loglikelihood_%.7f'%betas[i]]+[dictionary['loglikelihood_%.7f'%betas[i]][-1]]
                dictionary['loglikelihood_%.7f'%betas[i+1]]=dictionary['loglikelihood_%.7f'%betas[i+1]]+[dictionary['loglikelihood_%.7f'%betas[i+1]][-1]]
            
        return dictionary
    
    def _update_dict_after_parallel(self, dictionary, results, betas):
        """
        Update the dictionary after the parallel step.
        """
        for i in range(len(betas)):
            dictionary['pi_%.7f'%betas[i]]=np.vstack([dictionary['pi_%.7f'%betas[i]],results[i][0]])
            dictionary['zi_%.7f'%betas[i]]=np.vstack([dictionary['zi_%.7f'%betas[i]],results[i][1]])
            dictionary['mus_%.7f'%betas[i]]=np.vstack([dictionary['mus_%.7f'%betas[i]],results[i][2]])
            dictionary['Sigmas_%.7f'%betas[i]]=np.concatenate([dictionary['Sigmas_%.7f'%betas[i]],results[i][3]], axis=0)
            dictionary['loglikelihood_%.7f'%betas[i]]=dictionary['loglikelihood_%.7f'%betas[i]]+results[i][4]
        return dictionary
    
    def gibbs_sampler(self):
        dictionary={'pi':self.pi, 'zi':self.zi, 'mus':np.array(self.mus).reshape(1,self.n_clusters,self.D), 'Sigmas':np.array(self.Sigmas).reshape(1,self.n_clusters,self.D,self.D)}

        for _ in tqdm(range(self.n_iter)):
            self.gibbs_step()
            dictionary['pi']=np.vstack([dictionary['pi'],self.pi])
            dictionary['zi']=np.vstack([dictionary['zi'],self.zi])
            dictionary['mus']=np.vstack([dictionary['mus'],np.array(self.mus).reshape(1,self.n_clusters,self.D)])
            dictionary['Sigmas']=np.concatenate([dictionary['Sigmas'],np.array(self.Sigmas).reshape(1,self.n_clusters,self.D,self.D)], axis=0)
        
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
            dictionary.update({'pi_%.7f'%i:self.pi.reshape(1,self.n_clusters), 'zi_%.7f'%i:self.zi.reshape(1,self.N), 'mus_%.7f'%i:self.mus.reshape(1,self.n_clusters, self.D), 'Sigmas_%.7f'%i:np.array(self.Sigmas).reshape(1,self.n_clusters,self.D,self.D), 'loglikelihood_%.7f'%i:[]})
        
        for iter in tqdm(range(self.n_iter)):
            for i in betas:
                pi_=self.sample_pi(self.alpha, dictionary['zi_%.7f'%i][-1], self.numpy_randomGen)
                dictionary['pi_%.7f'%i]=np.vstack([dictionary['pi_%.7f'%i],pi_])
                
                zi_=self.sample_z(self.data, dictionary['mus_%.7f'%i][-1], dictionary['Sigmas_%.7f'%i][-1], dictionary['pi_%.7f'%i][-1], i, self.numpy_randomGen)
                dictionary['zi_%.7f'%i]=np.vstack([dictionary['zi_%.7f'%i],zi_])

                mus_, Sigmas_=self.sample_mus_Sigmas(self.data, dictionary['zi_%.7f'%i][-1], self.mu_0, self.lambda_, self.S, self.nu, i, self.numpy_randomGen)
                dictionary['mus_%.7f'%i]=np.vstack([dictionary['mus_%.7f'%i],np.array(mus_).reshape(1,self.n_clusters,self.D)])
                dictionary['Sigmas_%.7f'%i]=np.concatenate([dictionary['Sigmas_%.7f'%i],np.array(Sigmas_).reshape(1,self.n_clusters,self.D,self.D)], axis=0)
                
                dictionary['loglikelihood_%.7f'%i].append(self.compute_loglikelihood_(zi_, mus_, Sigmas_))
            
            if iter%2==0:
                dictionary=self._MH_step(dictionary, betas, 0)
            else:
                dictionary=self._MH_step(dictionary, betas, 1)

        return dictionary
    
    def _vector_for_parallel(self, dictionary, betas):
        """"Take the last values in the dictionary and creates a list that can be used in the parallel step."""
        list_=[]
        for i in betas:
            buff_list=[]
            buff_list.append(dictionary['pi_%.7f'%i][-1].reshape(1,self.n_clusters))
            buff_list.append(dictionary['zi_%.7f'%i][-1].reshape(1,self.N))
            buff_list.append(dictionary['mus_%.7f'%i][-1].reshape(1,self.n_clusters,self.D))
            buff_list.append(dictionary['Sigmas_%.7f'%i][-1].reshape(1,self.n_clusters,self.D,self.D))
            list_.append(buff_list)
        
        return list_


    def _process_beta(self, i, list_, n_iter, randomGen):
        """
        Process the diffusion step for a given temperature i.
        """

        pi_=list_[0]
        zi_=list_[1]
        mus_=list_[2]
        Sigmas_=list_[3]        
        loglikelihood=[]

        for _ in range(n_iter):
            _pi_ = self.sample_pi(self.alpha, zi_[-1], randomGen)
            pi_=np.vstack([pi_,_pi_])
            
            _zi_ = self.sample_z(self.data, mus_[-1], Sigmas_[-1], pi_[-1], i, randomGen)
            zi_=np.vstack([zi_,_zi_])

            _mus_, _Sigmas_ = self.sample_mus_Sigmas(self.data, zi_[-1], self.mu_0, self.lambda_, self.S, self.nu, i, randomGen)
            mus_=np.vstack([mus_,np.array(_mus_).reshape(1,self.n_clusters,self.D)])
            Sigmas_=np.concatenate([Sigmas_,np.array(_Sigmas_).reshape(1,self.n_clusters, self.D, self.D)], axis=0)

            loglikelihood.append(self.compute_loglikelihood_(_zi_, _mus_, _Sigmas_))
            
        return (pi_[1:], zi_[1:], mus_[1:], Sigmas_[1:], loglikelihood)
    

    def parallel_diffusion_gibbs(self, betas=None, n_gibbs_iter=10, n_jobs=4, n_chains=10, n_iter=None):
        """
        Parallel version of the diffusion gibbs sampler.
        """
        if not(n_iter is None):
            self.n_iter=n_iter

        if betas is None:
            betas=np.linspace(0,1,n_chains)**(1/0.3)
        betas=np.sort(betas)

        if np.unique(['%.7f'%i for i in betas]).shape[0]!=len(betas):
            raise ValueError('The algorithm will not work, some temperatures are too similar')

        ss = np.random.SeedSequence(self.seed)
        child_seeds = ss.spawn(len(betas))
        streams = [np.random.default_rng(s) for s in child_seeds]

        dictionary={}
        for i in betas:
            self.init_params(0)
            dictionary.update({'pi_%.7f'%i:self.pi.reshape(1,self.n_clusters), 'zi_%.7f'%i:self.zi.reshape(1,self.N), 'mus_%.7f'%i:self.mus.reshape(1,self.n_clusters, self.D), 'Sigmas_%.7f'%i:np.array(self.Sigmas).reshape(1,self.n_clusters,self.D,self.D), 'loglikelihood_%.7f'%i:[]})

        for iter in tqdm(range(self.n_iter)):
            
            list_=self._vector_for_parallel(dictionary, betas)
            results=Parallel(n_jobs=n_jobs)(delayed(self._process_beta)(betas[i], list_[i], n_gibbs_iter, streams[i]) for i in range(len(betas)))                
            
            dictionary=self._update_dict_after_parallel(dictionary, results, betas)

            if iter%2==0:
                dictionary=self._MH_step(dictionary, betas, 0)
            else:
                dictionary=self._MH_step(dictionary, betas, 1)

        n_count=[]
        for i in betas:
            dictionary['loglikelihood_%.7f'%i]=np.array(dictionary['loglikelihood_%.7f'%i])
            n_count.append(len(dictionary['loglikelihood_%.7f'%i]))

        if 0 in betas:
            z_avg, log_z_avg = arithmethic_average(dictionary['loglikelihood_%.7f'%0])
        if 1 in betas:
            z_hmean, log_z_hmean = harmonic_mean(dictionary['loglikelihood_%.7f'%1])

        thermodynamic_averages = [np.average(dictionary['loglikelihood_%.7f'%i]) for i in betas]
        th = (trapz(thermodynamic_averages, betas), simpson(thermodynamic_averages, betas))

        
        r_k=[1/n_count[i-1]*mp.fsum(vec_exp(dictionary['loglikelihood_%.7f'%betas[i-1]]*(betas[i]-betas[i-1]))) for i in range(1,len(betas))]
        stepping_stone = (mp.fprod(r_k), mp.fsum(vec_log(r_k)))
        
        return dictionary, (z_avg, log_z_avg), (z_hmean, log_z_hmean), thermodynamic_averages, th, r_k, stepping_stone

    def _process_beta_for_evidence(self, i, list_, n_iter, randomGen):
        """
        Process the diffusion step for a given temperature i.
        """

        pi_=list_[0]
        zi_=list_[1]
        mus_=list_[2]
        Sigmas_=list_[3]        
        loglikelihood=[]

        for _ in range(n_iter):
            pi_[-1] = self.sample_pi(self.alpha, zi_[-1], randomGen)
            
            zi_[-1] = self.sample_z(self.data, mus_[-1], Sigmas_[-1], pi_[-1], i, randomGen)

            mus_[-1], Sigmas_[-1] = self.sample_mus_Sigmas(self.data, zi_[-1], self.mu_0, self.lambda_, self.S, self.nu, i, randomGen)
            
            loglikelihood.append(self.compute_loglikelihood_(zi_[-1], mus_[-1], Sigmas_[-1]))

        return (pi_, zi_, mus_, Sigmas_, loglikelihood)
    
    def _update_dict_evidence(self, dictionary, results, betas):
        """
        Update the dictionary after the parallel step.
        """
        for i in range(len(betas)):
            dictionary['pi_%.7f'%betas[i]]=results[i][0]
            dictionary['zi_%.7f'%betas[i]]=results[i][1]
            dictionary['mus_%.7f'%betas[i]]=results[i][2]
            dictionary['Sigmas_%.7f'%betas[i]]=results[i][3]
            dictionary['loglikelihood_%.7f'%betas[i]]=dictionary['loglikelihood_%.7f'%betas[i]]+results[i][4]
        return dictionary
    
    def compute_evidence(self, betas=None, n_gibbs_iter=10, n_jobs=4, n_chains=10, n_iter=None, burnin=None):
        """
        Parallel version of the diffusion gibbs sampler that does not store all the samples of mus, Sigmas and pi (it still keeps the log likelihoods).
        This should be useful for memory purposes.
        If burnin is setted, only the samples after the burnin will be considered for the estimation of the evidences.
        """
        if not(n_iter is None):
            self.n_iter=n_iter

        if burnin is None:
            burnin=self.n_iter*n_gibbs_iter//5

        if betas is None:
            betas=np.linspace(0,1,n_chains)**(1/0.3)
        betas=np.sort(betas)

        if np.unique(['%.7f'%i for i in betas]).shape[0]!=len(betas):
            raise ValueError('The algorithm will not work, some temperatures are too similar')

        ss = np.random.SeedSequence(self.seed)
        child_seeds = ss.spawn(len(betas))
        streams = [np.random.default_rng(s) for s in child_seeds]

        dictionary={}
        for i in betas:
            self.init_params(0)
            dictionary.update({'pi_%.7f'%i:self.pi.reshape(1,self.n_clusters), 'zi_%.7f'%i:self.zi.reshape(1,self.N), 'mus_%.7f'%i:self.mus.reshape(1,self.n_clusters, self.D), 'Sigmas_%.7f'%i:np.array(self.Sigmas).reshape(1,self.n_clusters,self.D,self.D), 'loglikelihood_%.7f'%i:[]})

        for iter in tqdm(range(self.n_iter)):
            
            list_=self._vector_for_parallel(dictionary, betas)
            results=Parallel(n_jobs=n_jobs)(delayed(self._process_beta_for_evidence)(betas[i], list_[i], n_gibbs_iter, streams[i]) for i in range(len(betas)))                
            
            dictionary=self._update_dict_evidence(dictionary, results, betas)

            if iter%2==0:
                dictionary=self._MH_step(dictionary, betas, 0)
                
            else:
                dictionary=self._MH_step(dictionary, betas, 1)

        dict_output={'loglikelihood_%.7f'%i:dictionary['loglikelihood_%.7f'%i] for i in betas}

        for i in betas:
            dictionary['loglikelihood_%.7f'%i]=np.array(dictionary['loglikelihood_%.7f'%i][burnin:])
            dictionary['loglikelihood_%.7f'%i]=np.array(dictionary['loglikelihood_%.7f'%i][dictionary['loglikelihood_%.7f'%i]!=mp.mpf('-inf')])

        if 0 in betas:
            (z_avg, z_avg_error), (log_z_avg, log_z_avg_error), (var_z_avg, log_var_zavg) = arithmethic_average_error(dictionary['loglikelihood_%.7f'%0])
        print('Using the arithmetic mean method we get the following results:', z_avg, z_avg_error, log_z_avg, log_z_avg_error, var_z_avg, log_var_zavg)
        if 1 in betas:
            (z_hmean, z_hmean_var), (log_zhm, log_zhm_var), (rho, var_rho), (log_rho, log_var_rho) = harmonic_mean_blockerror(dictionary['loglikelihood_%.7f'%1])
        print('Using the harmonic mean method we get the following results:', z_hmean, z_hmean_var, log_zhm, log_zhm_var, rho, var_rho, log_rho, log_var_rho)

        thermodynamic_averages = []
        thermodynamic_averages_error = []
        for i in betas:
            (z__,z__std), (log_z, log_z_std), (z__var, log_z_var)=arithmethic_average_error(dictionary['loglikelihood_%.7f'%i], log_avg=True)
            thermodynamic_averages.append(z__)
            thermodynamic_averages_error.append(z__var)
        th_int_res = th_integration_error(thermodynamic_averages, thermodynamic_averages_error, betas)
        print('Using thermodynamic integration we get the following results:', th_int_res)

        r_k=[]
        var_rk=[]
        for i in range(1,len(betas)):
            (z_rk,z__std_rk), (log_z_rk, log_z_std_rk), (z__var_rk, log_z_var_rk)=arithmethic_average_error(dictionary['loglikelihood_%.7f'%betas[i-1]]*(betas[i]-betas[i-1]))
            r_k.append(z_rk)
            var_rk.append(z__var_rk)
        r_k=np.array(r_k)
        var_rk=np.array(var_rk)
        stepping_stone_res, log_stepping_stone = stepping_stone_error(r_k, var_rk)
        print('Using the stepping stone method we get the following results:', stepping_stone_res, log_stepping_stone)
        
        return dict_output, (log_z_avg, log_z_avg_error), (log_zhm, log_zhm_var), (log_rho, log_var_rho), th_int_res, log_stepping_stone
