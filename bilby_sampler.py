import bilby
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invwishart, dirichlet, multivariate_normal
import mpmath as mp

bilby.core.utils.logger.setLevel("ERROR")

samplers = dict(
    bilby_mcmc=dict(
        nsamples=100,
        L1steps=20,
        ntemps=10,
        printdt=10,
        outdir='outdir2'),
    dynesty=dict(npoints=10, sample="acceptance-walk", naccept=20),
    #pymultinest=dict(nlive=500),
    #nestle=dict(nlive=500),
    #emcee=dict(nwalkers=20, iterations=500),
    #ptemcee=dict(ntemps=10, nwalkers=20, nsamples=1000),
)

results = dict()



class Normal_Inverse_Wishart(bilby.core.prior.joint.BaseJointPriorDist):
    """Define a new prior class for the normal inverse wishart distribution
    A sample from this distribution is an array concatenating mu and a flattened sigma
    """

    def __init__(self, nu, S, lambda_, mu_0, bounds=None, names=None):
        super(Normal_Inverse_Wishart, self).__init__(
            names=names, bounds=bounds
        )
        self.nu = nu
        self.S = S
        self.lambda_ = lambda_
        self.mu_0 = mu_0

    def _sample(self, size=1):
        sigma = invwishart.rvs(df=self.nu, scale=self.S, size=size)
        if size==1:
            mu = np.array([multivariate_normal.rvs(mean=self.mu_0, cov=sigma/self.lambda_)])
            return np.hstack((mu.reshape(1,2), sigma.reshape(1,4)))
        else:
            mu = np.array([multivariate_normal.rvs(mean=self.mu_0, cov=sigma[i]/self.lambda_) for i in range(size)])
            return np.concatenate((mu, sigma.reshape(-1,4)), axis=1)

    #def prob(self, val):
    #    p=[]
    #    if len(val.shape)==1:
    #        p_mu = multivariate_normal.pdf(val[:2], mean=self.mu_0, cov=self.S/self.lambda_)
    #        p_sigma = invwishart.pdf(val[2:].reshape(2,2), df=self.nu, scale=self.S)
    #        return p_mu*p_sigma
    #    for x in val:
    #        p_mu = multivariate_normal.pdf(x[:2], mean=self.mu_0, cov=self.S/self.lambda_)
    #        p_sigma = invwishart.pdf(x[2:].reshape(2,2), df=self.nu, scale=self.S)
    #        p.append(p_mu*p_sigma)
    #    return np.array(p)
    
    def _ln_prob(self, val, lnprob, outbounds):
        p=[]
        if len(val.shape)==1:
            p_mu = multivariate_normal.logpdf(val[:2], mean=self.mu_0, cov=self.S/self.lambda_)
            p_sigma = invwishart.logpdf(val[2:].reshape(2,2), df=self.nu, scale=self.S)
            return p_mu+p_sigma
        for x in val:
            p_mu = multivariate_normal.logpdf(x[:2], mean=self.mu_0, cov=self.S/self.lambda_)
            p_sigma = invwishart.logpdf(x[2:].reshape(2,2), df=self.nu, scale=self.S)
            p.append(p_mu+p_sigma)
        return np.array(p)



class Dirichlet(bilby.core.prior.joint.BaseJointPriorDist):
    """Define a new prior class for the dirichlet distribution
    A sample from this distribution is an array of shape (n_samples, D) where D is the number of dimensions
    """

    def __init__(self, alpha, bounds=None, names=None):
        super(Dirichlet, self).__init__(
            names=names, bounds=bounds
        )
        self.alpha = alpha

    def _sample(self, size=1):
        return dirichlet.rvs(self.alpha, size=size)

    #def prob(self, val):
    #    assert(val.shape == (3,))
    #    return dirichlet.pdf(val, self.alpha)
    
    def _ln_prob(self, samp, lnprob, outbounds):
        for j in range(samp.shape[0]):
            if (not outbounds[j]) and (np.isclose(np.sum(samp[j]),1)):
                lnprob[j] = dirichlet.logpdf(samp[j], self.alpha)
        return lnprob
    

class data_likelihood(bilby.Likelihood):
    def __init__(self, data, n_clusters, D):
        """
        data: array_like
            The data to analyse
        """

        labels = [f"mu_{i}" for i in ['x','y']]*3 + \
                    [f"sigma_{i}{j}" for j in ['x','y'] for i in ['x','y']]*3 + \
                    [f"dirichlet_{i}" for i in range(1, 4)] 
        for j in range(3):
            for i in range(2):
                    labels[2*j+i]+=f"_{j}"
                    labels[6+4*j+i]+=f"_{j}"
                    labels[8+4*j+i]+=f"_{j}"
        self._labels = labels
        super().__init__(parameters={el: None for el in labels})

        self.data = data
        self.N = len(data)
        self.n_clusters = n_clusters
        self.D = D

    def log_likelihood(self):
    #def compute_likelihood(data, N_clusters, N_dim, params):
        """
        Returns the log likelihood for you dataset.
        """
        N_clusters=self.n_clusters
        N_dim=self.D
        labels = self._labels
        params=np.array([self.parameters[el] for el in labels])
        pi=params[-3:]
        #pi=np.hstack([pi,1-pi.sum()])
        assert(np.isclose(pi.sum(),1))
        mus=params[:N_clusters*N_dim].reshape(N_clusters,N_dim)
        Sigmas=params[N_clusters*N_dim:N_clusters*N_dim+N_clusters*N_dim**2].reshape(N_clusters,N_dim,N_dim)
        probabilities = np.array([multivariate_normal.pdf(data, mean=mus[i], cov=Sigmas[i]) for i in range(N_clusters)]).T*pi
        vec_log = lambda x: mp.log(x)
        vec_log = np.vectorize(vec_log)
        log_likelihood = mp.fsum(vec_log([mp.fsum(row) for row in probabilities]))
        #log_likelihood = np.sum(np.log(np.sum(probabilities, axis=1)))
        #if type(log_likelihood)==mp.ctx_mp_python.mpc:
        #    print(params, self.parameters,np.min(([mp.fsum(row) for row in probabilities])) )
        return np.float64(log_likelihood)


class mu_sigma(bilby.core.prior.joint.JointPrior):
    def __init__(self, dist, name=None, unit=None):
        super(mu_sigma, self).__init__(
            dist=dist, name=name, unit=unit
        )

class pi(bilby.core.prior.joint.JointPrior):
    def __init__(self, dist, name=None,  unit=None):
        super(pi, self).__init__(
            dist=dist, name=name, unit=unit
        )


if __name__=='__main__':
    data_=np.load('./data/data_new.npy')
    data=data_[:,:-1]
    true_labels=data_[:,-1]

    likelihood = data_likelihood(data, 3, 2)

    labels = [f"mu_{i}" for i in ['x','y']]*3 + \
            [f"sigma_{i}{j}" for j in ['x','y'] for i in ['x','y']]*3 + \
            [f"pi_{i}" for i in range(1, 4)]  
    for j in range(3):
        for i in range(2):
            labels[2*j+i]+=f"_{j}"
            labels[6+4*j+i]+=f"_{j}"
            labels[8+4*j+i]+=f"_{j}"
    print

    n_dim = 3
    #label = "dirichlet_"
    #priors = bilby.core.prior.DirichletPriorDict(n_dim=n_dim, label=label)
    #print(priors.keys())

    bounds_dir={f"dirichlet_{i}": [0,1] for i in range(1,4)}
    Dir = Dirichlet([1,2,1.5], bounds=bounds_dir, names=[f"dirichlet_{i}" for i in range(1,4)])
    Niw1 = Normal_Inverse_Wishart(5, np.eye(2), 1, np.zeros(2), names=["mu_x_0", "mu_y_0", 'sigma_xx_0','sigma_yx_0','sigma_xy_0','sigma_yy_0'] )
    Niw2 = Normal_Inverse_Wishart(5, np.eye(2), 1, np.zeros(2), names=["mu_x_1", "mu_y_1", 'sigma_xx_1','sigma_yx_1','sigma_xy_1','sigma_yy_1'] )
    Niw3 = Normal_Inverse_Wishart(5, np.eye(2), 1, np.zeros(2), names=["mu_x_2", "mu_y_2", 'sigma_xx_2','sigma_yx_2','sigma_xy_2','sigma_yy_2'] )

    priors=dict()
    priors.update({f"dirichlet_{i}": pi(Dir, f"dirichlet_{i}")  for i in range(1,4)})
    priors.update({el: mu_sigma(Niw1, el)  for el in ["mu_x_0", "mu_y_0", 'sigma_xx_0','sigma_yx_0','sigma_xy_0','sigma_yy_0']})
    priors.update({el: mu_sigma(Niw2, el)  for el in ["mu_x_1", "mu_y_1", 'sigma_xx_1','sigma_yx_1','sigma_xy_1','sigma_yy_1']})
    priors.update({el: mu_sigma(Niw3, el)  for el in ["mu_x_2", "mu_y_2", 'sigma_xx_2','sigma_yx_2','sigma_xy_2','sigma_yy_2']})

    for sampler in samplers:
        print(f"Running {sampler}")
        result = bilby.core.sampler.run_sampler(
            likelihood,
            priors=priors,
            sampler=sampler,
            label=sampler,
            resume=False,
            clean=True,
            verbose=True,
            **samplers[sampler]
        )
        results[sampler] = result
        print(results[sampler])
        #except Exception as e:
        #    print(f"Failed to run {sampler}")
        #    print(e)
        #    results[sampler] = None

    print("-" * 40)
    for sampler in results:
        print(sampler)
        print("-" * 40)
        print(results[sampler])
        print("-" * 40)
