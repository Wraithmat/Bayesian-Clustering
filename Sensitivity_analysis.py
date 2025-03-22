import numpy as np
from scipy.stats import invwishart, dirichlet
from diffusion_gibbs import DiffusionGibbs
from Bayes_factor import BF, log_BF, harmonic_BF
import mpmath as mp

seed = 0
null_randomGen = np.random.default_rng(seed)

def fiducial_values(alpha, mu_0, lambda_, S, nu, numpy_randomGen=null_randomGen):
    Pi=dirichlet.rvs(alpha, random_state=numpy_randomGen)
    Sigmas=[]
    mus=[]
    for i in range(len(alpha)):
        Sigmas.append(invwishart.rvs(nu, S, random_state=numpy_randomGen))
        mus.append(numpy_randomGen.multivariate_normal(mu_0, 1/lambda_*Sigmas[-1]))
    return Pi, mus, Sigmas

def generate_data(mus, Sigmas, K=3, assignements=[0]*300+[1]*300+[2]*300, numpy_randomGen=null_randomGen):   
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

    

    # We start analysing the result for different separation of the dataset -> different lambda
    BF_res=np.zeros((4,4,5,5)) 
    BF_res_err=np.zeros((4,4,5,5))

    iter_=-1
    for lambda_ in [0.025, 0.01, 0.5, 1]:
        iter_+=1
        alpha = [1, 2, 1.5]
        mu_0 = [0, 0]
        S = np.eye(len(mu_0)) 
        nu = 5
        numpy_randomGen = np.random.default_rng(seed)

        Pi, mus, Sigmas = fiducial_values(alpha, mu_0, lambda_, S, nu, numpy_randomGen)
        assignements = numpy_randomGen.choice(len(alpha), 50, p=Pi[0])
        data = generate_data(mus, Sigmas, assignements=assignements, numpy_randomGen=numpy_randomGen)

        alpha=[1,2,1.5,1,1]+[1]*10  
        z_avgs=[]
        var_z_avgs=[]
        #for the harmonic mean estimate
        rho=[]
        var_rho=[]
        #for the thermodynamic integration estimate
        log_z_thints=[]
        var_log_z_thints=[]
        #for the stepping stone estimate
        z_stepping_stones=[]
        var_z_stepping_stones=[]
        n_chains=25
        n_gibbs_iter=20
        burnin=200
        n_iter=50

        for i in range(1,6):
            print('Computing evidence for {} clusters'.format(i))
            sampler=DiffusionGibbs(data, alpha=alpha[:i], hot_start=1, n_iter=n_iter, n_clusters_0=i)
            
            sampler.init_params(0)
            print(np.unique(sampler.zi))
            _, (log_z_avg, log_z_avg_error), _, (log_rho, log_var_rho), th_int_res, log_stepping_stones = sampler.compute_evidence(n_chains=20, n_gibbs_iter=n_gibbs_iter)
            z_avgs.append(mp.exp(log_z_avg))
            var_z_avgs.append(mp.exp(log_z_avg_error))
            rho.append(mp.exp(log_rho))
            var_rho.append(mp.exp(log_var_rho))
            log_z_thints.append(th_int_res[0])
            var_log_z_thints.append(th_int_res[1])
            z_stepping_stones.append(mp.exp(log_stepping_stones[0]))
            var_z_stepping_stones.append(mp.exp(log_stepping_stones[1]))
        
        Bayes_factor=np.zeros((4,5,5)) # 4 methods, 5 models, 5 comparisons (oss: you should consider 1 for elements[:,i,i])
        Bayes_error=np.zeros((4,5,5))
        for i in range(5):
            for j in range(5):
                b, eb= BF(z_avgs[i], var_z_avgs[i], z_avgs[j], var_z_avgs[j])
                Bayes_factor[0,i,j]=b
                Bayes_error[0,i,j]=eb
                b, eb= harmonic_BF(mp.mpf(str(rho[i])), mp.mpf(str(var_rho[i])), mp.mpf(str(rho[j])), mp.mpf(str(var_rho[j])))
                Bayes_factor[1,i,j]=b
                Bayes_error[1,i,j]=eb
                b, eb= log_BF(log_z_thints[i], var_log_z_thints[i], log_z_thints[j], var_log_z_thints[j])
                Bayes_factor[2,i,j]=b
                Bayes_error[2,i,j]=eb
                b, eb= BF(z_stepping_stones[i], var_z_stepping_stones[i], z_stepping_stones[j], var_z_stepping_stones[j])
                Bayes_factor[3,i,j]=b
                Bayes_error[3,i,j]=eb
        
        BF_=Bayes_factor
        BF_err=Bayes_error
        import mpmath as mp
        vec_exp=np.vectorize(mp.exp)

        for i in [0,1,3]:
            BF_err[i]=vec_exp(-BF_[i]*2)*vec_exp(BF_err[i])
        
        BF_res[iter_]=BF_
        BF_res_err[iter_]=BF_err
    
    np.save('BF_res_lambda_change', BF_res)
    np.save('BF_res_err_lambda_change', BF_res_err)

    # We now analyse what happens when changing the prior

    BF_res=np.zeros((9,4,5,5)) 
    BF_res_err=np.zeros((9,4,5,5))

    data_ = np.load('../data_new.npy')
    data = data_[::20,:-1]

    iter_=-1
    for sigma_xy in [-0.25, 0.0, 0.5]:
        for nu in [1, 5, 10]:
            iter_+=1
            
            alpha=[1,2,1.5,1,1]+[1]*10  
            z_avgs=[]
            var_z_avgs=[]
            #for the harmonic mean estimate
            rho=[]
            var_rho=[]
            #for the thermodynamic integration estimate
            log_z_thints=[]
            var_log_z_thints=[]
            #for the stepping stone estimate
            z_stepping_stones=[]
            var_z_stepping_stones=[]
            n_chains=25
            n_gibbs_iter=20
            burnin=200
            n_iter=50

            for i in range(1,6):
                print('Computing evidence for {} clusters'.format(i))
                sampler=DiffusionGibbs(data, alpha=alpha[:i], S=np.eye(2)+np.array([[0,sigma_xy],[sigma_xy,0]]), nu=nu, hot_start=1, n_iter=n_iter, n_clusters_0=i)
                
                sampler.init_params(0)
                print(np.unique(sampler.zi))
                _, (log_z_avg, log_z_avg_error), _, (log_rho, log_var_rho), th_int_res, log_stepping_stones = sampler.compute_evidence(n_chains=20, n_gibbs_iter=n_gibbs_iter)
                z_avgs.append(mp.exp(log_z_avg))
                var_z_avgs.append(mp.exp(log_z_avg_error))
                rho.append(mp.exp(log_rho))
                var_rho.append(mp.exp(log_var_rho))
                log_z_thints.append(th_int_res[0])
                var_log_z_thints.append(th_int_res[1])
                z_stepping_stones.append(mp.exp(log_stepping_stones[0]))
                var_z_stepping_stones.append(mp.exp(log_stepping_stones[1]))
            
            Bayes_factor=np.zeros((4,5,5)) # 4 methods, 5 models, 5 comparisons (oss: you should consider 1 for elements[:,i,i])
            Bayes_error=np.zeros((4,5,5))
            for i in range(5):
                for j in range(5):
                    b, eb= BF(z_avgs[i], var_z_avgs[i], z_avgs[j], var_z_avgs[j])
                    Bayes_factor[0,i,j]=b
                    Bayes_error[0,i,j]=eb
                    b, eb= harmonic_BF(mp.mpf(str(rho[i])), mp.mpf(str(var_rho[i])), mp.mpf(str(rho[j])), mp.mpf(str(var_rho[j])))
                    Bayes_factor[1,i,j]=b
                    Bayes_error[1,i,j]=eb
                    b, eb= log_BF(log_z_thints[i], var_log_z_thints[i], log_z_thints[j], var_log_z_thints[j])
                    Bayes_factor[2,i,j]=b
                    Bayes_error[2,i,j]=eb
                    b, eb= BF(z_stepping_stones[i], var_z_stepping_stones[i], z_stepping_stones[j], var_z_stepping_stones[j])
                    Bayes_factor[3,i,j]=b
                    Bayes_error[3,i,j]=eb
            
            BF_=Bayes_factor
            BF_err=Bayes_error
            import mpmath as mp
            vec_exp=np.vectorize(mp.exp)

            for i in [0,1,3]:
                BF_err[i]=vec_exp(-BF_[i]*2)*vec_exp(BF_err[i])
            
            BF_res[iter_]=BF_
            BF_res_err[iter_]=BF_err


    np.save('BF_res_prior_change', BF_res)
    np.save('BF_res_err_prior_change', BF_res_err)

