from diffusion_gibbs import DiffusionGibbs
import numpy as np
import mpmath as mp

def BF(z1, var_z1, z2, var_z2):
    """Bayes factor using the delta method. 
    It returns the logarithm of the estimate and the logarithm of an estimate of the variance.
    """
    z1=mp.mpf(z1)
    var_z1=mp.mpf(var_z1)
    z2=mp.mpf(z2)
    var_z2=mp.mpf(var_z2)
    mean=z1/z2+var_z2*z1/z2**3
    var=(z1/z2)**2*(var_z1/z1**2+var_z2/z2**2)
    return mp.log(mean), mp.log(var)

def log_BF(log_z1, var_log_z1, log_z2, var_log_z2):
    mean=log_z1-log_z2
    var=var_log_z1+var_log_z2
    return mean, var

def harmonic_BF(rho1, var_rho1, rho2, var_rho2):
    rho1=mp.mpf(rho1)
    var_rho1=mp.mpf(var_rho1)
    rho2=mp.mpf(rho2)
    var_rho2=mp.mpf(var_rho2)
    log_mean, log_var = BF(rho2, var_rho2, rho1, var_rho1)
    return log_mean, log_var

if __name__ == '__main__':
    data_=np.load('./data/data_new.npy')
    data=data_[:,:-1]
    true_labels=data_[:,-1]
    alpha=[1,2,1.5,1,1]

    #for the arithmetic average estimate
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
    n_chains=20
    n_gibbs_iter=15
    burnin=200
    n_iter=600


    for i in [1,2,3,4,5]:
        print('Computing evidence for {} clusters'.format(i))
        sampler=DiffusionGibbs(data, alpha=alpha[:i], hot_start=0, n_iter=n_iter, n_clusters_0=i)
        
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
            try: 
                Bayes_error[1,i,j]=eb
            except:
                print('Here is the problem:' ,i,j,'\n',rho[i], var_rho[i], rho[j], var_rho[j], b, eb)

            b, eb= log_BF(log_z_thints[i], var_log_z_thints[i], log_z_thints[j], var_log_z_thints[j])
            Bayes_factor[2,i,j]=b
            Bayes_error[2,i,j]=eb

            b, eb= BF(z_stepping_stones[i], var_z_stepping_stones[i], z_stepping_stones[j], var_z_stepping_stones[j])
            Bayes_factor[3,i,j]=b
            Bayes_error[3,i,j]=eb

        print(f'Bayes factor for z_{i}/z_j')
        print([f'method:{l}, {i},{j} : {Bayes_factor[l,i,j]}+/-{ Bayes_error[l,i,j]}' for (l,j) in zip(np.arange(4*4)//4,np.arange(4*4)%4)])

    np.save('./data/Bayes_factor.npy', Bayes_factor)
    np.save('./data/Bayes_error.npy', Bayes_error)