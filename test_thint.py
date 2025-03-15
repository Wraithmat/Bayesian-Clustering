from diffusion_gibbs import DiffusionGibbs
import numpy as np

if __name__=='__main__':
    data_=np.load('./data/data_new.npy')
    data=data_[:,:-1]
    true_labels=data_[:,-1]
    sampler=DiffusionGibbs(data, alpha=[1,2,1.5], hot_start=0, n_iter=200)
    dictionary, (z_avg, log_z_avg), (z_hmean, log_z_hmean), thermodynamic_averages, th, r_k, stepping_stone =sampler.parallel_diffusion_gibbs(n_chains=20, n_jobs=-1, n_gibbs_iter=75)
    np.save('dictionary.npy', dictionary)
    print(log_z_avg, log_z_hmean, thermodynamic_averages, th, r_k, stepping_stone)