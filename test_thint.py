from diffusion_gibbs import DiffusionGibbs
import numpy as np

if __name__=='__main__':
    data_=np.load('./data/data_new.npy')
    data=data_[:,:-1]
    true_labels=data_[:,-1]
    sampler=DiffusionGibbs(data, alpha=[1,2,1.5], hot_start=0, n_iter=10)
    sampler.gibbs_sampler()
