In this repository we collect the code to analyse a Gaussian Mixture Model.

In tutorial.ipynb an example on how to use the different parts of the codes is shown.

Here a brief description of each file:
- data_generation.py can be run to create a dataset to analyse
- diffusion_gibbs.py contains the elements of the class which is used in the tutorial to perform Gibbs Sampling, error estimation and model selection
- Dirichlet_sampling.py contains the elements of the class used in the tutorial to analyse the Dirichlet Process
- bilby_sampler.py contains an implementation of the code with bilby library, though, the presence of NIW prior makes the implementation extremely slow

The directories contain the following information:
- data
