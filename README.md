# Gaussian Mixture Model Analysis

This repository contains code and resources for analyzing Gaussian Mixture Models (GMMs). It includes methods for data generation, Gibbs sampling, Diffusive Gibbs sampling, error estimation, model selection, and Dirichlet Process analysis. 

In order to start exploring the repository, you can check **tutorial.ipynb** and the animations in the `Visualizations` folder.

## 📁 Repository Structure

- **tutorial.ipynb**: A comprehensive example demonstrating the usage of the code components.

### Core Scripts
- **data_generation.py**: Generates datasets for GMM analysis.
- **diffusion_gibbs.py**: Implements the Diffusion Gibbs Sampling class, used for:
  - Gibbs Sampling
  - Error estimation
  - Model selection
- **Dirichlet_sampling.py**: Contains the class for analyzing the Dirichlet Process.
- **bilby_sampler.py**: An implementation using the `bilby` library. Note: The use of a NIW prior significantly slows down this method.
- **nested_sampling.py**: An implementation of nested sampling with the rejection scheme and stopping criteria.
- **Sensitivity_analysis.py** and **Bayes_factor.py** contain some code to compute Bayes factors and their errors.
- **gibbs_sampler.py**: a plain and simple implementation of gibbs sampling, used to profile and optimize the functions.

### Directory Overview
- **data/**: Contains generated datasets.
- **logs/**: Metadata from optimization processes, primarily used during development.
- **outdir/**: Stores result visualizations from the `bilby_sampler`.
- **submit/**: Bash scripts for running nested sampling.
- **Visualizations/**: Contains the code for visualizations and sample GIFs.

## 🔍 Additional Information
- The `diffusion_gibbs.py` and `Dirichlet_sampling.py` modules are designed for flexibility and can be adapted for further applications.
- The `bilby_sampler` gives troubles because of the N

## Some references
Part of the code is inspired by
- F. Feroz, M.P. Hobson and M. Bridges, “MULTINEST: an efficient and robust Bayesian inference tool for cosmology and particle physics". Mon. Not. R. Astron. Soc., 2008​
- Patricio M. R. , et al. “The stepping-stone sampling algorithm for calculating the evidence of gravitational wave models”, 2018​
- Wangang Xie, et al. “Improving Marginal Likelihood Estimation for Bayesian Phylogenetic Model Selection”, Syst. Biol. 60(2):150-160, 2011​
- Radford M. Neal, “Sampling from multimodal distributions using tempered transitions”, Statistics and Computing (1996)​
- Jason D. McEwen, et al., “Machine learning assisted Bayesian model comparison: learnt harmonic mean estimator”, 2023​
- Flyvbjerg et al., “Error estimates on averages of correlated data”, J. Chem. Phys. 91, 461–466 (1989)​
- Wolff, Ulli (1989-01-23). "Collective Monte Carlo Updating for spin Systems". Physical Review Letters
- Ashton, Gregory and others, "BILBY: A user-friendly Bayesian inference library for gravitational-wave astronomy", Astrophys. J. Suppl.    ​

## 📄 License
This project is licensed.

