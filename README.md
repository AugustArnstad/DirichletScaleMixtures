# DirichletScaleMixtures
This repository contains the code accompanying the article "Dirichlet Scale Mixture priors for BNNs". This repo is a distilled version of the full code repo 'shrinkage_priors_in_BNNs', to make it easier for interested people to see how it works.

## Overview

The repository includes:

- Model definitions (Stan files)
- Data loading utilities
- A unified model runner
- Example notebook for running experiments

Repository structure
```perl
utils
  ├── model_runner.py           # Main inference logic  
  ├── model_loader.py           # Main retrieving 
  ├── load_abalone.py           # Data loading utilities  
  ├── iohelpers.py              # Small helpers  
  ├── sparsity.py               # Pruning helpers
  ├── stan_data_generator.py    # set up stan models
stan_code
  ├── gaussian_tanh                        # Gaussian model
  ├── dirichlet_horseshoe_tanh_nodewise    # DHS model
  ├── dirichlet_student_t_tanh_nodewise    # DST model
  ├── regularized_horseshoe_tanh_nodewise  # RHS model
├── example.ipynb             # Example experiment  
├── results/                  # Output directory (generated automatically)  
└── README.md  
```


## Contact
August Arnstad

augusa@uio.no

University of Oslo, Norway
