[![Python 3.11](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3106/)
[![license](https://img.shields.io/badge/license-apache_2.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

## A Bi-Objective Acquisition Function for Batch Bayesian Global Optimization

<p align="center">
  <img height="300" src="readme_img/BiOBO_flow_dark.png">
</p>
 
Implementation of the Bi-Objective Acquisition Function Methodology proposed in

[Carciaghi, F., Magistri, S., Mansueto, P. & Schoen F., A Bi-Objective Optimization Based Acquisition Strategy for Batch Bayesian Global Optimization. Computational Optimization and Applications (2025)](
https://link.springer.com/article/10.1007/s10589-025-00696-7)


If you have used our code for research purposes, please cite the publication mentioned above.
For the sake of simplicity, we provide the Bibtex format:

```
﻿@Article{Carciaghi2025,
author={Carciaghi, Francesco
and Magistri, Simone
and Mansueto, Pierluigi
and Schoen, Fabio},
title={A Bi-Objective Optimization Based Acquisition Strategy for Batch Bayesian Global Optimization},
journal={Computational Optimization and Applications},
year={2025},
month={May},
day={23},
abstract={In this paper we deal with batch Bayesian Optimization (Bayes-Opt) problems over a box. Bayes-Opt approaches find their main applications when the objective function is very expensive to evaluate. Sometimes, given the availability of multi-processor computing architectures, function evaluation might be performed in parallel in order to lower the clock-time of the overall computation. This paper fits this situation and is devoted to the development of a novel bi-objective optimization (BOO) acquisition strategy to sample batches of points where to evaluate the objective function. The BOO problem involves the Gaussian Process posterior mean and variance functions, which, in most of the acquisition strategies from the literature, are generally used in combination, frequently through scalarization. However, such scalarization could compromise the Bayes-Opt process performance, as getting the desired trade-off between exploration and exploitation is not trivial in most cases. We instead aim to reconstruct the Pareto front of the BOO problem exploiting first order information of the posterior mean and variance, thus generating multiple trade-offs of the two functions without any a priori knowledge. The algorithm used for the reconstruction is the Non-dominated Sorting Memetic Algorithm (NSMA), recently proposed in the literature and proved to be effective in solving hard MOO problems. Finally, we present two clustering approaches, each of them operating on a different space, to select potentially optimal points from the Pareto front. We compare our methodology with well-known acquisition strategies from the literature, showing its effectiveness on a wide set of experiments.},
issn={1573-2894},
doi={10.1007/s10589-025-00696-7},
url={https://doi.org/10.1007/s10589-025-00696-7}
}
```

### Installation

In order to execute the code, you need an [Anaconda](https://www.anaconda.com/) environment. We provide a YAML file in order to facilitate the installation of the latter.
Open an Anaconda terminal in the project root folder and execute the following command. Note that the code was experimented in a computer with Ubuntu 22.04.

```
conda env create -f env_install.yml
```

#### Main Packages

* ```python v3.10.4```
* ```botorch v0.8.1```
* ```nsma v1.0.12```
* ```tensorflow v2.11.0```
* ```gurobipy v9.5.2```

#### Gurobi Optimizer

In order to run some parts of the code, the [Gurobi](https://www.gurobi.com/) Optimizer needs to be installed and, in addition, a valid Gurobi licence is required. 

### Usage

In ```utils/args_manager.py``` you can find all the possible arguments.
Given an Anaconda terminal opened in the root folder, an example of code execution could be the following:

```python -u main.py -acq_m nsma --n_batch 20 --batch_size 3 --selection_type X  --exp_name exp_name --function_name Rastrigin```

The execution results are saved in the ```experiments``` folder. If the latter does not exist, it is created at the beginning of the code execution.

#### Plot results

In order to get plots of the results obtained by one code execution, you can run ```plot_function_eval.py```: all figures will be created and saved in ```plots``` folder. 
An example of terminal command could be the following:

```python plot_function_eval.py --exp_paths experiments/exp_name```

In case you want to plot the results of more than one experiment together, you can provide a list of experiment paths as follows:

```python plot_function_eval.py --exp_paths experiments/exp_name_1 experiments/exp_name_2 experiments/exp_name_3```

in case you have run three different experiments.
