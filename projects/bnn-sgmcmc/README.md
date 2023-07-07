# SG-MCMC for Bayesian Inference

## Abstracts
[Stochastic Gradient Hamiltonian Monte Carlo (ICML 2014)](https://arxiv.org/abs/1402.4102)
> Hamiltonian Monte Carlo (HMC) sampling methods provide a mechanism for defining distant proposals with high acceptance probabilities in a Metropolis-Hastings framework, enabling more efficient exploration of the state space than standard random-walk proposals. The popularity of such methods has grown significantly in recent years. However, a limitation of HMC methods is the required gradient computation for simulation of the Hamiltonian dynamical system-such computation is infeasible in problems involving a large sample size or streaming data. Instead, we must rely on a noisy gradient estimate computed from a subset of the data. In this paper, we explore the properties of such a stochastic gradient HMC approach. Surprisingly, the natural implementation of the stochastic approximation can be arbitrarily bad. To address this problem we introduce a variant that uses second-order Langevin dynamics with a friction term that counteracts the effects of the noisy gradient, maintaining the desired target distribution as the invariant distribution. Results on simulated data validate our theory. We also provide an application of our methods to a classification task using neural networks and to online Bayesian matrix factorization.

[Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning (ICLR 2020)](https://arxiv.org/abs/1902.03932)
> The posteriors over neural network weights are high dimensional and multimodal. Each mode typically characterizes a meaningfully different representation of the data. We develop Cyclical Stochastic Gradient MCMC (SG-MCMC) to automatically explore such distributions. In particular, we propose a cyclical stepsize schedule, where larger steps discover new modes, and smaller steps characterize each mode. We also prove non-asymptotic convergence of our proposed algorithm. Moreover, we provide extensive experimental results, including ImageNet, to demonstrate the scalability and effectiveness of cyclical SG-MCMC in learning complex multimodal distributions, especially for fully Bayesian inference with modern deep neural networks.

## Usage examples
Run SGDM optimization for R20x1 on CIFAR10_x32:
```
python scripts/SGDM.py
    --resnet_depth=20 --resnet_width=1 --data_name=CIFAR10_x32
    --batch_size=80 --data_augmentation={false,true} --num_epochs=500
    --optim_lr=0.1 --optim_momentum=0.9 --prior_var=0.2
    --seed=42 --save=/path/to/directory
```

Run SGHMC sampling for R20x1 on CIFAR10_x32:
```
python scripts/SGDM.py
    --resnet_depth=20 --resnet_width=1 --data_name=CIFAR10_x32
    --batch_size=80 --data_augmentation={false,true}
    --num_epochs_quiet=0 --num_epochs_noisy=50 --num_cycles_burnin=20 --num_cycles_sample=30
    --optim_lr=0.1 --optim_momentum=0.9 --prior_var=0.2
    --posterior_tempering={1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001}
    --seed=42 --save=/path/to/directory
```
