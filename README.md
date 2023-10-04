### Optimized Covariance Design for AB Test on Social Network under Interference

------





#### Description

----

##### Files

- **main.py**: main file, containing implemented methods and simulation process.
- **optimize_cov_mat.py**: optimize the covariance matrix 



##### Folders

- **dataset**: two network topology datasets used for simulations.
- **experiment_outcome**: saved simulation results on two networks
- **optimized_cov**: optimized treatment assignment covariance matrices
- **tables**: saved latex table for simulation results





#### Usage

-----

To run the total experiments:

```bash
bash exp.sh
```



To run specific experiment:

```bash
python3 main.py --dataset YOUR_DATA --resolution YOUR_RES --gamma YOUR_GAMMA --model YOUR_MODEL --omega YOUR_OMEGA
```

- **dataset**: "stf" for FB-Stanford3 and "cornell" for FB-Cornell5
- **resolution**: (positive) int, clustering resolution, the parameter of *networkx.algorithms.community.louvain*  
- **gamma**: (positive) float, the interference parameter $\gamma$ 
- **model**: "linear" and "multi"
- **omega**: (positive) float, the parameter in comparability assumption



Before specifc experiment, please check whether the optimized covariance matrix has been calculated. If not, run

```bash
python3 main.py --dataset YOUR_DATA --resolution YOUR_RES --omega YOUR_OMEGA
```

The optimized covariance matrix will be saved to folder **optimized_cov**.





#### **Dependencies**

-----

Networkx >= 2.8.4

Scipy >= 1.9.3

Numpy >=1.23.4

Pandas >= 1.5.1

PyTorch >= 1.0 (only necessary for calculating new optimized covariance matrix)