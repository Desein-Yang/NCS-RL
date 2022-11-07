# README

This is a library of Evolutionary Reinforcement Learning(ERL) algorithms. The library contains a series of evolutionary algorithms based on negative correlation algorithms applied to reinforcement learning strategy optimization.

- [x] NCS(Negative Correlated Search) Framework
    - [x] NCS-C
    - [x] NCSCC
    - [x] NCNES
    - [x] NCSRE

### Usage

Dependency
- mpi4py (need mpirun environment)
- tensorflow=1.15
- gym=0.9.1
- click 
- numpy
- opencv-python


NCSCC

```
mpirun -np cpus python main.py [-v][-r][-e][-g][-f][-d][--epoch][--sigma0][--rvalue]
--version, -v     
--run_name, -r  
--env_type, -e   
--game,-g         
--function_id, -f 
--dimension, -d   
--k, -k       
--epoch,     
--sigma0,  
--rvalue 
```

NCSRE
```
mpirun -np cpus python NCSRE.py --[hyperparameter]
```

NCNES
```
mpirun -np cpus python NCNES.py --[hyperparameter]
```

NCS-C

```
mpirun -n cpus python NCS.py [-e][-g][-c][-r]
-n the num of cpus  
-e the num of individuals on 1 cpu
-g the name of benchmark (gamename in atari benchmark)
-r the name of log file
-c the configuration files default =./configurations/sample_configuration.json
```

### Files Tree
src/
    decomposer.py   utils for decision varibles decomposition utils 
    env_wrappers.py utils for env preprocess in gym                 
    ops.py          utils for RL policy building 
    models.py       Definition of Neural Network Models of policy
    policy.py       Definition of RL policy (with rollout) 
    testfunc.py     utils for CEC benchmarks (test EA algorithms)
    logger.py       utils for logging
data/                      
test/                     
scripts/   
examples/


##### Citation

If the repo is useful for you, please cite the paper as

```
@incollection{yang2021,
    author = {Yang, Qi and Yang, Peng and Tang, Ke},
	title = {Parallel Random Embedding with Negatively Correlated Search},
	volume = {12690},
    year = {2021},
	doi = {10.1007/978-3-030-78811-7_33}
	isbn = {978-3-030-78810-0 978-3-030-78811-7},
	pages = {339--351},
	booktitle = {Advances in Swarm Intelligence},
	publisher = {Springer International Publishing},
	editor = {Tan, Ying and Shi, Yuhui},
}
```
