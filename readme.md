# README

- [x] NCS(Negative Correlated Search) Framework
    - [x] NCS-C
    - [x] NCSCC
    - [x] NCNES
    - [x] NCSRE


src/
    decomposer.py   变量分组类
    env_wrappers.py gym环境中atari游戏的预处理等等操作
    ops.py          用于辅助定义atari游戏的策略模型
    models.py       定义了atari游戏的策略模型
    policy.py       atari游戏的策略模型类，封装了rollout（测试策略模型）和模型

    testfunc.py    CEC测试环境
    logger.py      工具文件，定义了日志类

其余文件：

data/                      
test/                     
scripts/                   


## 运行

### 环境要求

要求支持mpirun

python推荐环境(没有版本号表示不要求固定版本号)：

    tensorflow: 1.15
    gym: 0.9.1
    click 
    numpy
    opencv-python

### Usage

##### NCS

```
mpirun -n 9 python NCS.py [-e][-g][-c][-r]
-n 使用的cpu数
-e 每个cpu上运行的个体数
-g atari游戏名
-r log记录名
-c 配置文件路径，默认为./configurations/sample_configuration.json]
```



##### NCSCC

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

##### Citation

If you feel the repo is useful, please cite the paper as

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
