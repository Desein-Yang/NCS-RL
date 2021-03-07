# README

## 代码文件说明

本版本代码集成了现有的演化强化学习方法的实现，包含以下算法，勾选的表示测试完全通过。
- [x] CES(Canonical Evolution Strategy)
- [x] NCS(Negative Correlated Search) Framework
    - [ ] NCS-C
    - [x] NCSCC
    - [ ] NCNES
    - [ ] NCSRE

相关的代码文件有：

NCSCC.py 定义了算法类
src/
    decomposer.py   变量分组类
    env_wrappers.py gym环境中atari游戏的预处理等等操作
    ops.py          用于辅助定义atari游戏的策略模型
    models.py       定义了atari游戏的策略模型
    policy.py       atari游戏的策略模型类，封装了rollout（测试策略模型）和模型

    testfunc.py    CEC测试环境
    logger.py  工具文件，定义了日志类

其余文件：

data/                         CEC测试函数要求的数据
test/                         mpirun api测试的代码文件
scripts/                      采用过的运行脚本

## Update log
2020 0817
- 调试 NCS base
2020 0901
- 调试 CES
2020 0904
- random seed
- demo


## 运行

### 环境要求

要求支持mpirun

python推荐环境(没有版本号表示不要求固定版本号)：

    tensorflow: 1.15
    gym: 0.9.1
    click 
    numpy
    opencv-python

### 运行命令

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
--version, -v     算法的版本
--run_name, -r    本次运行的别名，保存日志文件时会用到这个名字
--env_type, -e    环境的类型（取值为 atari或者function）
--game,-g         Atari游戏的名字，不需要带NoFrameskip-v4，代码中会自动加具体的版本信息
--function_id, -f CEC测试函数的id，取值范围[1,2,3,4,5,6]
--dimension, -d   CEC测试函数的维度
--k, -k           演化过程中得到模型适应度值时模型被评估的次数
--epoch,          NCSCC算法中更新高斯噪声标准差周期，一般取值为5的倍数
--sigma0,         NCSCC算法中高斯噪声标准差的初始值
--rvalue          NCSCC算法中更新高斯噪声标准差的系数.
```

NCSRE
```
mpirun -np cpus python NCSRE.py --[hyperparameter]
```

NCNES
```
mpirun -np cpus python NCNES.py --[hyperparameter]
```

