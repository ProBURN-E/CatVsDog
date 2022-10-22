# CatVsDog
NEU ISE Python编程与深度学习时间 实验课作业 20192732

Classification of cat and dog images using deep learning methods using PyTorch template

Thanks to https://github.com/victoresque/pytorch-template
## Folder Structure
  ```
  CatVsDog/
│
├── train.py – 训练的主要脚本
├── test.py – 测试训练的模型
│
├── config.json – 训练的配置文件
├── parse_config.py - 处理配置文件和命令行选项的类
├── base/ - 抽象的基类
│   ├── base_data_loader.py
│   ├── base_model.py
│   └── base_trainer.py
│
├── data_loader/ - 所有数据加载相关内容
│   └── data_loaders.py
│
├── data/ - 存储输入数据的默认目录
│
├── model/ - 模型、损失和指标
│   ├── model.py
│   ├── metric.py
│   └── loss.py
│
├── saved/
│   ├── models/ - 训练好的模型保存目录
│   └── log/ - 日志和tensorboard输出的默认目录
│
├── trainer/ - trainers
│   └── trainer.py
│
├── logger/ - 日志记录和tensorboard模块
│   ├── visualization.py
│   ├── logger.py
│   └── logger_config.json
│  
└── utils/ - 其他的小实用功能
    ├── util.py
    └── ...
```
