[TOC]

复现论文：TRANSFORMER-BASED MULTIMODAL FUSION FOR EARLY DIAGNOSIS OF ALZHEIMER’S DISEASE USING STRUCTURAL MRI AND PET

# 数据存放

数据的存放有两种方式，但无论哪种方式都需要有可直接读取的、形状一致的npy格式MRI影像：

第一种：在datasets中新建目录npy_file，然后将npy文件全部放到npy_file中。

第二种：不新建npy_file，把npy文件全部放到`路径path`，然后修改config.py中的mri_path为`路径path`。

# 数据预处理

运行以下代码即可进行数据预处理，生成`datasets/train.csv`，`datasets/valid.csv`，`datasets/test.csv`

```bash
python data_preprocess.py
```

# 训练模型

运行以下代码即可训练模型，模型保存点将放在`checkpoints/<model_name>/`，`model_name`可在`train.py`中自定义：

```bash
python train.py
```

# 计算指标

运行以下代码即可计算指标：

```bash
python compute_performance.py
```