# LARI-PGMamba

这是最终整理版项目目录，用来集中放置当前实验对应的：

- 训练脚本
- 模型定义
- 数据集加载代码
- checkpoint 目录
- 运行说明


## 目录结构

```text
LARI-PGMamba/
├─ README.md
├─ requirements.txt
├─ train.py
├─ lari_pgmamba_model.py
├─ dtw_cuda.py
├─ soft_dtw_cuda.py
├─ length_adaptive_module.py
├─ tsc_mamba.py
├─ dataset/
│  ├─ __init__.py
│  ├─ datasetTrainAll_SF.py
│  ├─ datasetTest_SF.py
│  ├─ load_dataset.py
│  ├─ load_from_txt.py
│  └─ utils.py
├─ checkpoints/
│  └─ README.md
└─ docs/
   └─ checkpoints.md
```

## 已整理内容

### 1. 训练入口

- 主训练脚本：train.py

### 2. 模型相关文件

- lari_pgmamba_model.py
- length_adaptive_module.py
- tsc_mamba.py
- dtw_cuda.py
- soft_dtw_cuda.py

### 3. 数据集相关文件

- dataset/load_dataset.py
- dataset/datasetTrainAll_SF.py
- dataset/datasetTest_SF.py
- dataset/utils.py
- dataset/load_from_txt.py



### 3. Checkpoint 目录

目前只保留 `checkpoints/` 目录本身，不预放权重。

后续你可以把自己的最终模型直接放到这里。

## 数据路径说明

这个整理后的项目默认会自动查找以下数据目录：

1. 环境变量 `LARI_PGMAMBA_DATA_ROOT`
2. LARI-PGMamba/data/
3. 上一级目录的 data/（即当前工作区原来的 data/）
4. 当前运行目录下的 data/

因此在你当前工作区里，直接使用原来的 data/ 即可，不需要重复复制数据。

## 训练方式

直接右键运行 [train.py](train.py) 即可。

如果你要手动运行，也可以：

```text
python train.py --dataset all --epochs 50 --seed 111
```

如果数据不在默认位置，可以显式指定：

```text
python train.py --dataset all --data-root E:/your/data/root
```
