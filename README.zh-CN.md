# PitNETs

[English](README.md)

## 项目介绍

利用 H&E 全切片图像（WSI）进行垂体神经内分泌肿瘤分类，提供 WSI-only 和 WSI-clinical 模型。

**不提供患者数据和预训练权重。** Virchow2 需单独获取访问权限。

## 论文与引用

Zheng, A., Lin, R., Chen, C. et al. [A clinically conditional deep learning model for predicting histological subtypes of pituitary neuroendocrine tumors from hematoxylin-eosin whole-slide images](https://doi.org/10.1186/s12911-026-03918-y). *BMC Med Inform Decis Mak* (2026).

## 环境要求

使用 `pitnets` Conda 环境、Python 3.11 和 PyTorch 2.7.0+cu128，依赖见 [environment.yml](environment.yml)。特征提取、训练和推理需要 CUDA。

从仓库根目录执行 `conda env create -f environment.yml` 创建环境，再执行 `conda activate pitnets` 激活环境。

## 处理流程

从仓库根目录运行模块，使用 `--help` 查看参数。

1. `python -m preprocessing.select_patches` — 筛选组织图像块。
2. `python -m preprocessing.augment` — 采样未使用的图像块坐标。
3. `python -m preprocessing.extract_features` — 提取 Virchow2 CLS 特征。
4. `python -m training.train` — 训练 WSI-only 或 WSI-clinical 模型。
5. `python -m inference.predict` — 对五折 softmax 概率取平均。
6. `python -m evaluation.evaluate` — 输出指标、混淆矩阵和 ROC 曲线。


## 许可

自有代码采用 [MIT 许可](LICENSE)，第三方依赖与 Virchow2 遵循各自的许可及访问条件。
