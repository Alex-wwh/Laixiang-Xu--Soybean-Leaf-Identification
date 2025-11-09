# SCD-Net-for-Soybean-Disease-Identification


> 官方PyTorch实现 | 论文标题：《SCD-Net: An Efficient Swin Transformer and Cross-covariance Mechanism Network for Soybean Leaf Disease Recognition》  


> 提出SCD-Net模型，基于Pytorch框架实现四类大豆叶片病害的高精度识别，为农业病害智能诊断提供高效解决方案。


## 1. 研究背景与模型定位  

大豆作为重要粮食作物，其叶片病害（如灰斑病、黄斑病、花叶病等）严重威胁产量和品质。传统大豆病害识别方法依赖人工观察和经验判断，存在识别准确率低、效率不高的问题。 

 
本文提出**SCD-Net**模型，通过改进Swin Transformer架构、引入交叉协方差注意力机制和动态Tanh模块，解决大豆病害"特征复杂、背景干扰大、相似病害区分难"的问题。模型基于Pytorch框架实现，在四类大豆病害数据集上实现优异的分类性能，为农作物病害的智能识别与精准诊断提供了一种轻量化、高效率的解决方法。

## 2. SCD-Net核心创新点  

1. **三阶协同注意力机制**：
-**轻量化窗口注意力**：通过精简Transformer块与注意力头，在保留局部特征提取能力的同时显著降低计算复杂度。
-**交叉协方差注意力（XCA）**：替换原有窗口注意力，强化通道间特征关联建模，捕捉跨区域病斑关联。
-**动态Tanh注意力（DyT）**：将归一化层与激活函数统一为动态结构，自适应调节特征分布，增强关键病害特征表达。

2. **高效轻量级主干架构**：
对Swin Transformer进行针对性轻量化改造，将初始嵌入维度从96降至64，Transformer块数量调整为（1, 1, 3, 1），在减少40%参数量的同时保持基础特征提取能力。

3. **多模块交互优化**：
通过“基础层-注意力层-激活层”的三阶架构设计，实现模块间协同增强，既解决了有限数据集上的过拟合问题，又提升了模型对复杂病斑特征的判别能力。

## 3. 实验数据集：四类大豆病害数据集  

### 3.1 数据集概况  

本研究基于**四类大豆病害识别数据集**，包含四种常见大豆叶片状态，数据集存储于百度网盘，需自行下载后使用：  

| 数据集名称 | 包含类别 | 图像总数 | 图像分辨率 | 数据分布（训练:验证:测试） |
|------------|-------------------------|----------|------------|-----------------------|
| 四类大豆数据集 | 灰斑病（Grey spot）、黄斑病（Macular）、花叶病（Mosaic）+ 健康叶片（Healthy） | 7,000+ | 统一resize至244×244（适配模型输入） | 3:1:1 |  


### 3.2 数据集获取与结构  

1. **下载链接**：  
   百度网盘链接：https://pan.baidu.com/s/1zOLpjwrSh6RvajdGFlT17w  
   提取码: 0qyw  

2. **文件夹组织**（下载后解压至项目根目录，结构如下）：  
```  
soybean/  
├── grey spot/       # 大豆灰斑病叶片图像  
├── healthy/             # 健康大豆叶片图像  
├── macular/           # 大豆黄斑病叶片图像  
└── mosaic/              # 大豆花叶病叶片图像  
```  


## 4. 实验环境配置  

### 4.1 依赖安装  

推荐使用Anaconda创建虚拟环境，确保依赖版本匹配（Pytorch框架核心依赖）：  

```bash  
# 1. 创建并激活虚拟环境  
conda create -n pytorch_env python=3.80  
conda activate pytorch_env  

# 2. 安装Pytorch（支持GPU/CPU，示例为CPU版本）  
pip install torch~=2.4.1+cpu

# 3. 安装其他依赖库   
pip install numpy~=2.0.2 matplotlib~=3.9.4
pip install pillow~=11.2.1 scikit-learn~=1.5.1
pip install tqdm~=4.67.1 tensorboard~=2.15.1
```  

## 5 代码使用说明  

### 5.1 模型训练  

运行`train.py`脚本启动训练，支持通过参数调整训练配置，示例命令：  

```bash  
python train.py \  
  --data_dir ./soybean \  # 数据集根目录（解压后的路径）  
  --epochs 50 \  
  --batch_size 16 \  
  --lr 1e-4 \  
  --weight_decay 5e-2 \  
  --save_dir ./weights \  
  --log_interval 1  # 每1个batch打印一次训练日志  
```  
   

#### 关键参数说明：  

| 参数名 | 含义 | 默认值 |
|-----------------|---------------------------------------|-----------------|
| `--data_dir` | 数据集根目录路径 | `./soybean` |
| `--epochs` | 训练轮数 | 50 |
| `--batch_size` | 批次大小（根据显存调整，8/16/32） | 16 |
| `--lr` | 初始学习率 | 1e-4 |
| `--save_dir` | 训练权重保存目录 | `./weights` |
| `--device` | 训练设备（`GPU`或`CPU`） | `CPU` |  



### 5.2 模型预测  

本项目采用标准的数据划分流程，在训练阶段自动将数据集按3:1:1划分为训练集、验证集和测试集。模型训练完成后，可直接在预留的测试集上进行性能评估。
bash
# 在训练过程中自动评估
python train.py \
  --data_path ./soybean \  # 数据集根目录
  --weights ./swin_tiny_patch4_window7_224.pth \  # 预训练权重
  --num_classes 4 \  # 病害类别数
  --batch_size 16

评估输出：
text
Total images: 7516
Train: 4509 (60.0%)
Validation: 1503 (20.0%) 
Test: 1504 (20.0%)
Test Accuracy: 98.47%

## 6. 项目文件结构  

```  
SCD-Net-for-soybean-disease-identification/  
├── soybean/  # 四类大豆病害数据集（需从百度网盘下载）  
├── models/                 #整体模型实现
├── DyT/                 #DyT模块实现
├── utils/                 #数据读取、训练评估等工具函数
├── XCA/                 #XCA模块实现
├── my_dataset/                 #自定义数据集类
├── select_incorrect_samples/                 #筛选预测错误的样本
├── swin_tiny_patch4_window7_224/        # 预训练权重  
├── class_indices.json              #类别标签映射文件（4种大豆病害）
├── train.py              # 模型训练脚本（Pytorch版）  
├── predict.py            # 模型预测脚本（Pytorch版）    
├── weights/              # 模型权重保存目录（自动生成）  
└── README.md             # 项目说明文档（本文档）  
```  


## 7. 已知问题与注意事项  
1.**模型兼容性**：本项目基于PyTorch框架开发，依赖特定版本的PyTorch及相关库，不兼容TensorFlow环境；
2.**输入尺寸限制**：模型固定输入为224×224×3，训练和预测时会自动resize输入图像，建议原始图像分辨率≥256×256以保留病害细节特征；
3.**类别扩展限制**：当前模型针对4类大豆叶片病害（健康、灰斑病、黄斑病、花叶病）优化，新增病害类别需重新收集数据并修改model.py中num_classes参数；
  


## 8. 引用与联系方式  

### 8.1 引用方式  

论文处于投刊阶段，正式发表后将更新BibTeX引用格式，当前可临时引用：  

```bibtex  
@article{scd_soybean_disease,  
  title={SCD-Net: An Efficient Swin Transformer and Cross-covariance Mechanism Network for Soybean Leaf Disease Recognition},  
  author={[作者姓名，待发表时补充]},  
  journal={[期刊名称，待录用后补充]},  
  year={2025},  
  note={Manuscript submitted for publication}  
}  
```  


### 8.2 联系方式  

若遇到代码运行问题或学术交流需求，请联系：  
- 邮箱：yukaidi@huuc.edu.cn  
- GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。

