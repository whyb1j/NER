## 任务描述
基于Transformer编码器的命名体识别
## 数据集说明
数据文件的格式
- 每个文件包含多行，每行对应一个句子
- 对应的标签文件每行包含与文本文件中的每个字对应的标签
## 安装和依赖项
- wandb
## 使用说明
### 训练模型
python train.py --data_dir xxx --save_dir xxx -- and so on
### 评估模型
python evaluate.py
