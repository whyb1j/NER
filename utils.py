import matplotlib.pyplot as plt
from collections import Counter
from arg import parse_args
import os
from preprocess import read_data,count,get_map
def labels_distribution(counter: Counter, save_path: str) -> None:
    """生成标签分布条形图
    """
    # 计算总计数
    total_count = sum(counter.values())
    # 计算比例
    label_proportions = {label: count / total_count for label, count in counter.items()}
    # 提取标签和比例
    labels = list(label_proportions.keys())
    proportions = list(label_proportions.values())
    # 绘制比例图
    plt.figure(figsize=(10, 6))
    plt.bar(labels, proportions)
    plt.title('Label Proportions')
    plt.xlabel('Label')
    plt.ylabel('Proportion')
    plt.savefig(save_path)

def sens_distribution(sens: list, save_path: str) -> None:
    """生成原始句子长度分布条形图
    """
    # 计算每个列表的长度
    lengths = [len(lst) for lst in sens]
    max_length=max(lengths)
    # 生成直方图
    num_bins = (max_length + 100 - 1) // 100  # 根据最大长度计算需要的直方图组数
    bin_edges = [100 * i for i in range(num_bins + 1)]  # 生成直方图边界
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=bin_edges, edgecolor='black', align='left')
    # 添加标题和标签
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    # 显示图形
    plt.savefig(save_path)
if __name__=='__main__':
    args = parse_args()
    #读取数据，构建映射
    dev_sens=read_data(os.path.join(args.data_dir,'dev.txt'))
    dev_labels=read_data(os.path.join(args.data_dir,'dev_TAG.txt'))
    label_counts=count(dev_labels)
    label2idx,idx2label=get_map(label_counts)
    labels_distribution(label_counts,'../output/labels_dev.png')
    sens_distribution(dev_sens,'../output/sens_dev.png')