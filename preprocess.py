from collections import Counter
import os
from arg import parse_args
def read_data(file_path):
    """从指定路径按行读取文本内容

    Return: [[data1],[data2]]
    """
    data=[]
    with open(file_path, 'r', encoding='utf-8') as file :
        for line in file:
            line_=line.split()
            data.append(line_)
    return data
def count(label) -> Counter:
    """生成标签种类的计数
    """
    label_list=[tag for n in range(len(label)) for tag in label[n]]
    label_counter = Counter(label_list)
    return label_counter
def get_map(counter) -> list:
    """生成标签到数字的映射
    
    为了保证每次映射的一致性,根据标签个数的降序排列
    """
    sorted_tuples = counter.most_common()
    sorted_elements = [item[0] for item in sorted_tuples]
    label2idx={tag:idx for idx,tag in enumerate(sorted_elements)}
    idx2label={idx:tag for tag,idx in label2idx.items()}
    return label2idx,idx2label
if __name__ == "__main__":
    args=parse_args()
    train_sens=read_data(os.path.join(args.data_dir,'train.txt'))
    train_labels=read_data(os.path.join(args.data_dir,'train_TAG.txt'))
    label_counts=count(train_labels)
    label2idx,idx2label=get_map(label_counts)
    print(label2idx)