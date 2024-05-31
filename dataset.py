from torch.utils.data import Dataset
import torch
class NERDataset(Dataset):
    """数据处理,最终得到的数据是张量
    Attributes:
        encodings: 经过tokenize的句子,要求对齐 [[sen1],[sen2]]
        labels: 原始标签,是字符而不是数字编码。由于tokenize产生了cls、sep、pad, 因此需要重新对齐 [[label1],[label2]]
    """
    def __init__(self, encodings, labels, max_length, label2idx):
        super(NERDataset, self).__init__()
        self.encodings = encodings
        self.labels = labels
        self.max_length = max_length
        self.label2idx = label2idx
        self.new_labels = self.aligin_labels() 
    def __getitem__(self, idx) :
        # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
        item = {}
        item['input_ids'] = self.encodings['input_ids'][idx]
        item['attention_mask'] = self.encodings['attention_mask'][idx]
        item['labels'] = self.new_labels[idx]
        return item
    def __len__(self):
        return self.encodings['input_ids'].size(0)

    def aligin_labels(self):
        new_labels = []
        for labels in  self.labels:
            labels_copy = labels[:]  # 创建标签列表的副本，避免修改原列表
            length = len(labels_copy)
            if length <= self.max_length - 2:  # 考虑到要加上 '[CLS]' 和 '[SEP]'，因此最大长度要减去 2
                labels_copy.insert(0, '[CLS]')
                labels_copy.append('[SEP]')
                padding_length = self.max_length - len(labels_copy)
                labels_copy.extend(['[PAD]'] * padding_length)
            else:
                labels_copy = labels_copy[:self.max_length - 2]  # 如果超过最大长度，则截取前 max_length - 2 个元素
                labels_copy.insert(0, '[CLS]')
                labels_copy.append('[SEP]')
            new_labels.append(labels_copy)
        new_labels = [[self.label2idx.get(single_tag,-100) for single_tag in tag] for tag in new_labels] #如果是特殊符号,就设置为-100，让模型忽略
        return torch.tensor(new_labels)
