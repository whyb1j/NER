import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
import os
from tqdm import tqdm
from model import NERTransformer
from preprocess import read_data,count,get_map
from dataset import NERDataset
from arg import parse_args
def load_model(config, label2idx,vocab_size):
    model = NERTransformer(
        num_labels=len(label2idx),
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_hidden=config.ffn_hidden,
        max_length=config.max_length,
        dropout=config.dropout
    )
    model.load_state_dict(torch.load(config.checkpoint_path))
    model.to(config.device)
    model.eval()
    return model

def evaluate(config):
    # 加载数据
    test_sens = read_data(os.path.join(config.data_dir, 'test.txt'))
    train_labels = read_data(os.path.join(config.data_dir, 'train_TAG.txt'))
    label_counts=count(train_labels)
    label2idx,idx2label=get_map(label_counts)
    # 加载分词器
    tokenizer = BertTokenizerFast.from_pretrained("../hfl/chinese-macbert-base", local_files_only=config.local_files_only)
    # 编码数据
    test_encoding = tokenizer(test_sens, is_split_into_words=True, padding=True, truncation=False,return_tensors="pt")
    lengths = [len(lst) for lst in test_sens] 
    test_labels=[['O']*len(juzi) for juzi in test_sens]  
    testset = NERDataset(test_encoding, test_labels, max(lengths)+2, label2idx)
    testloader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    model = load_model(config, label2idx, tokenizer.vocab_size)
    # print(test_labels[0])
    # 预测并保存结果
    predictions = []
    with torch.no_grad():
        for _,batch in tqdm(enumerate(testloader),total=len(testloader)):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels=batch['labels'].to(config.device)
            # print(labels)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, -1)
            predicted = predicted.cpu().numpy()
            # print(predicted.shape) （64，1459) 
            for sent_pred, sent_label in zip(predicted, labels):
                filtered_labels = [idx2label[pred] for pred, label in zip(sent_pred, sent_label) if label != -100]
                predictions.append(" ".join(filtered_labels))
                # print(predictions)
    # 保存到文件
    with open(config.output_file, 'w', encoding='utf-8') as f:
        for sentence in predictions:
            f.write(sentence + "\n")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
