from dataset import NERDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import read_data,count,get_map
from arg import parse_args
from transformers import BertTokenizerFast
from model import Encoder
import torch
import wandb
from tqdm import tqdm
from model import NERTransformer
import os
import datetime
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
os.mkdir(nowtime)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args=parse_args()
def model_pipeline(config=args):
    # tell wandb to get started 
    with wandb.init(project=config.project_name, config=args,name=nowtime):
        # make the model, data, and optimization problem
        print("--loading--")
        model, trainloader, validloader, label2idx, optimizer, criterion = make(config)
        # and use them to train the model
        print("--training--")
        train(model, trainloader,validloader, optimizer, criterion, config)
        # and test its final performance
        return model
def make(config):
    # Make the data
    train_sens=read_data(os.path.join(args.data_dir,'train.txt'))
    train_labels=read_data(os.path.join(args.data_dir,'train_TAG.txt'))
    valid_sens=read_data(os.path.join(args.data_dir,'dev.txt'))
    valid_labels=read_data(os.path.join(args.data_dir,'dev_TAG.txt'))
    label_counts=count(train_labels)
    label2idx,idx2label=get_map(label_counts)
    tokenizer = BertTokenizerFast.from_pretrained("../hfl/chinese-macbert-base",local_files_only=config.local_files_only)

    train_encoding=tokenizer(train_sens, is_split_into_words=True, padding=True, truncation=True,max_length=args.train_length, return_tensors="pt")
    trainset=NERDataset(train_encoding, train_labels, args.train_length, label2idx)
    
    valid_encoding=tokenizer(valid_sens, is_split_into_words=True, padding=True, truncation=False, return_tensors="pt")

    lengths = [len(lst) for lst in valid_sens]

    validset=NERDataset(valid_encoding, valid_labels, max(lengths)+2, label2idx)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, persistent_workers=True)
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
    # Make the model
    model= NERTransformer(len(label2idx),tokenizer.vocab_size, config.d_model, config.num_layers, config.num_heads, config.ffn_hidden, config.max_length, config.dropout)
    if config.checkpoint:
        model.load_state_dict(torch.load(config.checkpoint_path))
    model=model.to(device)
    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    return model, trainloader, validloader, label2idx, optimizer, criterion
def train(model, trainloader, validloader, optimizer, criterion, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, log="gradients")
    # Run training and track with wandb
    # total_batches = len(loader) * config.epochs
    # example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in range(1,config.epochs+1):
        model.train()
        for _, batch in tqdm(enumerate(trainloader),total=len(trainloader),desc='Training'):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            loss = train_batch(input_ids, attention_mask, labels, model, optimizer, criterion)
            # example_ct +=  len(input_labels)
            batch_ct += 1
            # Report metrics every count batch
            if ((batch_ct + 1) % config.log_ct) == 0:
                wandb.log({"Train_loss": loss})
        torch.save(model.state_dict(), f'{nowtime}/model_weights_{epoch}.pth')
        val_loss, val_accuracy = validate(model, validloader, criterion)
        wandb.log({"Validation Loss": val_loss, "Validation Accuracy": val_accuracy})
        print(f"Epoch {epoch}/{config.epochs}, Train Loss: {loss.item()}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
def train_batch(input_ids, attention_mask, labels, model, optimizer, criterion):
    input_ids, attention_mask, labels= input_ids.to(device), attention_mask.to(device), labels.to(device)
    # Forward pass ➡
    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()
    # Step with optimizer
    optimizer.step()
    return loss
def validate(model, validloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(validloader),total=len(validloader),desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()
            # 计算准确率
            _, predicted = torch.max(outputs, -1)
            mask = labels != -100
            correct += ((predicted == labels) & mask).sum().item()
            total += mask.sum().item()
    avg_loss = total_loss / len(validloader)
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    model_pipeline(args)

