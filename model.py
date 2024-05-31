import torch
import torch.nn as nn
import math
class TransformersEmbeddings(nn.Module):
    """将输入的token_ids转换为密集向量表示,并添加位置信息,以便模型能够理解输入序列中每个token的位置
    """
    def __init__(self, vocab_size, hidden_size, max_length, dropout):
        super(TransformersEmbeddings, self).__init__()
        self.max_length = max_length
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.layer_norm=LayerNorm(hidden_size,1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        """
        
        Args: 
            input_ids (tensor): 句子的数字编码，形状为 [batch_size,seq_length]
        
        Returns:
            embeddings: 形状为 [batch_size,seq_length,hidden_size]
        
        """
        # seq_length = input_ids.size(1)
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(self.max_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)[:, :seq_length, :]
        embeddings = words_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError
        self.d_k = embedding_dim // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)
        self.dropout=nn.Dropout(dropout)
        
    def transpose_for_multihead(self, x):
        #x的形状为(batch_size, seq_length, d)，d=d_k*num_heads
        new_shape = x.size(0), x.size(1), self.num_heads, self.d_k
        #重排为(batch_size, num_heads, seq_length, d_k)
        x = x.view(new_shape).permute(0, 2, 1, 3)
        return x

    def forward(self, queries, keys, values, mask=None):
        queries = self.transpose_for_multihead(self.W_q(queries))
        keys = self.transpose_for_multihead(self.W_k(keys))
        values = self.transpose_for_multihead(self.W_v(values))
        #scores: [batch_size,num_heads,num_q,num_k]
        scores=torch.matmul(queries,keys.transpose(-1,-2))/math.sqrt(self.d_k)
        if mask is not None:
          scores = scores.masked_fill(mask == 0,-1e9)
        attention_weights=torch.nn.functional.softmax(scores, dim=-1)
        # print(attention_weights.shape)
        attention_weights = self.dropout(attention_weights) 
        #num_k=num_v
        #结果为 [batch_size,num_heads,num_q,d_k]
        attention_output=torch.matmul(attention_weights,values)
        #转变为[batch_size,num_q, d_k*num_heads] 最后维度即embeddings
        new_shape=attention_output.size(0),attention_output.size(2),-1
        attention_output=attention_output.permute(0,2,1,3).contiguous().view(*new_shape)
        output=self.W_o(attention_output)
        #形状为[batch_size, num_q, embeddings]
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, ffn_hidden=1024, dropout=0.1) :
        super(EncoderLayer, self).__init__()
        self.attn1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.ffn = Feed_Forward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self,x,mask):
        # 多头自注意力
        T1=self.drop1(self.attn1(x,x,x,mask))
        T2=x+T1
        T3=self.norm1(T2)
        T4=self.drop2(self.ffn(T3))
        T5=T4+T3
        H=self.norm2(T5)
        return H
# out->[batch_size, num_tokens, d_model]
class Feed_Forward(nn.Module):
    def __init__(self, d_model=512, hidden_dim=1024, dropout=0.1):
        super(Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(d_model,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.fc2(self.dropout(self.fc1(x).relu()))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b
class Encoder(nn.Module):
    def __init__(self,vocab_size=21128, d_model=512, num_layers=6, num_heads=8, ffn_hidden=1024, max_length=2048, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = TransformersEmbeddings(vocab_size=vocab_size, hidden_size=d_model, max_length=max_length, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, ffn_hidden, dropout)
                for _ in range(num_layers)
            ]
        )
    def forward(self, x, mask):
        x = self.embedding(x)
        # print(x.shape)
        for layer in self.layers:
            x = layer(x, mask)
            # print(x.shape)
        return x

class NERTransformer(nn.Module):
    def __init__(self, num_labels, vocab_size, d_model, num_layers, num_heads, ffn_hidden, max_length, dropout):
        super(NERTransformer, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, ffn_hidden, max_length, dropout)
        self.classifier=nn.Linear(d_model, num_labels)
        self.num_heads=num_heads
    def forward(self, input_ids, attention_mask):
        seq_length=input_ids.size(1)
        # print(seq_length)
        mask = attention_mask.unsqueeze(1).unsqueeze(2)  # shape [2, 1, 1, 5]
        mask = mask.expand(-1, self.num_heads, -1, -1)  # shape [2, 8, 1, 5]
        mask = mask.expand(-1, -1, seq_length , -1)  # shape [2, 8, 5, 5]
        encoder_output = self.encoder(input_ids, mask)
        # print(encoder_output.shape) [batch_size,seq_len,d_model]
        logits = self.classifier(encoder_output)
        # print(logits.shape)    [batch_size,seq_len,class]
        return logits