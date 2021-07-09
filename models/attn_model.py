from models.transformer.transformer_encoder import Encoder
import torch
import torch.nn as nn

class TransEncoder(nn.Module):
    def __init__(self,vocab_size,dim_word_vec,n_layers,n_head,dim_k,dim_v,dim_model,dim_hid,pad_idx):
        super(TransEncoder,self).__init__()
        self.encoder=Encoder(vocab_size,dim_word_vec,n_layers,n_head,dim_k,dim_v,dim_model,dim_hid,pad_idx)
        self.len_=104
        self.n_class=17
        self.fc1=nn.Linear(self.len_,1)
        self.fc2=nn.Linear(dim_model,self.n_class)
        self.dropout=nn.Dropout(0.3)
    
    def forward(self,x):
        encoder_output=self.encoder(x)
        x=encoder_output.transpose(-2,-1)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.fc1(x).squeeze(-1)
        x=F.relu(x)
        x=self.dropout(x)
        output=self.fc2(x)
        
        return output

#parameters
vocab_size=858+1
dim_word_vec=128
n_layers=3
n_head=4
dim_k=32
dim_v=32
dim_model=128
dim_hid=256
pad_idx=858

Attn_model=TransEncoder(vocab_size,dim_word_vec,n_layers,n_head,dim_k,dim_v,dim_model,dim_hid,pad_idx)