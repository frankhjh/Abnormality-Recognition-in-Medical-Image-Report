from models.transformer.transformer_encoder import Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransEncoder(nn.Module):
    def __init__(self):
        super(TransEncoder,self).__init__()
        self.vocab_size=858+1
        self.dim_word_vec=128
        self.n_layers=3
        self.n_head=4
        self.dim_k=32
        self.dim_v=32
        self.dim_model=128
        self.dim_hid=256
        self.pad_idx=858
        self.len_=104
        self.n_class=17
        self.fc1=nn.Linear(self.len_,1)
        self.fc2=nn.Linear(self.dim_model,self.n_class)
        self.dropout=nn.Dropout(0.3)
        self.encoder=Encoder(vocab_size=self.vocab_size,
                             dim_word_vec=self.dim_word_vec,
                             n_layers=self.n_layers,
                             n_head=self.n_head,
                             dim_k=self.dim_k,
                             dim_v=self.dim_v,
                             dim_model=self.dim_model,
                             dim_hid=self.dim_hid,
                             pad_idx=self.pad_idx)
    
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