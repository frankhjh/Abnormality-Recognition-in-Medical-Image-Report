#!/usr/bin/env python

import torch
import torch.nn as nn
from models.transformer.transformer_encoder import Encoder

class attn_lstm_cnn(nn.Module):
    def __init__(self):
        super(attn_lstm_cnn,self).__init__()
        self.vocab_size=858+1
        self.dim_word_vec=128
        self.n_layers=1
        self.n_head=4
        self.dim_k=32
        self.dim_v=32
        self.dim_model=128 # output size of attn compoent
        self.dim_hid=256 # hidden dim of feed-forward nn of attn compoent
        self.pad_idx=858
        self.text_len=104
        self.num_classes=17
        self.dropout=nn.Dropout(0.2)
        self.encoder=Encoder(vocab_size=self.vocab_size,
                             dim_word_vec=self.dim_word_vec,
                             n_layers=self.n_layers,
                             n_head=self.n_head,
                             dim_k=self.dim_k,
                             dim_v=self.dim_v,
                             dim_model=self.dim_model,
                             dim_hid=self.dim_hid,
                             pad_idx=self.pad_idx)
        
        self.hidden_size=64
        self.lstm=nn.LSTM(input_size=self.dim_model,
                          hidden_size=self.hidden_size,
                          num_layers=self.n_layers,
                          bidirectional=True,
                          #dropout=0.2,
                          batch_first=True)
        self.feature_size=128
        self.window_sizes=[2,4,6,8,10]
        self.convs=nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=2*self.hidden_size,
                                           out_channels=self.feature_size,
                                           kernel_size=self.window_sizes[i],
                                           stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=self.text_len-self.window_sizes[i]+1)) 
                                  for i in range(len(self.window_sizes))])
        
        self.fc1=nn.Linear(in_features=len(self.window_sizes)*self.feature_size,
                          out_features=128)
        self.fc2=nn.Linear(in_features=128,out_features=self.num_classes)
    
    def forward(self,x):
        # first feature builder
        attn_feat=self.encoder(x) # shape:[batch_size,text_len,dim_model]

        # second feature builder
        lstm_feat,(_,_)=self.lstm(attn_feat) # shape:[batch_size,text_len,2*hidden_size]
        lstm_feat=lstm_feat.permute(0,2,1)
        # conv as feature extractor
        conv_outs=[conv(lstm_feat) for conv in self.convs] # shape of each element:[batch_size,feature_size,1]
        
        cnn_out=torch.cat(conv_outs,dim=1) # shape:[batch_size,num_windows * feature_size,1]
      
        cnn_out=cnn_out.view(-1,cnn_out.size(1)) #drop last dimension
        dropout=self.dropout(cnn_out)
        out=self.fc2(self.fc1(dropout))
        
        return out