#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.functional import dropout

class Multi_kernel_cnn(nn.Module):
    def __init__(self):
        super(Multi_kernel_cnn,self).__init__()
        self.is_training=True
        self.num_classes=17
        self.vocab_size=859 
        self.embedd_dim=128 
        self.text_len=104 
        self.dropout_rate=0.5
        self.feature_size=128 
        self.window_sizes=[2,3,4,5,6]
        
        self.embedding_layer=nn.Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embedd_dim,
                                          padding_idx=858)
        
        self.convs=nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=self.embedd_dim,
                                           out_channels=self.feature_size,
                                           kernel_size=self.window_sizes[i],
                                           stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=self.text_len-self.window_sizes[i]+1)) 
                                  for i in range(len(self.window_sizes))])
        
        self.fc=nn.Linear(in_features=len(self.window_sizes)*self.feature_size,
                          out_features=self.num_classes)
    
    def forward(self,x): 
        embedded_x=self.embedding_layer(x)#size=[batch_size,text_len,embedd_dim]
       
        embedded_x=embedded_x.permute(0,2,1)#size=[batch_size,embedd_dim,text_len]
       
        conv_x=[conv(embedded_x) for conv in self.convs]#size(conv_x[i])=[batch_size,feature_size,1]
        
        tmp_output=torch.cat(conv_x,dim=1) #size=[batch_size,5*feature_size,1]
      
        tmp_output=tmp_output.view(-1,tmp_output.size(1)) #size=[batch_size,5*feature_size]
        tmp_output=dropout(input=tmp_output,p=self.dropout_rate)
        final_output=self.fc(tmp_output)
        
        
        return final_output     