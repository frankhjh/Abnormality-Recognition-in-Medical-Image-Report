#!/usr/bin/env python
import torch
import torch.nn as nn

class lstm_cnn(nn.Module):
    def __init__(self):
        super(lstm_cnn,self).__init__()
        self.is_training=True
        self.num_classes=17
        self.vocab_size=859 #858+1
        self.embedd_dim=128
        self.text_len=104 
        self.hidden_size=64
        self.feature_size=128 
        self.num_layers=2
        self.window_sizes=[2,4,6,8,10] #kernel sizes for cnn
        self.dropout=nn.Dropout(0.2)
        self.embedding_layer=nn.Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embedd_dim,
                                          padding_idx=858)
        
        self.lstm=nn.LSTM(input_size=self.embedd_dim,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          bidirectional=True,
                          dropout=0.2,
                          batch_first=True)
        
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
        embedded_x=self.embedding_layer(x) # shape:[batch_size,text_len,embedd_dim]
        lstm_out,(_,_)=self.lstm(embedded_x) # shape:[batch_size,text_len,2*hidden_size]
        
        # use lstm layer as feature builder
        lstm_feat=lstm_out.permute(0,2,1) # shape:[batch_size,2*hidden_size,text_len]
        # use conv as feature extractor
        conv_outs=[conv(lstm_feat) for conv in self.convs] # shape of each element:[batch_size,feature_size,1]
        
        cnn_out=torch.cat(conv_outs,dim=1) # shape:[batch_size,num_windows * feature_size,1]
      
        cnn_out=cnn_out.view(-1,cnn_out.size(1)) #drop last dimension
        dropout=self.dropout(cnn_out)
        out=self.fc2(self.fc1(dropout))
        
        return out