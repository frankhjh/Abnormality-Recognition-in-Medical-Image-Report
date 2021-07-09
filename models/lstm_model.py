import torch
import torch.nn as nn
from torch.nn.functional import dropout
class lstm(nn.Module):
    def __init__(self):
        super(lstm,self).__init__()
        self.is_training=True
        self.num_classes=17
        self.vocab_size=859
        self.embedd_dim=128
        self.text_len=104 
        self.dropout_rate=0.5
        self.hidden_size=128 
        self.num_layers=3
        
        self.embedding_layer=nn.Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embedd_dim,
                                          padding_idx=858)
        self.lstm=nn.LSTM(input_size=self.embedd_dim,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          bidirectional=True,
                          dropout=0.2,
                          batch_first=True)
        self.fc=nn.Linear(2*self.hidden_size,self.num_classes)
    
    def forward(self,x):
        embedded_x=self.embedding_layer(x)#size=[batch_size,text_len,embedd_dim]
        lstm_output,(hc,_)=self.lstm(embedded_x)
        
        tmp_output=torch.cat([hc[-2,:,:],hc[-1,:,:]],dim=1) 
        tmp_output=dropout(input=tmp_output,p=self.dropout_rate)
        final_output=self.fc(tmp_output)
        
        return final_output       