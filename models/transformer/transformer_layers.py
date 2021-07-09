import torch.nn as nn
import torch
from models.transformer.transformer_sublayers import ScaleDotProductAttention,MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self,dim_model,dim_hid,n_head,dim_k,dim_v,dropout=0.2):
        super(EncoderLayer,self).__init__()
        self.slf_attn=MultiHeadAttention(n_head,dim_model,dim_k,dim_v)
        self.ffn=PositionwiseFeedForward(dim_model,dim_hid,dropout=dropout)
    

    def forward(self,enc_input,slf_attn_mask=None):
        attn_output=self.slf_attn(enc_input,enc_input,enc_input,mask=slf_attn_mask) #mask:Boolean构成的[B,1,L]
        output=self.ffn(attn_output)
        return output
        



