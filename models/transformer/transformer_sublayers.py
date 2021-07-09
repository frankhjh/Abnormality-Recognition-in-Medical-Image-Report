import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleDotProductAttention(nn.Module):
    def __init__(self,scale,atten_dropout=0.1):
        super(ScaleDotProductAttention,self).__init__()
        self.dropout=nn.Dropout(atten_dropout)
        self.scale=scale
    
    def forward(self,q,k,v,mask=None): #shape=[B,H,L,D]
        attn=torch.matmul(q,k.transpose(-2,-1)) #这里q:[B,H,L,D] k:[B,H,D,L]
        scaled_attn=attn/self.scale #attn 的output:[B,H,L,L]

        if mask is not None: #传入的mask:[B,1,1,L]
            scaled_attn.masked_fill(mask==0,-1e9)
        
        scaled_attn=self.dropout(F.softmax(scaled_attn,dim=-1))
        output=torch.matmul(scaled_attn,v) 
        return output,scaled_attn

class MultiHeadAttention(nn.Module):
    def __init__(self,n_head,dim_model,dim_k,dim_v,dropout=0.2):
        super(MultiHeadAttention,self).__init__()
        self.dim_model=dim_model
        self.n_head=n_head
        self.dim_k=dim_k #query 和 key的维度相同所以这里只定义一个
        self.dim_q=dim_k
        self.dim_v=dim_v

        self.w_q=nn.Linear(dim_model,n_head*dim_k,bias=False)
        self.w_k=nn.Linear(dim_model,n_head*dim_k,bias=False)
        self.w_v=nn.Linear(dim_model,n_head*dim_v,bias=False)
        self.fc=nn.Linear(n_head*dim_v,dim_model,bias=False)

        self.attention=ScaleDotProductAttention(scale=dim_k**0.5)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(dim_model,eps=1e-6)
    
    def forward(self,q,k,v,mask=None):
        d_k,d_v,n_head=self.dim_k,self.dim_v,self.n_head
        batch_size,len_q,len_k,len_v=q.size(0),q.size(1),k.size(1),v.size(1)

        residual=q
        q=self.w_q(q).view(batch_size,len_q,n_head,d_k) #将head单独取出作为一维
        k=self.w_k(k).view(batch_size,len_k,n_head,d_k)
        v=self.w_v(v).view(batch_size,len_v,n_head,d_v)

        #在attention前将len_ 与 head维度互换
        q,k,v=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2) #shape=[B,H,L,D]

        if mask is not None: #传入的mask:[B,1,L]
            mask = mask.unsqueeze(1)   # For head axis broadcasting--->mask:[B,1,1,L]
        
        #attention
        output,attn=self.attention(q,k,v,mask=mask)
        output=output.transpose(1,2).contiguous().view(batch_size,len_q,-1)#合并heads
        output=self.dropout(self.fc(output))
        #print(output.shape,q.shape)
        output+=residual #+residual
        output=self.layer_norm(output) #layer normalization
        return output

class PositionwiseFeedForward(nn.Module):
    '''two feed forward layers'''
    def __init__(self,dim_in,dim_hid,dropout=0.2):
        super(PositionwiseFeedForward,self).__init__()
        self.w1=nn.Linear(dim_in,dim_hid)
        self.w2=nn.Linear(dim_hid,dim_in) #输出维度不变
        self.layer_norm=nn.LayerNorm(dim_in,eps=1e-6)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        residual=x
        x=self.w2(self.dropout(F.relu(self.w1(x))))
        x+=residual
        return x

    







        










