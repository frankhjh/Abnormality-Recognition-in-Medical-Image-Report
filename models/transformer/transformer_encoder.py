import torch
import torch.nn as nn
import numpy as np
from models.transformer.transformer_layers import EncoderLayer

def get_pad_mask(seq,pad_idx):
    return (seq!=pad_idx).unsqueeze(-2)

#定义位置信息
class PositionalEncoding(nn.Module):

    def __init__(self, dim_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        #缓存在内存中，常量
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, dim_hid))

    def _get_sinusoid_encoding_table(self, n_position, dim_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / dim_hid) for hid_j in range(dim_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):

    def __init__(self,vocab_size,dim_word_vec,n_layers,n_head,dim_k,dim_v,dim_model,dim_hid,pad_idx,dropout=0.2,n_position=200):
        super(Encoder,self).__init__()
        self.embedding_layer=nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=dim_word_vec,
                                          padding_idx=pad_idx)
        self.positionencode=PositionalEncoding(dim_hid=dim_word_vec,n_position=200)
        self.dropout=nn.Dropout(dropout)
        self.layer_stacks=nn.ModuleList([
            EncoderLayer(dim_model=dim_model,dim_hid=dim_hid,n_head=n_head,dim_k=dim_k,dim_v=dim_v)
          for _ in range(n_layers)])
        self.layer_norm=nn.LayerNorm(dim_model,eps=1e-6)
        self.dim_model=dim_model
        self.pad_idx=pad_idx
    
    def forward(self,x):
        
        token_embedd=self.embedding_layer(x)
        token_position_embedd=self.dropout(self.positionencode(token_embedd))
        encode_output=self.layer_norm(token_position_embedd) #shape=[B,L,E]--->(batch_size,seq_len,embed_dim)

        mask=get_pad_mask(x,self.pad_idx) #shape=[B,1,L]
        for encode_layer in self.layer_stacks:
            encode_output=encode_layer(encode_output,slf_attn_mask=mask)
        
        return encode_output















        
        
        
