#!/usr/bin/env python

import pandas as pd
import json
from collections import defaultdict
from utils.dataset import Corpus_Dataset
from data_prepare import train_text,train_label,test_text
import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import dropout
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

from models.attn_model import TransEncoder
from models.lstm_model import lstm
from models.text_cnn_model import Multi_kernel_cnn
from models.attn_cnn_model import attn_cnn
from models.lstm_cnn_model import lstm_cnn
from models.attn_lstm_cnn_model import attn_lstm_cnn
import argparse

parser=argparse.ArgumentParser(description='training parameters')
parser.add_argument('--model_name',type=str)
parser.add_argument('--seed',type=int)
parser.add_argument('--epochs',type=int)
parser.add_argument('--device',type=str)
args=parser.parse_args()


def evalute(metric,model,loader,device):
    val_loss=0.0
    for step,(x,y) in enumerate(loader):
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            output=model(x)
            loss=metric(output,y)
            val_loss+=loss.item()
    return val_loss/(step+1)
        
        
def train(mn,model,train_dataloader,val_dataloader,epochs,device,lr=1e-3):
    m=model
    optimizer=optim.Adam(m.parameters(),lr=lr)
    metric=nn.BCEWithLogitsLoss()
    
    min_loss,best_epoch=100.0,1
    loss_dict=defaultdict(list) # store the loss in each epoch

    # initial loss
    init_train_loss=0.0
    init_val_loss=0.0
    with torch.no_grad():
        for step,(x,y) in enumerate(train_dataloader):
            x,y=x.to(device),y.to(device)
            output=m(x)
            loss=metric(output,y)
            init_train_loss+=loss.item()
        init_train_loss/=(step+1)
        loss_dict['train_loss'].append(init_train_loss)

        for step,(x,y) in enumerate(val_dataloader):
            x,y=x.to(device),y.to(device)
            output=m(x)
            loss=metric(output,y)
            init_val_loss+=loss.item()
        init_val_loss/=(step+1)
        loss_dict['val_loss'].append(init_val_loss)

    print('epoch 0,training loss:{}'.format(init_train_loss)+' validation loss:{}'.format(init_val_loss))
    
    for epoch in range(1,epochs+1):
        total_loss=0.0
        for step,(x,y) in enumerate(train_dataloader):
            x,y=x.to(device),y.to(device)
            output=m(x)
            loss=metric(output,y)
            
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            total_loss+=loss.item()
        avg_loss=total_loss/(step+1)
        loss_dict['train_loss'].append(avg_loss)

        val_loss=evalute(metric,m,val_dataloader,device)
        loss_dict['val_loss'].append(val_loss)
        if val_loss<min_loss:
            min_loss=val_loss
            best_epoch=epoch
            torch.save(model.state_dict(),'./train_out/bm.ckpt')
        
        print('epoch {},training loss:{}'.format(epoch,avg_loss)+' validation loss:{}'.format(val_loss))
    with open(f'./train_out/{mn}_loss.json','w') as f:
        json.dump(loss_dict,f)
    print('>>Training done!\n')

def predict(model,test_data):
    model.load_state_dict(torch.load('./train_out/bm.ckpt'))
    prediction=model(test_data)
    print('>>Predict done!\n')
    return prediction

def submit(pred_res):
    prediction=pred_res.sigmoid().detach().numpy().tolist()

    output_dict={'report_ID':[str(i)+'|' for i in range(pred_res.shape[0])],
             'Prediction':prediction}
    
    output_df=pd.DataFrame(output_dict)
    output_df['Prediction']=output_df['Prediction'].\
    apply(lambda x:[str(i) for i in x]).\
    apply(lambda x:'|'+' '.join(x))

    output_df.to_csv('./pred_out/submission.csv',index=False,header=None,sep=',')
    print('>>Submitted!')

def Main(model_name,seed,epochs,device):
    
    #split data into train/validation set
    train_text_tensor=torch.LongTensor(train_text[:9000])
    validation_text_tensor=torch.LongTensor(train_text[9000:])
    train_label_tensor=torch.Tensor(train_label[:9000])
    validation_label_tensor=torch.Tensor(train_label[9000:].reset_index(drop=True))

    test_data=torch.LongTensor(test_text)

    #data loader
    train_data=Corpus_Dataset(train_text_tensor,train_label_tensor)
    train_dataloader=DataLoader(train_data,batch_size=32,shuffle=True)
    val_data=Corpus_Dataset(validation_text_tensor,validation_label_tensor)
    val_dataloader=DataLoader(val_data,batch_size=32)

    if model_name=='cnn':
        model=Multi_kernel_cnn()
    if model_name=='rnn':
        model=lstm()
    if model_name=='attn':
        model=TransEncoder()
    if model_name=='attn_cnn':
        model=attn_cnn()
    if model_name=='lstm_cnn':
        model=lstm_cnn()
    if model_name=='attn_lstm_cnn':
        model=attn_lstm_cnn()
    
    torch.manual_seed(seed)
    # train
    train(model_name,model,train_dataloader,val_dataloader,epochs,device)
    # predict
    pred=predict(model,test_data)
    # submit
    submit(pred)
        

if __name__=='__main__':
    # parameters
    model_name=args.model_name
    seed=args.seed
    epochs=args.epochs
    device=args.device
    
    # run 
    Main(model_name,seed,epochs,device)


