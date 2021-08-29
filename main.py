import pandas as pd
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
import argparse

parser=argparse.ArgumentParser(description='training parameters')
parser.add_argument('--model',type=str)
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
        
        
def train(model,train_dataloader,val_dataloader,epochs,device,lr=1e-3):
    m=model
    optimizer=optim.Adam(m.parameters(),lr=lr)
    metric=nn.BCEWithLogitsLoss()
    
    min_loss,best_epoch=1000.0,0
    for epoch in range(epochs):
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
        
        val_loss=evalute(metric,m,val_dataloader,device)
        if val_loss<min_loss:
            min_loss=val_loss
            best_epoch=epoch
            torch.save(model.state_dict(),'./train_out/bm.ckpt')
        print('epoch {},training loss:{}'.format(epoch,avg_loss)+' validation loss:{}'.format(val_loss))
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

def Main(model,seed,epochs,device):
    
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

    if model=='cnn':
        model=Multi_kernel_cnn()
    if model=='rnn':
        model=lstm()
    if model=='attn':
        model=TransEncoder()
    
    torch.manual_seed(seed)
    # train
    train(model,train_dataloader,val_dataloader,epochs,device)
    # predict
    pred=predict(model,test_data)
    # submit
    submit(pred)
        

if __name__=='__main__':
    # parameters
    model=args.model
    seed=args.seed
    epochs=args.epochs
    device=args.device
    
    # run 
    Main(model,seed,epochs,device)


