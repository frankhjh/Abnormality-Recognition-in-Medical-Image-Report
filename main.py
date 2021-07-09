from utils.dataset import Corpus_Dataset
from data_prepare import train_text,train_label,test_text
import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import dropout
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

from models.attn_model import Attn_model
from models.lstm_model import lstm
from models.text_cnn_model import Multi_kernel_cnn


def evalute(metric,model,loader):
    val_loss=0.0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            output=model(x)
            loss=metric(output,y)
            val_loss+=loss.item()
    return val_loss
        
        
def train_predict(model,metric,train_dataloader,val_dataloader,test_data,epochs,lr):
    m=model
    optimizer=optim.Adam(m.parameters(),lr=lr)

    
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

        val_loss=evalute(metric2,m,val_dataloader)
        if val_loss<min_loss:
            min_loss=val_loss
            best_epoch=epoch
            torch.save(model.state_dict(),'bm.ckpt')
        print('epoch {},training loss:{}'.format(epoch,total_loss)+' validation loss:{}'.format(val_loss))
    print('Training done!\n')

    print('Start predicting...')
    m.load_state_dict(torch.load('bm.ckpt'))
    prediction=m(test_data)
    print('All Done!')
    
    return prediction

def Main(model,metric,train_dataloader,val_dataloader,test_data,epochs,lr):
    pred=train_predict(model,metric,train_dataloader,val_dataloader,test_data,epochs,lr)
    prediction=pred.sigmoid().detach().numpy().tolist()

    output_dict={'report_ID':[str(i)+'|' for i in range(pred.shape[0])],
             'Prediction':prediction}
    
    output_df=pd.DataFrame(output_dict)
    output_df['Prediction']=output_df['Prediction'].\
    apply(lambda x:[str(i) for i in x]).\
    apply(lambda x:'|'+' '.join(x))

    output_df.to_csv('submission.csv',index=False,header=None,sep=',')


if __name__=='__main__':
    #data
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
    
    #parameters
    epochs=20
    lr=1e-3
    device=torch.device('cpu')
    torch.manual_seed(5)
    metric=nn.BCEWithLogitsLoss()

    #take cnn as example
    model=Multi_kernel_cnn()

    Main(model,metric,train_dataloader,val_dataloader,test_data,epochs,lr)
    


