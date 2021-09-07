#!/usr/bin/env python
import json
import matplotlib.pyplot as plt
import os

# compare the training losses for each model
def plot_loss_epochs(path,train=True):
    out={}
    for file in os.listdir(path):
        if file.endswith('.json'):
            with open(path+file,'r') as f:
                loss=json.load(f)
                if train:
                    out[file[:-5]]=loss['train_loss']
                else:
                    out[file[:-5]]=loss['val_loss']
    
    epochs=[i for i in range(16)]
    title='Training Loss' if train else 'Validation Loss'
    plt.title(title)
    plt.plot(epochs,out['cnn_loss'],color='red',label='cnn')
    plt.plot(epochs,out['rnn_loss'],color='yellow',label='rnn(lstm)')
    plt.plot(epochs,out['attn_loss'],color='green',label='attention')
    plt.plot(epochs,out['lstm_cnn_loss'],color='blue',label='lstm-cnn')
    plt.plot(epochs,out['attn_cnn_loss'],color='skyblue',label='attention-cnn')
    plt.plot(epochs,out['attn_lstm_cnn_loss'],color='black',label='attention-lstm-cnn')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(f'../plot_out/{title}.png')
    

if __name__=='__main__':
    #plot_loss_epochs('../train_out/',train=True)
    plot_loss_epochs('../train_out/',train=False)


