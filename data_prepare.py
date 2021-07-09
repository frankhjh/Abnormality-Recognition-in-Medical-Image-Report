import pandas as pd
import numpy as np
from utils.trans_label import label_transform


#load data
train_raw=pd.read_csv('data/track1_round1_train_20210222.csv',delimiter='|',header=None)
test_raw=pd.read_csv('data/track1_round1_testA_20210222.csv',delimiter='|',header=None)

#data clean
train_set=train_raw[[0,2,4]].rename(columns={0:'id',2:'description',4:'label'})
train_set['label']=train_set['label'].fillna('-')
test_set=test_raw[[0,2]].rename(columns={0:'id',2:'description'})

#data prepare
train_text=train_set['description'].apply(str.split).tolist()
train_text=[list(map(lambda x:int(x),text)) for text in train_text]
test_text=test_set['description'].apply(str.split).tolist()
test_text=[list(map(lambda x:int(x),text)) for text in test_text]

#check the size of vocab
wordset=set()
for text in train_text:
    wordset.update(text)
#print('corpus-size:{}'.format(len(wordset)))

#padding--text
padding_len=104
train_text=[text+[858]*(padding_len-len(text)) if padding_len>=len(text) else text[:padding_len] for text in train_text]
test_text=[text+[858]*(padding_len-len(text)) if padding_len>=len(text) else text[:padding_len] for text in test_text]

#label
train_label=train_set['label'].apply(label_transform)




