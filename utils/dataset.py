from torch.utils.data import Dataset

class Corpus_Dataset(Dataset):
    def __init__(self,text,label):
        super(Corpus_Dataset,self).__init__()
        self.text=text
        self.label=label
    
    def __len__(self):
        return len(self.text)
        
    def __getitem__(self,idx):
        return self.text[idx],self.label[idx] 