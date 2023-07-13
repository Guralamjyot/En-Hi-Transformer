import torch
import transformer
import pandas as pd
import numpy as np
import torchtext
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from torch.utils.data import Dataset, DataLoader
import spacy
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torchtext.transforms
torch.set_default_device('cuda')



#///hyperparameters///
epochs=10
DEVICE='cuda'
nhead=8
dropout_prob=0.2
model_dim=512
model_layers=6
Batch=32
source_sen='how hard can it be to translate?'



spacy_en=spacy.load('en_core_web_sm')

def target_tok(text):
    return[tok for tok in trivial_tokenize(text,lang='hi')]
    
def source_tok(text):
    return[tok.text for tok in spacy_en.tokenizer(text)]

class src_tgt(Dataset):
    def __init__(self,src_path,tgt_path):
        super().__init__()
        self.src_data=[]
        self.tgt_data=[]
        with open(src_path,'r',encoding='utf8') as f:
            for line in f:
                self.src_data.append(line.strip().lower())
        with open(tgt_path,'r',encoding='utf8') as f:
            for line in f:
                self.tgt_data.append(line.strip().lower())
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, index):
        return self.src_data[index], self.tgt_data[index]


source_vocab=torch.load('source_vocab.pth')
target_vocab=torch.load('target_vocab.pth')


def seq_transform(*transforms):
    def func(txt):
        for transform in transforms:
            txt=transform(txt)
        return txt
    return func

def tgt_to_tensor(tok_id: list[int]):
    return torch.cat((torch.tensor([2]),torch.tensor(tok_id),torch.tensor([3])))

def src_to_tensor(tok_id: list[int]):
    return torch.tensor(tok_id)

def get_src_mask(src_tok,pad_id=1):
    batch_size=src_tok.shape[0]
    src_mask=(src_tok!=pad_id).view(batch_size,1,1,-1)
    return src_mask

def get_tgt_mask(tgt_tok,pad_id=1):
    batch_size=tgt_tok.shape[0]
    seq_len=tgt_tok.shape[1]
    tgt_pad_mask=(tgt_tok!=pad_id).view(batch_size,1,1,-1)
    tgt_no_look_fwd=torch.triu(torch.ones((1,1,seq_len,seq_len),device='cuda')==1).transpose(2, 3)
    tgt_mask= tgt_pad_mask & tgt_no_look_fwd
    return tgt_mask

source_transform= seq_transform(source_tok,
                                source_vocab,
                                src_to_tensor,
                                )

target_transform= seq_transform(target_tok,
                                target_vocab,
                                tgt_to_tensor,
                                )

def collate(batch):
    source=[]
    target=[]
    for src,tgt in batch:
        s=source_transform(src.rstrip('\n'))
        t=target_transform(tgt.rstrip('\n'))
        if s.shape[-1]<25 and t.shape[-1]<25 and s.shape[-1]>1 and t.shape[-1]>1:
            source.append(s)
            target.append(t)
 
    #if len(source)==0:
        #source.append(source_transform('yes'))
        #target.append(target_transform('हाँ'))

    source= pad_sequence(source,padding_value=1,batch_first=True)
    target= pad_sequence(target,padding_value=1,batch_first=True)
    trg_tok=target[:,:-1]
    target_gt=target[:,1:].reshape(-1,1)
        
    return source, trg_tok, target_gt


src_vocab_size=len(source_vocab)
tgt_vocab_size=len(target_vocab)


model=transformer.Transformer(model_dim,src_vocab_size,tgt_vocab_size,nhead,model_layers,dropout_prob)

train_data=src_tgt('source_train.txt','target_train.txt')
train_loader=DataLoader(train_data,batch_size=Batch,collate_fn=collate)

valid_data=src_tgt('source_test.txt','target_test.txt')
valid_loader=DataLoader(valid_data,batch_size=Batch,collate_fn=collate)

loss_fn=torch.nn.KLDivLoss(reduction='batchmean')
label_smoothing=transformer.SmoothLabel(0.1,1,tgt_vocab_size)
optimizer=transformer.CustomLRAdam(torch.optim.Adam(model.parameters(),betas=(0.9,0.98),eps=1e-9),model_dim,num_warmup_steps=5000)



example=[]
for x in [source_sen]:
    s=source_transform(x.rstrip('\n'))
    example.append(s)
example=pad_sequence(example,padding_value=1,batch_first=True)











    