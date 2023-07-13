import torch
import transformer
import pandas as pd
import numpy as np
import torchtext
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import spacy
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torchdata.datapipes as dp
import torchtext.transforms
torch.set_default_device('cuda')

rebuild_vocab=False




spacy_en=spacy.load('en_core_web_sm')

def target_tok(text):
    return[tok for tok in trivial_tokenize(text,lang='hi')]
    
def source_tok(text):
    return[tok.text for tok in spacy_en.tokenizer(text)]

def target_yield_tokens(data):
    for lines in data:
        yield target_tok(lines)

def source_yield_tokens(data):
    for lines in data:
        yield source_tok(lines)

class vocab_builder(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        self.data= []
        with open(path,'r',encoding='utf8') as f:
            for line in f:
                self.data.append(line.strip().lower())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

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

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols=['<unk>','<pad>','<bos>','<eos>']


if rebuild_vocab:
    dataset= vocab_builder('source_train.txt')
    tgt_data= vocab_builder('target_train.txt')

    source_vocab=torchtext.vocab.build_vocab_from_iterator(tqdm(source_yield_tokens(dataset)),min_freq=20,specials=special_symbols,special_first=True)
    source_vocab.set_default_index(UNK_IDX)

    target_vocab= torchtext.vocab.build_vocab_from_iterator(target_yield_tokens(tgt_data),min_freq=20,specials=special_symbols,special_first=True)
    target_vocab.set_default_index(UNK_IDX)
    torch.save(source_vocab,'source_vocab.pth')
    torch.save(target_vocab,'target_vocab.pth')

else:
    source_vocab=torch.load('source_vocab.pth')

    target_vocab=torch.load('target_vocab.pth')


def seq_transform(*transforms):
    def func(txt):
        for transform in transforms:
            txt=transform(txt)
        return txt
    return func

def tgt_to_tensor(tok_id: list[int]):
    return torch.cat((torch.tensor([BOS_IDX]),torch.tensor(tok_id),torch.tensor([EOS_IDX])))

def src_to_tensor(tok_id: list[int]):
    return torch.tensor(tok_id)

def get_src_mask(src_tok,pad_id=PAD_IDX):
    batch_size=src_tok.shape[0]
    src_mask=(src_tok!=pad_id).view(batch_size,1,1,-1)
    return src_mask

def get_tgt_mask(tgt_tok,pad_id=PAD_IDX):
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
        if s.shape[-1]<20 and t.shape[-1]<20 and s.shape[-1]>1 and t.shape[-1]>1:
            source.append(s)
            target.append(t)
 
    if len(source)==0:
        source.append(source_transform('yes'))
        target.append(target_transform('हाँ'))
    source= pad_sequence(source,padding_value=PAD_IDX,batch_first=True)
    target= pad_sequence(target,padding_value=PAD_IDX,batch_first=True)
    trg_tok=target[:,:-1]
    target_gt=target[:,1:].reshape(-1,1)
        
    return source, trg_tok, target_gt

epochs=3
DEVICE='cuda'
nhead=8
dropout_prob=0.2
model_dim=512
model_layers=6
Batch=32

src_vocab_size=len(source_vocab)

tgt_vocab_size=len(target_vocab)


model=transformer.Transformer(model_dim,src_vocab_size,tgt_vocab_size,nhead,model_layers,dropout_prob)

train_data=src_tgt('source_valid.txt','target_valid.txt')

train_loader=DataLoader(train_data,batch_size=Batch,collate_fn=collate)

loss_fn=torch.nn.KLDivLoss(reduction='batchmean')
label_smoothing=transformer.SmoothLabel(0.1,1,tgt_vocab_size)
optimizer=transformer.CustomLRAdam(torch.optim.Adam(model.parameters(),betas=(0.9,0.98),eps=1e-9),model_dim,num_warmup_steps=5000)

source_sen='dont do this man'

example=[]
for x in [source_sen]:
    s=source_transform(x.rstrip('\n'))
    example.append(s)
example=pad_sequence(example,padding_value=PAD_IDX,batch_first=True)



model.load_state_dict(torch.load('best.pt'))


model.to('cuda')
good_data=True
for epoch in range(epochs):
    print(f"{epoch+1} of {epochs} epochs")
    model.train()
    losses=0
    i=0
    """ for src,tgt,tgt_gt in tqdm(train_loader):
        i+=1
        src.to('cuda')
        tgt.to('cuda')
        tgt_gt.to('cuda')
        src_mask=get_src_mask(src)
        tgt_mask=get_tgt_mask(tgt)
        predicted_log_dist=model(src,src_mask,tgt,tgt_mask)
        smooth_tgt=label_smoothing(tgt_gt)
        optimizer.zero_grad()

        loss=loss_fn(predicted_log_dist,smooth_tgt)          
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses+=loss
        if torch.isnan(loss):
            good_data=False  """


    if ((epoch+1)%3==0):
        #print(f"total valid loss = {losses/i}")

        print(f'translation of the sentence: {source_sen} by model trained for {epoch+1} epochs')
        torch.save(model.state_dict(),f'best_{epoch+1}.pt')
        with torch.no_grad():
            model.eval()
            example_mask=get_src_mask(example)
            src_rep=model.encode(example,example_mask)
            x=transformer.greedy_decoding(model,src_rep,example_mask,target_vocab,max_target_tokens=50)
            x=x[0]
            list=[]
            list.append([target_vocab.lookup_token(idx) for idx in x])
            sen=list[0]
            listToStr = ' '.join([str(elem) for elem in sen])
            print(listToStr)
            



#torch.save(model.state_dict(),f'best.pt')







    #print([target_vocab.lookup_token(idx) for idx in x])



    