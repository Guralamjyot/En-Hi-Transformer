from torch.utils.data import Dataset, DataLoader
import torchtext
from tqdm import tqdm
import torch
import spacy
from indicnlp.tokenize.indic_tokenize import trivial_tokenize


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
spacy_en=spacy.load('en_core_web_sm')

special_symbols=['<unk>','<pad>','<bos>','<eos>']

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

min_freq=20

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


dataset= vocab_builder('source_train.txt')
tgt_data= vocab_builder('target_train.txt')

source_vocab=torchtext.vocab.build_vocab_from_iterator(tqdm(source_yield_tokens(dataset)),min_freq=min_freq,specials=special_symbols,special_first=True)
source_vocab.set_default_index(UNK_IDX)

target_vocab= torchtext.vocab.build_vocab_from_iterator(target_yield_tokens(tgt_data),min_freq=min_freq,specials=special_symbols,special_first=True)
target_vocab.set_default_index(UNK_IDX)
torch.save(source_vocab,'source_vocab.pth')
torch.save(target_vocab,'target_vocab.pth')