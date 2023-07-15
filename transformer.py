import torch
import math
import copy
from nltk.translate.bleu_score import corpus_bleu

class Transformer(torch.nn.Module):
    def __init__(self, model_dim,src_vocab_size,tgt_vocab_size,no_heads,no_layers,dropout,store_att_wei=False):
        super().__init__()
        self.src_emb= Embedding(src_vocab_size,model_dim)
        self.tgt_emb= Embedding(tgt_vocab_size,model_dim)
        

        self.src_pos_emb= PositionalEmb(model_dim,dropout)
        self.tgt_pos_emb= PositionalEmb(model_dim,dropout)

        MultiAtt= MultiHeadAttention(model_dim,no_heads,dropout,store_att_wei)
        #MultiAtt= torch.nn.MultiheadAttention(model_dim,no_heads,dropout)
        positionalFF=PositionalFF(model_dim,dropout)
        
        encoder_layer= EncoderLayer(model_dim,dropout,MultiAtt,positionalFF)
        decoder_layer= DecoderLayer(model_dim,dropout,MultiAtt,positionalFF)

        self.encoder= Encoder(encoder_layer, no_layers)
        self.decoder= Decoder(decoder_layer,no_layers)

        self.decodegen = DecodeGenerator(model_dim,tgt_vocab_size)

        self.init_params()

    def init_params(self):
        for name, p in self.named_parameters():
            if p.dim()>1:
                torch.nn.init.xavier_uniform_(p)
  
    def encode(self,src_tok_batch,src_mask):
        src_emb_batch= self.src_emb(src_tok_batch)
        src_emb_batch= self.src_pos_emb(src_emb_batch)
        src_rep= self.encoder(src_emb_batch,src_mask)
        return src_rep 

    def decode(self,tgt_tok_batch,tgt_mask,src_rep,src_mask):
        tgt_emb_batch = self.tgt_emb(tgt_tok_batch)
        tgt_emb_batch= self.tgt_pos_emb(tgt_emb_batch)
        tgt_rep= self.decoder(tgt_emb_batch,tgt_mask,src_rep,src_mask)
        tgt_probs= self.decodegen(tgt_rep)
        tgt_probs= tgt_probs.reshape(-1,tgt_probs.shape[-1])

        return tgt_probs #<- in log form and reshaped to (batch*maxlen, vocabsize) for easier pass through KL divloss

    def forward(self,src_tok_batch,src_mask,tgt_tok_batch,tgt_mask):
        src_rep=self.encode(src_tok_batch,src_mask)
        tgt_rep=self.decode(tgt_tok_batch,tgt_mask,src_rep,src_mask)
        return tgt_rep

def get_clones(module,no_of_layers):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(no_of_layers)])

class EncoderLayer(torch.nn.Module):
    def __init__(self,model_dim,dropout,MultiAtt,ff_net):
        super().__init__()
        no_layer=2
        self.sublayers= get_clones(SublayerLogic(model_dim,dropout),no_layer)
        self.mha=MultiAtt
        self.ff_net=ff_net
        self.model_dim=model_dim

    def forward(self,src_rep,src_mask):
        encoder_selfatt= lambda src_rep: self.mha(q=src_rep,k=src_rep,v=src_rep,mask=src_mask)
        src_rep= self.sublayers[0](src_rep,encoder_selfatt)
        src_rep= self.sublayers[1](src_rep,self.ff_net)
        return src_rep

class Encoder(torch.nn.Module):
    def __init__(self,encoder_layer, no_layers):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer),f'Expected EncoderLayer got {type(encoder_layer)}.'

        self.encoder_layers= get_clones(encoder_layer,no_layers)
        self.norm=torch.nn.LayerNorm(encoder_layer.model_dim)

    def forward(self,src_emb,src_mask):
        for encoder_layer in self.encoder_layers:
            src_emb=encoder_layer(src_emb,src_mask)
        return self.norm(src_emb)

class DecoderLayer(torch.nn.Module):
    def __init__(self,model_dim,dropout,MultiAtt,ff_net):
        super().__init__()
        no_layers=3
        self.sublayers= get_clones(SublayerLogic(model_dim,dropout),no_layers)
        self.src_mha=copy.deepcopy(MultiAtt)
        self.tgt_mha=copy.deepcopy(MultiAtt)
        self.ff_net=ff_net
        self.model_dim=model_dim

    def forward(self,src_emb,src_mask,tgt_emb,tgt_mask):
        decoder_tgt_selfatt= lambda tgt: self.tgt_mha(q=tgt,k=tgt,v=tgt,mask=tgt_mask)
        decoder_src_selfatt= lambda tgt: self.src_mha(q=tgt,k=src_emb,v=src_emb,mask=src_mask)

        tgt_emb=self.sublayers[0](tgt_emb,decoder_tgt_selfatt)
        tgt_emb=self.sublayers[1](tgt_emb,decoder_src_selfatt)
        tgt_emb=self.sublayers[2](tgt_emb,self.ff_net)

        return tgt_emb

class Decoder(torch.nn.Module):
    def __init__(self,decoder_layer,no_layers):
        super().__init__()
        assert isinstance(decoder_layer,DecoderLayer),f"Expected DecoderLayer got {type(decoder_layer)}."

        self.decoder_layers= get_clones(decoder_layer,no_layers)
        self.norm=torch.nn.LayerNorm(decoder_layer.model_dim)

    def forward(self,tgt_emb,tgt_mask,src_emb,src_mask):
        for layer in self.decoder_layers:
            tgt_emb=layer(src_emb,src_mask,tgt_emb,tgt_mask)
        return self.norm(tgt_emb)
    
class DecodeGenerator(torch.nn.Module):
    def __init__(self,model_dim,tgt_vocab_size):
        super().__init__()
        self.linear=torch.nn.Linear(model_dim,tgt_vocab_size)
        self.log_softmax= torch.nn.LogSoftmax(-1)

    def forward(self,tgt_rep):
        return self.log_softmax(self.linear(tgt_rep))
    
class SublayerLogic(torch.nn.Module):
    def __init__(self,model_dim,dropout):
        super().__init__()
        self.norm=torch.nn.LayerNorm(model_dim)
        self.dropout=torch.nn.Dropout(dropout)

    def forward(self,emb_rep,sublayer_module):
        return emb_rep + self.dropout(sublayer_module(self.norm(emb_rep)))
    
class PositionalFF(torch.nn.Module):
    def __init__(self,model_dim,dropout,width=4):
        super().__init__()
        self.linear_1= torch.nn.Linear(model_dim,model_dim*width)
        self.linear_2=torch.nn.Linear(model_dim*width,model_dim)
        self.dropout=torch.nn.Dropout(dropout)
        self.relu=torch.nn.ReLU()

    def forward(self,emb_rep):
        return self.linear_2(self.dropout(self.relu(self.linear_1(emb_rep))))

class Embedding(torch.nn.Module):
    def __init__(self,vocab_size,model_dim):
        super().__init__()
        self.emb_table=torch.nn.Embedding(vocab_size,model_dim)
        self.model_dim=model_dim

    def forward(self,tok_ids):
        assert tok_ids.ndim==2,f"expected(B,maxlen), got {tok_ids.shape}."

        emb=self.emb_table(tok_ids.int())#.long()removed

        return emb* math.sqrt(self.model_dim)
    
class PositionalEmb(torch.nn.Module):
    def __init__(self,model_dim,dropout,maxlen=5000):
        super().__init__()
        self.dropout= torch.nn.Dropout(dropout)

        pos_id= torch.arange(0,maxlen).unsqueeze(1)
        freq=torch.pow(10000,-torch.arange(0,model_dim,2,dtype=torch.float)/model_dim)

        pos_enc_table= torch.zeros(maxlen,model_dim)
        pos_enc_table[:,0::2]=torch.sin(pos_id*freq)
        pos_enc_table[:,1::2]=torch.cos(pos_id*freq)

        self.register_buffer("pos_enc_table",pos_enc_table)

    def forward(self,emb):
        assert emb.ndim==3 and emb.shape[-1]==self.pos_enc_table.shape[1], f"Expected B,Maxlen,model_dim got {emb.shape}"

        posenc= self.pos_enc_table[:emb.shape[1]]

        return self.dropout(emb + posenc)
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self,model_dim,no_heads,dropout,store_att_wei):
        super().__init__()
        assert model_dim % no_heads ==0, "model_dim should be perfectly divisible by no_heads"
        self.dropout=torch.nn.Dropout(dropout)
        self.head_dim=int(model_dim/no_heads)
        self.no_heads=no_heads
        self.model_dim=model_dim

        self.qkv_nets= get_clones(torch.nn.Linear(model_dim,model_dim),3)
        self.out_proj= torch.nn.Linear(model_dim,model_dim)
        self.softmax=torch.nn.Softmax(-1)

        self.store_att_wei=store_att_wei
        self.att_wei=None


    def attention(self,q,k,v,mask):
        scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)

        if mask is not None:
            scores.masked_fill_(mask==torch.tensor(False),float("-inf"))

        att_wei=self.dropout(self.softmax(scores))
        inter_tok_rep=torch.matmul(att_wei,v)
        return inter_tok_rep, att_wei

    def forward(self,q,k,v,mask):
        batch_size= q.shape[0]

        q,k,v=[net(x).view(batch_size,-1,self.no_heads,self.head_dim).transpose(1,2) for net,x in zip(self.qkv_nets, (q,k,v))]
        tok_rep, att_wei= self.attention(q,k,v,mask)
    

        if self.store_att_wei:
            self.att_wei=att_wei

        reshaped= tok_rep.transpose(1,2).reshape(batch_size,-1,self.no_heads*self.head_dim)

        tok_rep= self.out_proj(reshaped)
        return tok_rep
    

class CustomLRAdam:
    def __init__(self,optimizer,model_dim,num_warmup_steps=500):
        self.optimizer=optimizer
        self.model_dim=model_dim
        self.num_steps=num_warmup_steps

        self.current_steps=0
    def step(self):
        self.current_steps+=1
        current_lr=self.get_LR()
        for p in self.optimizer.param_groups:
            p['lr']=current_lr
            self.optimizer.step()

    def get_LR(self):
        step=self.current_steps
        warmup=self.num_steps
        return self.model_dim**(-0.5)* min(step**(-0.5), step*warmup**(-1.5))
    
    def zero_grad(self):
        self.optimizer.zero_grad()
                                   
class SmoothLabel(torch.nn.Module):

    def __init__(self, smoothing_value,pad_token_id,tgt_vocab_size,device='cuda'):
        assert 0.0 <= smoothing_value <=1.0
        super().__init__()
        self.confidence= 1.0-smoothing_value
        self.smoothing_value=smoothing_value

        self.pad_tok_id=pad_token_id
        self.tgt_vocab_size=tgt_vocab_size
        self.device=device

    def forward(self,tgt_tok_id):
        batch_size=tgt_tok_id.shape[0]
        smooth_target= torch.zeros((batch_size,self.tgt_vocab_size),device=self.device)
        smooth_target.fill_(self.smoothing_value / (self.tgt_vocab_size - 2))

        smooth_target.scatter_(1, tgt_tok_id, self.confidence)
        smooth_target[:, self.pad_tok_id] = 0.

        # If we had a pad token as a target we set the distribution to all 0s instead of smooth labeled distribution
        #smooth_target.masked_fill_(tgt_tok_id == self.pad_tok_id, 0.)

        return smooth_target

def get_src_mask(src_tok,pad_id=1):
    batch_size=src_tok.shape[0]
    src_mask=(src_tok!=pad_id).view(batch_size,1,1,-1)
    return src_mask

def get_tgt_mask(tgt_tok,pad_id=1):
    batch_size=tgt_tok.shape[0]
    seq_len=tgt_tok.shape[1]
    tgt_pad_mask=(tgt_tok!=pad_id).view(batch_size,1,1,-1)
    tgt_no_look_fwd=torch.triu(torch.ones((1,1,seq_len,seq_len),device='cuda')==1).transpose(2, 3)
    tgt_mask= tgt_pad_mask& tgt_no_look_fwd
    return tgt_mask

def calculate_bleu_score(transformer, token_ids_loader, trg_field_processor):
    with torch.no_grad():
        pad_token_id = trg_field_processor.vocab.stoi['<pad>']

        gt_sentences_corpus = []
        predicted_sentences_corpus = []

 
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            src_token_ids_batch, trg_token_ids_batch = token_ids_batch.src, token_ids_batch.trg
           
            src_mask, _ = get_src_mask(src_token_ids_batch)
            src_representations_batch = transformer.encode(src_token_ids_batch, src_mask)

            predicted_sentences = greedy_decoding(transformer, src_representations_batch, src_mask, trg_field_processor)
            predicted_sentences_corpus.extend(predicted_sentences)  # add them to the corpus of translations

            # Get the token and not id version of GT (ground-truth) sentences
            trg_token_ids_batch = trg_token_ids_batch.cpu().numpy()
            for target_sentence_ids in trg_token_ids_batch:
                target_sentence_tokens = [trg_field_processor.vocab.itos[id] for id in target_sentence_ids if id != pad_token_id]
                gt_sentences_corpus.append([target_sentence_tokens])  # add them to the corpus of GT translations

        bleu_score = corpus_bleu(gt_sentences_corpus, predicted_sentences_corpus)
        print(f'BLEU-4 corpus score = {bleu_score}, corpus length = {len(gt_sentences_corpus)}')
        return bleu_score
    
def greedy_decoding(baseline_transformer, src_representations_batch, src_mask, tgt_vocab, max_target_tokens=300):
   
    device = 'cuda'
    pad_token_id = 1

    # Initial prompt is the beginning/start of the sentence token. Make it compatible shape with source batch => (B,1)
    target_sentences_tokens = [[2] for _ in range(src_representations_batch.shape[0])]
    trg_token_ids_batch = torch.tensor([token for token in target_sentences_tokens], device=device)
    
    
    # Set to true for a particular target sentence once it reaches the EOS (end-of-sentence) token
    is_decoded = [False] * src_representations_batch.shape[0]

    while True:
        trg_mask = get_tgt_mask(trg_token_ids_batch, pad_token_id)
        
        # Shape = (B*T, V) where T is the current token-sequence length and V target vocab size
        predicted_log_distributions = baseline_transformer.decode(trg_token_ids_batch, trg_mask, src_representations_batch, src_mask)

        # Extract only the indices of last token for every target sentence (we take every T-th token)
        num_of_trg_tokens = len(target_sentences_tokens[0])
        predicted_log_distributions = predicted_log_distributions[num_of_trg_tokens-1::num_of_trg_tokens]

        # This is the "greedy" part of the greedy decoding:
        # We find indices of the highest probability target tokens and discard every other possibility
        most_probable_last_token_indices = torch.argmax(predicted_log_distributions, dim=-1).cpu().numpy()
        
        # Find target tokens associated with these indices
        predicted_words = [index for index in most_probable_last_token_indices]

        for idx, predicted_word in enumerate(predicted_words):
            target_sentences_tokens[idx].append(predicted_word)

            if predicted_word == 3:  # once we find EOS token for a particular sentence we flag it
                is_decoded[idx] = True

        if all(is_decoded) or num_of_trg_tokens == max_target_tokens:
            break

        # Prepare the input for the next iteration (merge old token ids with the new column of most probable token ids)
        trg_token_ids_batch = torch.cat((trg_token_ids_batch, torch.unsqueeze(torch.tensor(most_probable_last_token_indices, device=device), 1)), 1)

    # Post process the sentences - remove everything after the EOS token
 
    return target_sentences_tokens

