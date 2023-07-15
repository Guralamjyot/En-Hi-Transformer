from datapipeline import *

checkpoint=torch.load('best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.to('cuda')


source_sen='might as well hire me?'
example=[]
for x in [source_sen]:
    s=source_transform(x.rstrip('\n'))
    example.append(s)
example=pad_sequence(example,padding_value=1,batch_first=True)
example_mask=get_src_mask(example)
src_rep=model.encode(example,example_mask)
x=transformer.greedy_decoding(model,src_rep,example_mask,target_vocab,max_target_tokens=50)
x=x[0]
list=[]
list.append([target_vocab.lookup_token(idx) for idx in x])
sen=list[0]
listToStr = ' '.join([str(elem) for elem in sen])
print(listToStr)