from datapipeline import *


checkpoint=torch.load('best.pt')
#model.load_state_dict(torch.load('best.pt'))
save_every=1
inference_epoch=3
translate_eg=False
valid=False
Loss=[]

model.load_state_dict(checkpoint['model_state_dict'])
optimizer=checkpoint['optimizer_state_dict']
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.to('cuda')

data=data()

for epoch in range(epochs):
    print(f"{epoch+1} of {epochs} epochs")
    model.train()
    for x in data:
        for src,tgt,tgt_gt in tqdm(x) :
            src.to('cuda')
            tgt.to('cuda')
            tgt_gt.to('cuda')
            src_mask=get_src_mask(src)
            tgt_mask=get_tgt_mask(tgt)
            predicted_log_dist=model(src,src_mask,tgt,tgt_mask)
            smooth_tgt=label_smoothing(tgt_gt)
            optimizer.zero_grad()

            loss=loss_fn(predicted_log_dist,smooth_tgt)          
            if not torch.isnan(loss):  #(dont need anymore, fixed with line 106)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

   

    model.eval()
    model_epoch=checkpoint['epoch']+1


    if ((epoch+1)%inference_epoch==0):
        print(f'translation of the sentence: {source_sen} by model trained for {epoch+1} epochs')
        torch.save(model.state_dict(),f'best_{epoch+1}.pt')
        with torch.no_grad():
            if translate_eg:
                example_mask=get_src_mask(example)
                src_rep=model.encode(example,example_mask)
                x=transformer.greedy_decoding(model,src_rep,example_mask,target_vocab,max_target_tokens=50)
                x=x[0]
                list=[]
                list.append([target_vocab.lookup_token(idx) for idx in x])
                sen=list[0]
                listToStr = ' '.join([str(elem) for elem in sen])
                print(listToStr)

    if valid==True:        
        loss=0
        i=0
        for src,tgt,tgt_gt in tqdm(valid_loader):
            i+=1
            src.to(DEVICE)
            tgt.to(DEVICE)
            tgt_gt.to(DEVICE)
            src_mask=get_src_mask(src)
            tgt_mask=get_tgt_mask(tgt)
            predicted_log_dist=model(src,src_mask,tgt,tgt_mask)
            smooth_tgt=label_smoothing(tgt_gt)
            loss+=loss_fn(predicted_log_dist,smooth_tgt)
        print(f"loss= {loss/i}")
        Loss.append(loss)

    if (epoch+1)%save_every==0:
        torch.save({
            'epoch':  model_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer,
            'loss': Loss,
            },'best.pt' )

torch.save({
            'epoch': model_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer,
            'loss': Loss,
            },'best.pt' )

