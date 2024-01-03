#train_inits= [{'model_type':3,'model_name':"RatUNet"},{'model_type':1,'model_name':"ResUNet"},{'model_type':2,'model_name':"Attention_Unet"},]
#train_inits= [  {'model_type':2,'model_name':"Attention_Unet"},{'model_type':4,'model_name':"ours"} , ]
import gc
import torch
from dataloader import myDataLoaderWaterlooGrayTrain 
 
from functions import train_model 

import torch.optim.lr_scheduler as lr_scheduler
from model.talenet import TaleNet  
batch_size =  30
num_epochs = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
channel_mult = 18  
lr=0.001
tunnel_size=4  
  
rayleigh_list = [ 0.5,0.25,0.15 ]
dataloaders = myDataLoaderWaterlooGrayTrain(batch_size=batch_size, file_name="waterloo_train").getDataLoader()
model_name="TALE_Net"
model = TaleNet(out_channel=1 ,channel_mult=channel_mult,tunnel_size=tunnel_size).to(device=device)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False, maximize=False, foreach=None, capturable=False)
for r in rayleigh_list:
    gc.collect()
    torch.cuda.empty_cache() 
    
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    train_model( dataloaders=dataloaders, model_name=model_name, noise_level=r, 
                model=model,optimizer=optimizer, device=device, lr_scheduler=scheduler,num_epochs=num_epochs,lr_step=False)
