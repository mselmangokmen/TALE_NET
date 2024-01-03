import torch 
from functions import     test_model_image, test_model_image_from_file
import numpy as np
from scipy.io import loadmat
import os 
from PIL import Image

from model.talenet import TaleNet 

test_path = "dataset/testsets/BSD68"   
test_dirs = os.listdir(test_path)
output_path= "BSD68_results"

img_size=256 
test_dirs = [d  for d in test_dirs if not d.startswith('._')  ]

test_list=[] 
for t in test_dirs:
    test_list.append(os.path.join(test_path,t))     
 
 
channel_mult=18 

noise_level = 0.5

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') 
channel_size=1  

model_path = 'AirUNet_Waterloo_50' 
model_name='TALE_NET'
tunnel_size=4
total_score=0 
best_psnr=0
total_score=0
results=''  
           
model = TaleNet(out_channel=1,  channel_mult=channel_mult,tunnel_size=tunnel_size).to(device=device)


model.load_state_dict(torch.load(model_path))     

for i,img_path in enumerate(test_list):     
            #print(x_test[i].shape)
    result,psnr_score = test_model_image_from_file(clean_imag_path= img_path,model=model,device=device,image_index=i,model_name=model_name
                ,noise_level=noise_level,save_option=True,output_path=output_path)
                #print(x_test[i].shape)
    result = str(i) + ": " +  result 
    print( result)
    if psnr_score>best_psnr:
                best_psnr= psnr_score
    total_score+=psnr_score
    results +=result
    results+='best psnr score: '+ str(best_psnr)+'\n'
    results+='avg psnr score: '+ str(total_score/len(test_list))+'\n'

    text_file = open(output_path+ '/'+model_name +'_'+str(int(noise_level*100))+'.txt', "w+")
    text_file.write(results)
    text_file.close()

 