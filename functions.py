
import os
import torch
import torch.nn as nn
import gc   
import copy 
import numpy as np  
from piq import   SSIMLoss,   ssim  
import ssl
import matplotlib.pyplot as plt 
from skimage.metrics import structural_similarity as ssim_scikit
from skimage.metrics import peak_signal_noise_ratio as psnr_scikit 

import cv2 as cv2 

ssl._create_default_https_context = ssl._create_unverified_context  
import bm3d    
 
from torchvision import transforms 
from torchvision.utils import save_image

from PIL import Image, ImageOps 
def norm_array(x):

    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    return x_norm

 


def add_noise_torch(img_tensor, scale,std): 
    # PyTorch tensorünü NumPy dizisine dönüştürün
    
    #if data_type=="UNS":
    #print(img_array.shape)
    
    img_array = img_tensor.cpu().numpy()
    std=0.3
    rayleigh_noise= generate_rayleigh(img_tensor.shape,std)
    #rayleigh_noise=np.clip(rayleigh_noise,-1,1)*scale
    #noise= np.random.normal(0,1,size=img_tensor.shape)*scale
    noisy_img_array =   img_array + (rayleigh_noise*scale)
    #noisy_img_array=np.clip(noisy_img_array,-1,1)
    noisy_img_tensor = torch.from_numpy(noisy_img_array).float()
    return noisy_img_tensor 


def add_noise_torch_to_img(img_array, scale,std):  

     
    std=0.3
    rayleigh_noise= generate_rayleigh(img_array.shape,std)
    #rayleigh_noise=np.clip(rayleigh_noise,-1,1)*scale
    #noise= np.random.normal(0,1,size=img_tensor.shape)*scale
    noisy_img_array =   img_array + (rayleigh_noise*scale)
    #noisy_img_array=np.clip(noisy_img_array,-1,1) 
    return noisy_img_array 


def generate_rayleigh(shape, std):
    randoms=  np.random.random(size=shape)
    #std=np.sqrt(1)
    rayleigh= std*np.sqrt(-2* np.log2(randoms))
    return rayleigh
    
 
def BM3D_denoise( noise_level,clean_imag_path,clean_image_path,noisy_image_path,estimated_image_path ): 
  
        clean_image = Image.open(clean_imag_path) 
        clean_image= ImageOps.grayscale(clean_image)  
        clean_image= np.array(clean_image)
        clean_image =  clean_image/255
        noisy_image = add_noise_torch_to_img(clean_image,noise_level,0.3)
        estimated_image= bm3d.bm3d(noisy_image*255, sigma_psd=noise_level*100) 
        estimated_image=estimated_image/255 

        ssim_value, _ = ssim_scikit(clean_image, estimated_image, full=True,data_range=1)

        # PSNR değerini hesapla
        psnr_value = psnr_scikit(clean_image, estimated_image)

        clean_image= Image.fromarray((clean_image * 255).astype(np.uint8), mode='L')
        noisy_image= Image.fromarray((noisy_image * 255).astype(np.uint8), mode='L')
        print(np.amax(estimated_image))
        estimated_image= Image.fromarray((estimated_image * 255).astype(np.uint8 ), mode='L')
        clean_image.save(clean_image_path)
        noisy_image.save(noisy_image_path)
        estimated_image.save(estimated_image_path) 
        result = 'model name: BM3D noise level: '+ str(noise_level)+' ssim score: '+ str(ssim_value)+'\n'+' psnr score: '+ str(psnr_value)+'\n'
        return result,psnr_value 
      
  
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

 

def PSNR(target, pred):
    mse = torch.mean((target - pred) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def lossFun(pred, target,device):
    #softmax = nn.Softmax(dim=1)
    #mse_criterion = nn.L1Loss()  
    mse_criterion = nn.L1Loss( ).to(device=device)   
    bce_criterion = nn.BCELoss( ).to(device=device)     
    mse_criterion = nn.MSELoss( ).to(device=device)   
    ssim_criterion = SSIMLoss(data_range=1.)  
  
    ssim_score = ssim(pred,target, data_range=1.) # normal ssim score digerleri icin
    lossClass= mse_criterion(pred, target) * 100
    lossClass_bce= bce_criterion(pred, target)

    psnr_score = PSNR(target,pred)
    mse_loss= mse_criterion(pred, target) * 100
    lossClass_ssim= ssim_criterion(pred, target).to(device=device)
    #print(lossClass.item())
    return lossClass, ssim_score,psnr_score,lossClass_bce,lossClass_ssim,mse_loss

 
 
def calc_loss_and_score(pred, target,device, metrics):  
    
    l1_loss,ssim_score,psnr_score,bce_loss,lossClass_ssim,mse_loss = lossFun(pred, target,device) 
    metrics['loss'].append( l1_loss.item() )   
    metrics['score'].append( ssim_score.item() )  
    metrics['psnr'].append( psnr_score.item() )  
    metrics['bce_loss'].append( bce_loss.item() )  
    metrics['mse_loss'].append( mse_loss.item() )  
    metrics['ssim_loss'].append( lossClass_ssim.item() )  
    return l1_loss


def calc_loss_and_score_test(pred, target,device, metrics): 
 
    
    l1_loss,ssim_score,psnr_score,bce_loss,lossClass_ssim,mse_loss = lossFun(pred, target,device) 
    metrics['loss'] = l1_loss.item() 
    metrics['score'] =  ssim_score.item() 
    metrics['psnr'] =  psnr_score.item() 
    metrics['bce_loss'] =  bce_loss.item() 
    metrics['ssim_loss'] =  lossClass_ssim.item() 
    metrics['mse_loss'] =  mse_loss.item() 

    metrics['score_list'].append( ssim_score.item() )  
    metrics['loss_list'].append( l1_loss.item() )   
    metrics['psnr_list'].append( psnr_score.item() )  
    metrics['bce_loss_list'].append( bce_loss.item() )  
    metrics['mse_loss_list'].append( mse_loss.item() )  
    metrics['ssim_loss_list'].append( lossClass_ssim.item() )  
    return l1_loss.item(),ssim_score.item(),psnr_score.item()
 
 
def print_metrics(main_metrics_train,model_name,main_metrics_val,metrics, phase,epoch, num_epochs,rayleigh):
    
    loss= metrics['loss'] 
    score= metrics['score'] 
    psnr= metrics['psnr'] 
    bce_loss= metrics['bce_loss'] 
    ssim_loss= metrics['ssim_loss'] 
    mse_loss= metrics['mse_loss'] 
    if(phase == 'train'):
        main_metrics_train['ssim_loss'].append( np.mean(ssim_loss)) 
        main_metrics_train['mse_loss'].append( np.mean(mse_loss)) 
        main_metrics_train['loss'].append( np.mean(loss)) 
        main_metrics_train['score'].append( np.mean(score)) 
        main_metrics_train['psnr'].append( np.mean(psnr)) 
        main_metrics_train['bce_loss'].append( np.mean(bce_loss)) 
        #main_metrics_train['accuracy'].append( accuracy ) 
    else:
        main_metrics_val['mse_loss'].append(np.mean(mse_loss)) 
        main_metrics_val['ssim_loss'].append(np.mean(ssim_loss)) 
        main_metrics_val['loss'].append(np.mean(loss)) 
        main_metrics_val['score'].append(np.mean(score)) 
        main_metrics_val['psnr'].append( np.mean(psnr)) 
        main_metrics_val['bce_loss'].append( np.mean(bce_loss)) 

    result = 'Epoch {}/{}'.format(epoch, num_epochs - 1)
    
    result += ' Raylegih level: '+ str(rayleigh) + ' model name : '+ str(model_name) +' \n'
    result += '-' * 10
    result += '\n'
    result += "phase: "+str(phase)  +  ' \nloss : {:4f}'.format(np.mean(loss))      +  ' \nscore : {:4f}'.format(np.mean(score))+  \
    ' \npsnr : {:4f}'.format(np.mean(psnr))    + ' \nbce_loss : {:4f}'.format(np.mean(bce_loss))     + ' \nssim_loss : {:4f}'.format(np.mean(ssim_loss))   \
     + ' \nmse_loss : {:4f}'.format(np.mean(mse_loss))     + '\n' 
    return result 

def print_test_metrics(test_metrics,  rayleigh,batch_num, model_name ):
    
    loss= test_metrics['loss'] 
    score= test_metrics['score']    
    psnr= test_metrics['psnr']   
    mse_loss= test_metrics['mse_loss']   
    bce_loss= test_metrics['bce_loss']   
    ssim_loss= test_metrics['ssim_loss']   
    result="" 
    result += '\nBest Batch Results : ' + str(batch_num) +' \n'
    result += 'Model Name : '+ model_name + ' \n'
    result += 'Raylegih level: '+ str(rayleigh) + ' \n' 
    result +=  'Loss : {:4f}'.format(loss)      +"\n"  +  'score : {:4f}'.format(score)    +"\n"     +  'psnr : {:4f}'.format(psnr)   +"\n"   + 'bce_loss : {:4f}'.format(bce_loss)   \
    +"\n" + 'ssim_loss : {:4f}'.format(ssim_loss)  \
    + ' \nmse_loss : {:4f}'.format(mse_loss)     + '\n' 
    return result 

def print_avg_test_metrics(test_metrics,  rayleigh, model_name ): 
    loss= test_metrics['loss_list'] 
    score= test_metrics['score_list']    
    psnr= test_metrics['psnr_list']    
    mse_loss= test_metrics['mse_loss_list']    
    bce_loss= test_metrics['bce_loss_list']     
    ssim_loss= test_metrics['ssim_loss_list']     
    
    result="" 
    result += '=' * 10 
    result += 'Average Results :   \n'
    result += 'Model Name : '+ model_name + ' \n' 
    result += 'Raylegih level: '+ str(rayleigh) + ' \n'
    result +=  'Loss : {:4f}'.format(np.mean(loss))      +  ' \nscore : {:4f}'.format(np.mean(score))     + ' \n' 
    result +=  'Psnr : {:4f}'.format(np.mean(psnr))    + ' \nbce_loss : {:4f}'.format(np.mean(bce_loss))      +"\n"
    result +=  'ssim_loss : {:4f}'.format(np.mean(ssim_loss))      +"\n"
    result +=  'mse_loss : {:4f}'.format(np.mean(mse_loss))       +"\n"
  
    return result 


def write_to_file(y1,y2,x,file_name):

    with open(file_name, 'w') as f: 
        for y in y1: 
            f.write(str(y)+ " ")

        f.write('\n')
        for y in y2: 
            f.write(str(y)+ " ") 
        f.write('\n')
        for x_val in x: 
            f.write(str(x_val)+ " ") 
        f.write('\n')

def print_save_figure(train_dict,val_dict,num_epochs,fname): 
  fig, ax = plt.subplots( nrows=1, ncols=1 ) 
  ax.plot(range(1,num_epochs+1), train_dict['loss'], label='Train')
  ax.legend(loc="upper right")
  ax.plot(range(1,num_epochs+1), val_dict['loss'], label='Validation')
  ax.legend(loc="upper right")
  write_to_file(train_dict['loss'],val_dict['loss'],[i for i in range(1,num_epochs+1)],fname+'_loss.txt')
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Loss")
  fig.savefig(fname+'_loss.png') 
  plt.close(fig) 
  fig, ax = plt.subplots( nrows=1, ncols=1 ) 
  ax.plot(range(1,num_epochs+1), train_dict['score'], label='Train')
  ax.legend(loc="lower right")
  ax.plot(range(1,num_epochs+1), val_dict['score'], label='Validation')
  ax.legend(loc="lower right")

  write_to_file(train_dict['score'],val_dict['score'],[i for i in range(1,num_epochs+1)],fname+'_accuracy.txt')
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Accuracy")

  fig.savefig(fname+'_accuracy.png') 
  plt.close(fig)

  fig, ax = plt.subplots( nrows=1, ncols=1 ) 
  ax.plot(range(1,num_epochs+1), train_dict['psnr'], label='Train')
  ax.legend(loc="lower right")
  ax.plot(range(1,num_epochs+1), val_dict['psnr'], label='Validation')
  ax.legend(loc="lower right")

  write_to_file(train_dict['psnr'],val_dict['psnr'],[i for i in range(1,num_epochs+1)],fname+'_psnr.txt')
  ax.set_xlabel("Epochs")
  ax.set_ylabel("PSNR")

  fig.savefig(fname+'_psnr.png') 
  plt.close(fig)

  fig, ax = plt.subplots( nrows=1, ncols=1 ) 
  ax.plot(range(1,num_epochs+1), train_dict['bce_loss'], label='Train')
  ax.legend(loc="upper right")
  ax.plot(range(1,num_epochs+1), val_dict['bce_loss'], label='Validation')
  ax.legend(loc="upper right")

  write_to_file(train_dict['bce_loss'],val_dict['bce_loss'],[i for i in range(1,num_epochs+1)],fname+'_bce_loss.txt')
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Loss")

  fig.savefig(fname+'_bce_loss.png') 
  plt.close(fig)

def print_save_test_figure(test_dict,batch_num,fname): 
  fig, ax = plt.subplots( nrows=1, ncols=1 ) 
  ax.plot(range(1,batch_num+1), test_dict['loss_list'], label='Test') 

  ax.set_xlabel("Epochs")
  ax.set_ylabel("Loss")

  fig.savefig(fname+'_loss.png')  
  plt.close(fig) 

  fig, ax = plt.subplots( nrows=1, ncols=1 ) 
  ax.plot(range(1,batch_num+1), test_dict['score_list'], label='Test') 

  ax.set_xlabel("Epochs")
  ax.set_ylabel("Accuracy")

  fig.savefig(fname+'_accuracy.png')  
  plt.close(fig)


  fig, ax = plt.subplots( nrows=1, ncols=1 ) 
  ax.plot(range(1,batch_num+1), test_dict['psnr_list'], label='Test') 

  ax.set_xlabel("Epochs")
  ax.set_ylabel("PSNR")

  fig.savefig(fname+'_psnr.png') 
  plt.close(fig)

  fig, ax = plt.subplots( nrows=1, ncols=1 ) 
  ax.plot(range(1,batch_num+1), test_dict['bce_loss_list'], label='Test') 

  ax.set_xlabel("Epochs")
  ax.set_ylabel("Loss")

  fig.savefig(fname+'_bce_loss.png') 
  plt.close(fig)

 


def train_model(dataloaders,model_name,model,optimizer,noise_level,device, lr_scheduler,num_epochs=100 ,lr_step=False): 
 
    train_dict= dict()
    train_dict['loss']= list()  
    train_dict['score']= list() 
    train_dict['psnr']= list() 
    train_dict['bce_loss']= list()  
    train_dict['ssim_loss']= list()  
    train_dict['mse_loss']= list()  
    val_dict= dict()
    val_dict['loss']= list()  
    val_dict['score']= list()  
    val_dict['psnr']= list()  
    val_dict['bce_loss']= list()  
    val_dict['ssim_loss']= list()  
    val_dict['mse_loss']= list()  


    model_path = model_name

    isExist = os.path.exists(model_path)
    if not isExist:
 
        os.makedirs(model_path)
    model_name_new = model_name +'_ray_' +str(int(noise_level*100)) 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 99999
    best_score = -99999 
    best_psnr = -99999 
    train_string = "" 
    temp_str=""
      

    for epoch in range(num_epochs):
          #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
          #print('-' * 10) 

        for phase in ['train', 'val']: # phase = train
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()
                   # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = dict()
            metrics['loss']=list()
            metrics['score']=list()
            metrics['psnr']=list()
            metrics['bce_loss']=list()
            metrics['mse_loss']=list()
            metrics['ssim_loss']=list()
              #metrics['correct']=0
              #metrics['total']=0
            
            for inputs in dataloaders[phase]: 
                  # inputs = [75, 1, 64,64]
                  # labels = [75, 1, 64,64]  GT image
                  labels = inputs.clone() 

                  #inputs = add_rayleigh_noise_torch(inputs,noise_level) 
                  inputs = add_noise_torch(inputs,noise_level,noise_level)  
                      
                  inputs = inputs.to(device=device, dtype=torch.float) 
                  labels = labels.to(device=device, dtype=torch.float)
                  #save_tensor_as_image(inputs,'noise_image_mult'+str(int(noise_level*100)))
                  #save_tensor_as_image(labels,'clean_image_mult'+str(int(noise_level*100))) 
                   # noisy_images = [75, 1, 64,64]# expected output
                  
                  # zero the parameter gradients
                  optimizer.zero_grad()

                  with torch.set_grad_enabled(phase == 'train'):
                      outputs = model(inputs) 
                      #save_tensor_as_image(outputs,'train')
                      #print('outputs size: '+ str(outputs.size()) )
                      loss = calc_loss_and_score(pred= outputs,target= labels,device=device, metrics= metrics)   
                      # backward + optimize only if in training phase
                      if phase == 'train':
                          loss.backward()
                          optimizer.step()

                  # statistics
                  #print('epoch samples: '+ str(epoch_samples)) 
            if lr_step:
                lr_scheduler.step()
            epoch_result= print_metrics(main_metrics_train=train_dict, rayleigh=noise_level, model_name=model_name,
            main_metrics_val=val_dict,metrics=metrics,phase=phase,epoch= epoch, num_epochs=num_epochs )
            train_string +=epoch_result
            print(epoch_result)
            epoch_loss = np.mean(metrics['loss'])
            epoch_score= np.mean(metrics['score'])
            epoch_psnr= np.mean(metrics['psnr'])

            if phase == 'val' and epoch_psnr > best_psnr:
                      #print("saving best model")
                      best_psnr = epoch_psnr  
                      best_model_wts = copy.deepcopy(model.state_dict())
                      temp_str= '\n'+ ('-'*10) + '\n' + 'best psnr : '+ str(epoch_psnr)+'\n' \
                      + 'epoch num : '+ str(epoch)+'\n'+ ('-'*10) + '\n' 
    train_string+=temp_str
    text_file = open(model_name +'/'+ model_name_new +'.txt', "w+")
    text_file.write(train_string)
    text_file.close()
    print('Best psnr val: {:4f}'.format(best_psnr))

    torch.save(best_model_wts, model_name +'/'+ model_name_new)
    print_save_figure(train_dict,val_dict,num_epochs,model_name +'/'+ model_name_new)


def test_model(model_name,model,test_loader,noise_level,device ): 
 
  isExist = os.path.exists('test_results')
  if not isExist: 
        os.makedirs('test_results')
  model.eval()  
  test_dict= dict()
  test_dict['loss_list']= list()  
  test_dict['score_list']= list()   
  test_dict['psnr_list']= list()   
  test_dict['bce_loss_list']= list()   
  test_dict['ssim_loss_list']= list()  
  test_dict['mse_loss_list']= list()   
  test_dict['loss']= 0.0
  test_dict['score']= 0.0
  test_dict['psnr']= 0.0
  test_dict['bce_loss']= 0.0
  test_dict['ssim_loss']= 0.0
  test_dict['mse_loss']= 0.0
  model_name_new = model_name +'_avg_ray_' +str(int(noise_level*100)) 
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 99999 
  best_score=0.0
  best_batch_result=""  
  best_batch_psnr=""
  batch_result= ''
  batch_cnt=0
  best_psnr=0
  for inputs in test_loader:  
    with torch.no_grad():
    
      labels = inputs.clone() 

      inputs = add_noise_torch(inputs,noise_level,noise_level)       
      #inputs = add_rayleigh_noise_torch(inputs,noise_level) 
                  # noisy_images = [75, 1, 64,64]
 
      inputs = inputs.to(device=device,dtype=torch.float) 
      labels = labels.to(device=device,dtype=torch.float )
      outputs = model(inputs) 
      
      loss,score,psnr = calc_loss_and_score_test(pred= outputs,target= labels,device=device, metrics= test_dict)     
      
      batch_result= print_test_metrics(test_metrics = test_dict,  rayleigh= noise_level,batch_num= batch_cnt,model_name= model_name )

      batch_cnt +=1 
      #print(batch_result)
      #if best_loss > loss:
      #  best_loss = loss   
      #  best_batch_result= batch_result  
      if best_score < score:
        best_score = score   
        best_batch_result= batch_result  

      if best_psnr < psnr:
        best_psnr = psnr   
        best_batch_psnr= batch_result  

  total_result= print_avg_test_metrics(test_metrics = test_dict,  rayleigh= noise_level,model_name= model_name  )
  total_result += "\nBest Batch for SSIM Score:"+ best_batch_result
  total_result += "\nBest Batch for PSNR Score:"+ best_batch_psnr
  text_file = open('test_results/'+model_name_new +'.txt', "w+")
  text_file.write(total_result)
  text_file.close()
  print(total_result) 
  #print_save_test_figure(test_dict,batch_cnt,model_name_new) 


 
def PSNR_numpy(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def test_model_image(clean_image,image_index,model_name ,model,noise_level,device,save_option):
    #result_path = 'ultrasound_test_results/'+model_name+'/image'+str(image_index)
    result_path = 'ultrasound_test_results/'+model_name+'/image'+str(image_index)
    clean_image_path = result_path + '/clean_image'+str(image_index)+'_'+str(int(noise_level*100))+'.png'
    estimated_image_path =  result_path+'/estimated_image'+str(image_index)+'_'+str(int(noise_level*100))+'.png'
    noisy_image_path =  result_path+'/noisy_image'+str(image_index)+'_'+str(int(noise_level*100))+'.png'
    isExist = os.path.exists(result_path)

    
    if not isExist and save_option==True: 
        os.makedirs(result_path)

    if "BM3D" in model_name:  
        clean_image=torch.unsqueeze(clean_image, dim=0)
        clean_image=clean_image.to(device=device, dtype=torch.float) 
        save_image(clean_image, clean_image_path ,normalize=True )  
        result,psnr_value = BM3D_denoise(clean_imag_path=clean_image_path,clean_image_path=clean_image_path,
                estimated_image_path=estimated_image_path,noise_level=noise_level,noisy_image_path=noisy_image_path)
        return result,psnr_value 
    
    else:

        gc.collect()
        torch.cuda.empty_cache()  
        model.eval() 
        #if(len(clean_image.shape)==3):

            #clean_image = np.mean( clean_image, axis=0) 
        
        ssim_score = 0 
        psnr_score = 0 
        result=''
        clean_image=torch.unsqueeze(clean_image, dim=0)
        clean_image=clean_image.to(device=device, dtype=torch.float)    
        with torch.no_grad():

            noisy_image = add_noise_torch(clean_image,noise_level,noise_level)   
    
            noise_tensor = noisy_image.to(device=device, dtype=torch.float)   
            estimated_image = model(noise_tensor).to(device=device )  

            ssim_score = ssim(clean_image,estimated_image, data_range=1.)  
            
            psnr_score = PSNR(clean_image,estimated_image) 
            

            save_image(estimated_image, estimated_image_path ,normalize=True )
            save_image(noisy_image, noisy_image_path ,normalize=True  )
            save_image(clean_image, clean_image_path ,normalize=True  )  
        result = 'model name: '+ model_name + ' noise level: '+ str(noise_level)+' ssim score: '+ str(ssim_score.item())+'\n'+' psnr score: '+ str(psnr_score.item())+'\n'
        return result,psnr_score.item()
 
 


def test_model_image_from_file(clean_imag_path,output_path,image_index,model_name ,model,noise_level,device,save_option,img_size=256):
    #result_path = 'ultrasound_test_results/'+model_name+'/image'+str(image_index)

    clean_image = Image.open(clean_imag_path) 
    clean_image= ImageOps.grayscale(clean_image)  
    orj_w= (clean_image.width //2)*2
    orj_h= (clean_image.height//2)*2

    if "RatUNet" in model_name:
        clean_image=clean_image.resize(size=(img_size,img_size))
    else:
        clean_image= clean_image.resize(size=(orj_w,orj_h))
    
    clean_image= np.array(clean_image)
    clean_image =  clean_image/255  

    trans_to_tensor = transforms.Compose([
  transforms.ToTensor(), 
])
    clean_image = trans_to_tensor(clean_image) 
    #clean_image= torch.unsqueeze(clean_image,dim=0)

    result_path = output_path + '/'+model_name+'/image'+str(image_index)
    clean_image_path = result_path + '/clean_image'+str(image_index)+'_'+str(int(noise_level*100))+'.png'
    estimated_image_path =  result_path+'/estimated_image'+str(image_index)+'_'+str(int(noise_level*100))+'.png'
    noisy_image_path =  result_path+'/noisy_image'+str(image_index)+'_'+str(int(noise_level*100))+'.png' 
    isExist = os.path.exists(result_path)
    if not isExist and save_option==True: 
        os.makedirs(result_path)

    gc.collect()
    torch.cuda.empty_cache()  
    if model is not None:
        model.eval() 
    #if(len(clean_image.shape)==3):

        #clean_image = np.mean( clean_image, axis=0) 
    
    ssim_score = 0 
    psnr_score = 0 
    result=''

    if "BM3D" in model_name:  
        result,psnr_value = BM3D_denoise(clean_imag_path=clean_imag_path,clean_image_path=clean_image_path,
                estimated_image_path=estimated_image_path,noise_level=noise_level,noisy_image_path=noisy_image_path)
        return result,psnr_value 
    
    else:

        clean_image=torch.unsqueeze(clean_image, dim=0)
        clean_image=clean_image.to(device=device, dtype=torch.float)    
        with torch.no_grad(): 
            noisy_image = add_noise_torch(clean_image,noise_level,noise_level)   

            noise_tensor = noisy_image.to(device=device, dtype=torch.float)   
            estimated_image = model(noise_tensor).to(device=device )   
            
            #estimated_image= trans_to_tensor(estimated_image).to(device=device, dtype=torch.float )  
            ssim_score = ssim(clean_image,estimated_image, data_range=1.)  
            
            psnr_score = PSNR(clean_image,estimated_image)  
            

            save_image( (estimated_image ), estimated_image_path ,normalize=True  )
            save_image( (noisy_image ), noisy_image_path ,normalize=True  )
            save_image( (clean_image ), clean_image_path ,normalize=True  ) 
 
  
    result = 'model name: '+ model_name + ' noise level: '+ str(noise_level)+' ssim score: '+ str(ssim_score.item())+'\n'+' psnr score: '+ str(psnr_score.item())+'\n'
    return result,psnr_score.item()

def save_tensor_as_image(x,category):
  tensor_len = x.shape[0]
  tensor_len = 1
  #print(x.shape)
  tensor_imgs= 'tensor_images'
  tensor_imgs= os.path.join(tensor_imgs, category)
  if not os.path.exists(tensor_imgs):
      os.mkdir(tensor_imgs) 
  
  for i in range(tensor_len):
    img_folder = os.path.join(tensor_imgs, "img_"+str(i))
    if not os.path.exists(img_folder):
      os.mkdir(img_folder) 
    if x.shape[1]>30:
        for j in range(30):
            img_path= os.path.join(img_folder, "img_"+str(i)+"_"+str(j)+".png")
            img_tensor= x[i,j,:,:]
            save_image(torch.clip(img_tensor), img_path)
    else:
        img_path= os.path.join(img_folder, "img_"+str(i)+".png")
        img_tensor= x[i,0,:,:]
        save_image(torch.clip(img_tensor), img_path)

 