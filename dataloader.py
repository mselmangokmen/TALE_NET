
import torch 
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms  

import numpy as np
from scipy.io import loadmat
class myDatasetWaterloo(Dataset):
  def __init__(self, x,gray=False):
    #x = [500,64,64]
    self.input_images = x  
    self.gray=gray
    #y = [500,64,64]
  def __len__(self):
    return len(self.input_images)

  def __getitem__(self, idx):
    image = self.input_images[idx]   
    image = image/255   

    trans = transforms.Compose([
  transforms.ToTensor(), 
  
])
     
    image_t = trans(image)  
    if  self.gray:
       image_t=torch.mean(image_t,dim=0)
       image_t=torch.unsqueeze(image_t,dim=0)

    #[1,64,64]
    return  image_t  



class myDataLoaderWaterlooGrayTrain():
    def __init__(self, batch_size,file_name ) -> None:
        
        data_path= "dataset/"+file_name+".mat"
        mymatfile = loadmat(data_path)

        x_train = np.array(mymatfile["x_train"] )  
        x_val = np.array(mymatfile["x_val"] ) 

        train_set= myDatasetWaterloo(x= x_train,gray=True )  
        val_set= myDatasetWaterloo(x= x_val ,gray=True )  
        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True), 
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True),  
        }
        self.dataloaders = dataloaders 
        
    def getDataLoader(self): 
        return self.dataloaders

class myDataLoaderWaterlooGrayTest():
    def __init__(self, batch_size,file_name ) -> None:
        
        data_path= "dataset/"+file_name+".mat"
        
        mymatfile = loadmat(data_path)
        x_test = np.array(mymatfile["x_test"] )  

        test_set= myDatasetWaterloo(x= x_test,gray=True )    

        dataloaders = {
            'test': DataLoader(test_set, batch_size=batch_size, shuffle=True  ) 
        }
        self.dataloaders = dataloaders 
        
    def getDataLoader(self): 
        return self.dataloaders
    