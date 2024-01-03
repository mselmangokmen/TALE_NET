import numpy as np  
import os  
from PIL import Image, ImageOps  
from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt
from scipy.io import savemat

train_path = "dataset/train" 
test_path = "dataset/test" 
train_dirs = os.listdir(train_path)
test_dirs = os.listdir(test_path)
img_size=128
train_dirs = [d  for d in train_dirs if not d.startswith('._')  ]
test_dirs = [d  for d in test_dirs if not d.startswith('._')  ]
#train_dirs = train_dirs[:len(train_dirs)//2]
#test_dirs = test_dirs[:len(test_dirs)//2]
train_list=[] 
test_list=[] 

for t in train_dirs:
    train_list.append(os.path.join(train_path,t))    

for t in test_dirs:
    test_list.append(os.path.join(test_path,t))     

train_images = np.zeros(shape=(len(train_list),img_size,img_size) ) 
test_images = np.zeros(shape=(len(test_list),img_size,img_size) ) 

for i in range(len(train_list)): 

    train_img = Image.open(train_list[i])  
    train_img= ImageOps.grayscale(train_img)  
    
    train_images[i,:,:]= train_img 

for i in range(len(test_list)): 

    test_img = Image.open(test_list[i])  
    test_img= ImageOps.grayscale(test_img)  
    
    test_images[i,:,:]= test_img 
 


train_images, val_images  = train_test_split(train_images, test_size=0.25 )  
print(train_images.shape)
print(val_images.shape)
print(test_images.shape)

mymatfiletrain = dict()
mymatfiletrain['x_train']=train_images 
mymatfiletrain['x_val']=val_images 
mymatfiletest = dict()
mymatfiletest['x_test']=test_images
##US4 and nerve segmentation
savemat("dataset/ultrasound/dataset_train.mat", mymatfiletrain) 
savemat("dataset/ultrasound/dataset_test.mat", mymatfiletest) 