
 
from torch import nn 
import numpy as np

from model.blocks.xunit import xUnitS  
class TunnelBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels,dilation=1,tunnel_size=4,pooling=True ):
        super().__init__() 
        kernel_size=3
        padding=1
        self.pooling=pooling
        self.conv_in_out=   nn.Sequential( # [10,1,64,64]
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,bias=True,dilation=dilation),   
        xUnitS(num_features=out_channels,batch_norm=False  ),
        nn.LeakyReLU( 0.1 ),
    )
         
        self.poolingLayer= nn.MaxPool2d(2) 
        
        layers = [] 
        dimensions= np.linspace(in_channels,out_channels, tunnel_size,dtype=np.int32)
        for i in range(len(dimensions)-1):
            #layer= DoubleConv(dimensions[i], dimensions[i+1] )
            layer= nn.Conv2d(dimensions[i], dimensions[i+1],kernel_size=kernel_size,padding=padding )
            layers.append(layer) 
              #layers.append( nn.GroupNorm(( dimensions[i+1]//tunnel_size)*2,dimensions[i+1]))
            layers.append( nn.GroupNorm( dimensions[i+1]//2, dimensions[i+1]))
            layers.append(nn.LeakyReLU( 0.1 ))   
            if self.pooling and i==1:
                layers.append(nn.MaxPool2d(2))   
              
        self.tunnel = nn.Sequential(*layers)   
         
    def forward(self, x):   
        res= x.clone()
        #res= self.unit_act_d(res)    
        x= self.tunnel(x)
        res= self.conv_in_out(res)

        if self.pooling:
           res= self.poolingLayer(res) 
        x=  res- x
        return  x
 