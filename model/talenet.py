

import torch 
from torch import nn
from model.blocks.attentiongate import AttentionGate
from model.blocks.tunnelblock import TunnelBlock  
from torchvision.utils import save_image
import os 
class TaleNet(nn.Module):


    def __init__(self, out_channel=1,channel_mult=16,in_channel=1,tunnel_size=5):

        super().__init__()   
        self.dilation=1   
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, 
                                    mode='bilinear', align_corners=True,)    
        self.sigmoid = nn.Sigmoid() 
        
        self.dconv_down1 = TunnelBlock(in_channel, channel_mult,tunnel_size=tunnel_size) 
        self.dconv_down2 = TunnelBlock(channel_mult, channel_mult*2,tunnel_size=tunnel_size)
        self.dconv_down3 = TunnelBlock(channel_mult*2, channel_mult*4,tunnel_size=tunnel_size)
        self.dconv_down4 = TunnelBlock(channel_mult*4, channel_mult*8,tunnel_size=tunnel_size)
        self.dconv_down5 = TunnelBlock(channel_mult*8, channel_mult*16,tunnel_size=tunnel_size,pooling=False)  
 
        self.Att4 = AttentionGate(F_g=channel_mult*16,F_l=channel_mult*8,F_int=channel_mult*8) 
        self.Att3 = AttentionGate(F_g=channel_mult*8,F_l=channel_mult*4,F_int=channel_mult*4)
        self.Att2 = AttentionGate(F_g=channel_mult*4,F_l=channel_mult*2,F_int= channel_mult*2)
        self.Att1 = AttentionGate(F_g=channel_mult*2,F_l=channel_mult,F_int=channel_mult)

        self.dconv_up4 = TunnelBlock(channel_mult*16 + channel_mult*8 , channel_mult*8 ,tunnel_size=tunnel_size,pooling=False )
        self.dconv_up3 = TunnelBlock(channel_mult*8 + channel_mult*4 , channel_mult*4 ,tunnel_size=tunnel_size,pooling=False)
        self.dconv_up2 = TunnelBlock(channel_mult*4  + channel_mult*2, channel_mult*2,tunnel_size=tunnel_size,pooling=False )
        self.dconv_up1 = TunnelBlock(channel_mult*2 + channel_mult , channel_mult*2,tunnel_size=tunnel_size,pooling=False )
        self.conv_last = nn.Conv2d(channel_mult*2, out_channel, 1)
          
    def forward(self, x): # [10,1,64,64]   =1)       
       
        conv1 = self.dconv_down1(x)
        #x = self.maxpool(conv1)

        conv2 = self.dconv_down2(conv1)
        #x = self.maxpool(conv2)

        conv3 = self.dconv_down3(conv2)
        #x = self.maxpool(conv3)

        conv4 = self.dconv_down4(conv3)
        #x = self.maxpool(conv4)

        x = self.dconv_down5(conv4)   

        #print(x.shape)
        #print(conv4.shape)
        
        x = torch.cat([  self.Att4(g=x,x=conv4),x], dim=1)
        x = self.upsample(x)  
        x = self.dconv_up4(x)  

        x = torch.cat([  self.Att3(g=x,x=conv3),x], dim=1)
        x = self.upsample(x) 
        x = self.dconv_up3(x) 
 
        att2= self.Att2(g=x,x=conv2)
 
        x = torch.cat([ att2 ,x], dim=1)
        x = self.upsample(x)  
        x = self.dconv_up2(x) 
        
        x = torch.cat([  self.Att1(g=x,x=conv1),x], dim=1)
        x = self.upsample(x) 
        x = self.dconv_up1(x)
        x = self.conv_last(x)
        out = self.sigmoid(x)
 
        return out    
       
 