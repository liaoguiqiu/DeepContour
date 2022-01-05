#This is tht high compact version that share a lot of the layers
from model.networks import UnetSkipConnectionBlock
from model.networks import get_norm_layer
import test_model .layer_body_sheath_res2  as baseM 
# the NET dbody for the sheath contour tain  upadat 5th octo 2020
import torch
import torch.nn as nn
import  arg_parse
from arg_parse import kernels, strides, pads
from  dataset_sheath import Path_length,Batch_size,Resample_size
import torchvision.models
import numpy as np
import cv2
Out_c = 2 # depends on the bondaried to be preicted 
Input_c = 3  #  the gray is converted into 3 channnels image 
class _BackBoneUnet(nn.Module):
    def __init__(self,input_nc=3, output_nc=256, num_downs=8, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False ):
        super(_BackBoneUnet, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
        norm_layer = get_norm_layer(norm_type='instance')
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # here this is modifiy to be a feature extracter
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        
    def forward(self, x):
         
        
        return self.model(x) 

class _BackBonelayer (nn.Module):
    def __init__(self, inputd = Input_c):
        super(_BackBonelayer, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
   
        feature =16  

        self.side_branch1  =  nn.ModuleList()    
        
        self.side_branch1.append(  baseM.conv_keep_W(inputd,feature))# 128*256 - 64*128
        
        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))# 32*128  - 16*128
        feature = feature *2
        
       
        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))# 8*64  - 4*64
        feature = feature *2
        self.depth = feature
    def forward(self, x):

        side_out =x
        for j, name in enumerate(self.side_branch1):
            side_out = self.side_branch1[j](side_out)

      
        
        return side_out 
class _2LayerScale1(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self,backboneDepth,feature):
        super(_2LayerScale1, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
   
         

         
         
        #a side branch predict with original iamge with rectangular kernel
        # 256*256 - 128*256
        #limit=1024
        self.side_branch1  =  nn.ModuleList()    
          
        # 32*256  - 16*256
         
        self.side_branch1.append(  baseM.conv_keep_W(backboneDepth,2*feature))
        feature = feature *2
        # 16*256  - 8*256
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        #self.side_branch1.append(  conv_keep_all(feature, feature))

        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))
        feature = feature *2
        # 8*256  - 4*256
       
        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature)) # 4*256
        feature = feature *2
      

        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature,k=(4,1),s=(1,1),p=(0,0)))  # 2*256
         
    
    def forward(self, x):

        side_out =x
        for j, name in enumerate(self.side_branch1):
            side_out = self.side_branch1[j](side_out)

      
        
        return side_out 


class _2LayerScale2(nn.Module):
#output width=((W-F+2*P )/S)+1
    
    def __init__(self,backboneDepth,feature):
        super(_2LayerScale2, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
   
        

         
        #self.layer_num =Input_c
        #a side branch predict with original iamge with rectangular kernel
        #limit=1024
        self.side_branch1  =  nn.ModuleList()    
        
       
      

        self.side_branch1.append(  baseM.conv_dv_2(backboneDepth,2*feature))# 128*256 - 64*128
        feature = feature *2
        
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))# 64*128  - 32*128
        feature = feature *2
        
          
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        #self.side_branch1.append(  conv_keep_all(feature, feature))

        self.side_branch1.append(  baseM.conv_dv_2(feature,2*feature))# 16*128  - 8*64
        feature = feature *2
           

        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature,k=(4,1),s=(1,1),p=(0,0))) #2*64
         
    def forward(self, x):

        side_out =x
        for j, name in enumerate(self.side_branch1):
            side_out = self.side_branch1[j](side_out)
 
  
        return side_out
              
        #return out,side_out,side_out2
# mainly based on the resnet  


class _2LayerScale3(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self,backboneDepth,feature):
        super(_2LayerScale3, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
   
        

         
        #self.layer_num =Input_c
        #a side branch predict with original iamge with rectangular kernel
        #limit=1024
        self.side_branch1  =  nn.ModuleList()    
        

        self.side_branch1.append(  baseM.conv_dv_2(backboneDepth,2*feature))# 256*256 - 128*256
        feature = feature *2
        
         
      
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  baseM.conv_dv_2(feature,2*feature))# 64*128  - 32*128
        feature = feature *2
        
        
        
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        #self.side_branch1.append(  conv_keep_all(feature, feature))

        self.side_branch1.append(  baseM.conv_dv_2(feature,2*feature))# 16*128  - 8*64
        feature = feature *2
        
      

        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature,k=(4,1),s=(1,1),p=(0,0))) #2*64
        

    def display_one_channel(self,img):
        gray2  =   img[0,0,:,:].cpu().detach().numpy()*104+104
        cv2.imshow('down on one',gray2.astype(np.uint8)) 

    def forward(self, x):

        #m = nn.AdaptiveAvgPool2d((64,Path_length))
        #m = nn.AdaptiveMaxPool2d((64,Path_length))
        #m = nn. MaxPool2d((2,2))


        #AdaptiveMaxPool2d
        #x_s = m(x)
      
        #self.display_one_channel(x_s)
        #one chaneel:
        
        side_out =x
        for j, name in enumerate(self.side_branch1):
            side_out = self.side_branch1[j](side_out)
 
        return side_out 
class Fusion(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self,classfy = False):
        super(Fusion, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
        self. up2 = nn.ConvTranspose2d(512, 512,(1,4), (1,4), (0,0), bias=False)   
        self. up3 =   nn.ConvTranspose2d(512, 512,(1,8), (1,8), (0,0), bias=False)  
        self. fusion = nn.Conv2d(512+512+512,512,(1,3), (1,1), (0,1), bias=False)    # from 3 dpth branch to one   
        self. fusion2 = nn.Conv2d(512,512 ,(1,3), (1,1), (0,1), bias=False)    # from 3 dpth branch to one
        if (classfy == False):
            self. fusion3 = nn.Conv2d(512 ,Out_c,(1,3), (1,1), (0,1), bias=False)    # from 3 dpth branch to one   
        else:
            self. fusion3 = nn.Sequential(
              
             nn.Conv2d(512 ,Out_c,(1,3), (1,1), (0,1), bias=False), #2*64           
             #nn.BatchNorm2d(1),
             nn.Softmax()
                                 )
        self.tan_activation = nn.Tanh()
        
    def forward(self,side_out1,side_out2,side_out3):
        side_out2 = self. up2(side_out2)
        side_out3 =  self. up3(side_out3) 


        fuse=torch.cat((side_out1,side_out2,side_out3),1)
        fuse=self.fusion(fuse)
        fuse=self.fusion2(fuse)
        fuse=self.fusion3(fuse)


        

        local_bz,num,_,local_l = fuse.size() 

        out  = fuse.view(-1,num,local_l).squeeze(1)# squess before fully connected
        return out
       
 
# mainly based on the resnet  
class _2layerFusionNets_(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self,classfy = False,UnetBack_flag = True):
        super(_2layerFusionNets_, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
        self. UnetBack_flag = UnetBack_flag

        if UnetBack_flag == True:
            unetf = 100
            self.Unet_back = _BackBoneUnet( output_nc=unetf,use_dropout=True)
            self.pixencoding = baseM.conv_keep_all(unetf,1,k=(1,1),s=(1,1),p=(0,0),resnet=False,final=True)
            self.backbone =  _BackBonelayer(unetf)
        else:
            self.backbone =  _BackBonelayer()

        backboneDepth = self.backbone.depth
        feature = 32
        self.side_branch1  =  _2LayerScale1(backboneDepth,feature)
         
        self.side_branch2  =  _2LayerScale2(backboneDepth,feature)  
        self.side_branch3  =  _2LayerScale3(backboneDepth,feature)   
        self.fusion_layer = Fusion(classfy)
        self.low_level_encoding = nn.Conv2d(512 ,Out_c,(1,3), (1,1), (0,1), bias=False)
         
    #def fuse_forward(self,side_out1,side_out2,side_out3):
    def upsample_path(self,side_out_low):
        side_out_long = nn.functional.interpolate(side_out_low, size=(1, Path_length), mode='bilinear') 

        local_bz,num,_,local_l = side_out_low.size() 
        side_out_low = side_out_low.view(-1,num,local_l).squeeze(1)# squess before fully connected 
        #local_bz,_,num,local_l = side_out_low.size() 
        #side_out_low = side_out_low.view(-1,num,local_l).squeeze(1)# squess before fully connected
        local_bz,num,_,local_l = side_out_long.size() 
        side_out_long = side_out_long.view(-1,num,local_l).squeeze(1)# squess before fully connected 
        return side_out_low,side_out_long
    #    out = self.fusion_layer( side_out1,side_out2,side_out3)
       

    #    return out 
    def forward(self, x):
        if self. UnetBack_flag  == True:
            unet_f = self.Unet_back(x)
            pix_seg = self. pixencoding(unet_f) # use the Unet features to predict a pix wise segmentation
            # pix_seg=unet_f # one feature backbone
            backbone_f = self.backbone(unet_f)

        else:
            backbone_f = self.backbone(x)
            pix_seg = None
        f1 = self.side_branch1 (backbone_f) # coordinates encoding
        f2 = self.side_branch2 (backbone_f) # coordinates encoding
        f3 = self.side_branch3 (backbone_f) # coordinates encoding
        out =  self.fusion_layer(f1,f2,f3)
        side_out1l = self. low_level_encoding(f1)
        side_out2l = self. low_level_encoding(f2)
        side_out3l = self. low_level_encoding(f3)
        side_out1l,side_out1H = self.upsample_path(side_out1l)
        side_out2l,side_out2H = self.upsample_path(side_out2l)
        side_out3l,side_out3H = self.upsample_path(side_out3l)

        return out,side_out1l ,side_out2l,side_out3l,pix_seg
        
        #return out,side_out,side_out2
# mainly based on the resnet  
         
        #return out,side_out,side_out2
# mainly based on the resnet  


