import test_model .layer_body_sheath_res2  as baseM 
# the NET dbody for the sheath contour tain  upadat 5th octo 2020
import torch
import torch.nn as nn
import  arg_parse
from arg_parse import kernels, strides, pads
from  dataset_sheath import Path_length,Batch_size,Resample_size
import torchvision.models

class _2LayerScale1(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self):
        super(_2LayerScale1, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
   
        feature = 18

         
        self.layer_num =2
        #a side branch predict with original iamge with rectangular kernel
        # 256*256 - 128*256
        #limit=1024
        self.side_branch1  =  nn.ModuleList()    
        self.side_branch1.append(  baseM.conv_keep_W(3,feature))
        # 128*256 - 64*256
      

        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))
        feature = feature *2
        # 64*256  - 32*256
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))
        feature = feature *2
        # 32*256  - 16*256
         
        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))
        feature = feature *2
        # 16*256  - 8*256
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        #self.side_branch1.append(  conv_keep_all(feature, feature))

        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))
        feature = feature *2
        # 8*256  - 4*256
       
        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature)) # 4*256
        feature = feature *2
      

        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature,k=(3,1),s=(1,1),p=(0,0)))  # 2*256
         
        feature = feature *2
        #self.fullout = nn.Sequential(
              
        #     nn.Conv2d(feature, 256,(1,1), (1,1), (0,0), bias=False)   #2*256       
        #     #nn.BatchNorm2d(1),
        #     #nn.LeakyReLU(0.1,inplace=True)
        #                                            )
        self. low_scale_out = nn.Sequential(
              
             nn.Conv2d(feature, 1 ,(1,1), (1,1), (0,0), bias=False) #2*64           
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
               
                                 )
    def forward(self, x):

        side_out =x
        for j, name in enumerate(self.side_branch1):
            side_out = self.side_branch1[j](side_out)

        #side_out_full = self.fullout(side_out)
        side_out_low = self.low_scale_out (side_out)
        # remove one dimention
        side_out_long = nn.functional.interpolate(side_out_low, size=(2, Path_length), mode='bilinear') 

        local_bz,_,num,local_l = side_out_low.size() 
        side_out_low = side_out_low.view(-1,num,local_l).squeeze(1)# squess before fully connected 
        #local_bz,_,num,local_l = side_out_low.size() 
        #side_out_low = side_out_low.view(-1,num,local_l).squeeze(1)# squess before fully connected
        local_bz,_,num,local_l = side_out_long.size() 
        side_out_long = side_out_long.view(-1,num,local_l).squeeze(1)# squess before fully connected 
        #local_bz,_,num,local_l = side_out_full.size() 
        #side_out_full = side_out_full.view(-1,num,local_l).squeeze(1)# squess before fully connected
  
        return side_out, side_out_low,side_out_long


class _2LayerScale2(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self):
        super(_2LayerScale2, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
   
        feature = 18

         
        self.layer_num =2
        #a side branch predict with original iamge with rectangular kernel
        #limit=1024
        self.side_branch1  =  nn.ModuleList()    
        
        self.side_branch1.append(  baseM.conv_keep_W(3,feature))# 256*256 - 128*256

        
      

        self.side_branch1.append(  baseM.conv_dv_2(feature,2*feature))# 128*256 - 64*128
        feature = feature *2
        
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))# 64*128  - 32*128
        feature = feature *2
        
         
        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))# 32*128  - 16*128
        feature = feature *2
        
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        #self.side_branch1.append(  conv_keep_all(feature, feature))

        self.side_branch1.append(  baseM.conv_dv_2(feature,2*feature))# 16*128  - 8*64
        feature = feature *2
        
       
        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))# 8*64  - 4*64
        feature = feature *2
      

        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature,k=(3,1),s=(1,1),p=(0,0))) #2*64
        feature = feature *2

        #self.fullout = nn.Sequential(
              
        #     nn.ConvTranspose2d(feature, 256,(1,4), (1,4), (0,0), bias=False)   #2*256       
        #     #nn.BatchNorm2d(1),
        #     #nn.LeakyReLU(0.1,inplace=True)
        #                                            )
        self. low_scale_out = nn.Sequential(
              
             nn.Conv2d(feature, 1 ,(1,1), (1,1), (0,0), bias=False) #2*64           
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
               
                                 )   
    def forward(self, x):

        side_out =x
        for j, name in enumerate(self.side_branch1):
            side_out = self.side_branch1[j](side_out)

        #side_out_full = self.fullout(side_out)
        side_out_low = self.low_scale_out (side_out)
        # remove one dimention
        # remove one dimention
        side_out_long = nn.functional.interpolate(side_out_low, size=(2, Path_length), mode='bilinear') 

        local_bz,_,num,local_l = side_out_low.size() 
        side_out_low = side_out_low.view(-1,num,local_l).squeeze(1)# squess before fully connected 
        #local_bz,_,num,local_l = side_out_low.size() 
        #side_out_low = side_out_low.view(-1,num,local_l).squeeze(1)# squess before fully connected
        local_bz,_,num,local_l = side_out_long.size() 
        side_out_long = side_out_long.view(-1,num,local_l).squeeze(1)# squess before fully connected 
        #local_bz,_,num,local_l = side_out_full.size() 
        #side_out_full = side_out_full.view(-1,num,local_l).squeeze(1)# squess before fully connected
  
        return side_out, side_out_low,side_out_long
              
        #return out,side_out,side_out2
# mainly based on the resnet  


class _2LayerScale3(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self):
        super(_2LayerScale3, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
   
        feature = 18

         
        self.layer_num =2
        #a side branch predict with original iamge with rectangular kernel
        #limit=1024
        self.side_branch1  =  nn.ModuleList()    
        
        self.side_branch1.append(  baseM.conv_dv_2(3,feature))# 256*256 - 128*128

        
      

        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))# 128*128 - 64*128
        feature = feature *2
        
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  baseM.conv_dv_2(feature,2*feature))# 64*128  - 32*64
        feature = feature *2
        
         
        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))# 32*64  - 16*64
        feature = feature *2
        
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        #self.side_branch1.append(  conv_keep_all(feature, feature))

        self.side_branch1.append(  baseM.conv_dv_2(feature,2*feature))# 16*64  - 8*32
        feature = feature *2
        
       
        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature))# 8*32  - 4*32
        feature = feature *2
      

        self.side_branch1.append(  baseM.conv_keep_W(feature,2*feature,k=(3,1),s=(1,1),p=(0,0))) #2*32
        feature = feature *2


        #self.fullout = nn.Sequential(
              
        #     nn.ConvTranspose2d(feature, 256,(1,8), (1,8), (0,0), bias=False)   #2*256
        #     #nn.ConvTranspose2d(feature, self.layer_num,(1,4), (1,4), (0,0), bias=False)    
        #     #nn.BatchNorm2d(1),
        #     #nn.LeakyReLU(0.1,inplace=True)
        #                                            )
        self. low_scale_out = nn.Sequential(
              
             nn.Conv2d(feature, 1,(1,1), (1,1), (0,0), bias=False)  #2*32      
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
               
                                 )   
    def forward(self, x):

        side_out =x
        for j, name in enumerate(self.side_branch1):
            side_out = self.side_branch1[j](side_out)

        #side_out_full = self.fullout(side_out)
        side_out_low = self.low_scale_out (side_out)

        # remove one dimention
        side_out_long = nn.functional.interpolate(side_out_low, size=(2, Path_length), mode='bilinear') 

        local_bz,_,num,local_l = side_out_low.size() 
        side_out_low = side_out_low.view(-1,num,local_l).squeeze(1)# squess before fully connected 
        #local_bz,_,num,local_l = side_out_low.size() 
        #side_out_low = side_out_low.view(-1,num,local_l).squeeze(1)# squess before fully connected
        local_bz,_,num,local_l = side_out_long.size() 
        side_out_long = side_out_long.view(-1,num,local_l).squeeze(1)# squess before fully connected 
        #local_bz,_,num,local_l = side_out_full.size() 
        #side_out_full = side_out_full.view(-1,num,local_l).squeeze(1)# squess before fully connected
  
        return side_out , side_out_low,side_out_long
class Fusion(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self):
        super(Fusion, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
        self. up2 = nn.ConvTranspose2d(1152, 576,(1,4), (1,4), (0,0), bias=False)   
        self. up3 =   nn.ConvTranspose2d(1152, 576,(1,8), (1,8), (0,0), bias=False)  
        self. fusion = nn.Conv2d(1152*2,1,(1,3), (1,1), (0,1), bias=False)    # from 3 dpth branch to one   
        self.tan_activation = nn.Tanh()
        
    def forward(self,side_out1,side_out2,side_out3):
        side_out2 = self. up2(side_out2)
        side_out3 = self. up3(side_out3)


        fuse=torch.cat((side_out1,side_out2,side_out3),1)
        fuse=self.fusion(fuse)
    

        local_bz,_,num,local_l = fuse.size() 

        out  = fuse.view(-1,num,local_l).squeeze(1)# squess before fully connected
        return out
       
 
# mainly based on the resnet  
class _2layerFusionNets_(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self):
        super(_2layerFusionNets_, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph
   
        self.side_branch1  =  _2LayerScale1()
         
        self.side_branch2  =  _2LayerScale2()  
        self.side_branch3  =  _2LayerScale3()   
        self.fusion_layer = Fusion( )
         
    def fuse_forward(self,side_out1,side_out2,side_out3):
         
        out = self.fusion_layer( side_out1,side_out2,side_out3)
       
       

        return out 
    def forward(self, x):
      
         
        return out,side_out1l ,side_out2l,side_out3l
        
        #return out,side_out,side_out2
# mainly based on the resnet  
         
        #return out,side_out,side_out2
# mainly based on the resnet  


