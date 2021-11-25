import numpy as np
import torch
import torch.nn as nn
import  arg_parse
from arg_parse import kernels, strides, pads
from  dataset_layers import Path_length,Batch_size,Resample_size
import torchvision.models
nz = int( arg_parse.opt.nz)
ngf = int( arg_parse.opt.ngf)
ndf = int( arg_parse.opt.ndf)
nc = 1

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 16, kernels[0], strides[0], pads[0], bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernels[1], strides[1], pads[1], bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernels[2], strides[2], pads[2], bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 4,     ngf*2, kernels[3], strides[3], pads[3], bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2,     ngf, kernels[4], strides[4], pads[4], bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(    ngf,      nc, kernels[5], strides[5], pads[5], bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
           
        )

    def forward(self, input):
        output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
         # input is (nc) x 128 x 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, ndf, kernels[5], strides[5], pads[5], bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # input is (nc) x 64 x 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernels[4], strides[4], pads[4], bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # state size. (ndf) x 32 x 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf * 4, kernels[3], strides[3], pads[3], bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # state size. (ndf*2) x 16 x 16
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernels[2], strides[2], pads[2], bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # state size. (ndf*4) x 8 x 8
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, kernels[1], strides[1], pads[1], bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # state size. (ndf*8) x 4 x 4
        self.conv6 = nn.Sequential(
            nn.Conv2d(ndf * 16, Path_length, kernels[0], strides[0], pads[0], bias=False),
            nn.Sigmoid()
        )
        #self.conv6 = nn.Sequential(
        #    nn.Conv2d(ndf * 16, 1, kernels[0], strides[0], pads[0], bias=False),
        #    nn.Sigmoid()
        #)


    def forward(self, input):
        # output = self.main(input)

        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        #return out_conv6.view(-1, 1).squeeze(1)
        return out_conv6 

    def get_features(self, input):
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)

        max_pool1 = nn.MaxPool2d(int(out_conv1.size(2) / 4))
        max_pool2 = nn.MaxPool2d(int(out_conv2.size(2) / 4))
        max_pool3 = nn.MaxPool2d(int(out_conv3.size(2) / 4))
        # max_pool4 = nn.MaxPool2d(int(out_conv4.size(2) / 4))

        vector1 = max_pool1(out_conv1).view(input.size(0), -1).squeeze(1)
        vector2 = max_pool2(out_conv2).view(input.size(0), -1).squeeze(1)
        vector3 = max_pool3(out_conv3).view(input.size(0), -1).squeeze(1)
        # vector4 = max_pool4(out_conv4).view(input.size(0), -1).squeeze(1)

        return torch.cat((vector1, vector2, vector3), 1)
 
def conv_keep_W(indepth,outdepth,k=(4,5),s=(2,1),p=(1,2)):
#output width=((W-F+2*P )/S)+1
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    module = nn.Sequential(
             nn.Conv2d(indepth, outdepth,k, s, p, bias=False),          
             nn.BatchNorm2d(outdepth),
             nn.LeakyReLU(0.1,inplace=True)
             )
    return module
 
def upconv_keep_W(indepth,outdepth,k=(4,5),s=(2,1),p=(1,2)):
#output width=((W-F+2*P )/S)+1
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    module = nn.Sequential(
             nn.ConvTranspose2d(indepth, outdepth,k, s, p, bias=False),          
             nn.BatchNorm2d(outdepth),
             nn.LeakyReLU(0.1,inplace=True)
             )
    return module
def upconv_keep_W_final(indepth,outdepth,k=(4,5),s=(2,1),p=(1,2)):
#output width=((W-F+2*P )/S)+1
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    module = nn.Sequential(
             nn.ConvTranspose2d(indepth, outdepth,k, s, p, bias=False),          
             #nn.BatchNorm2d(outdepth),
             nn.Tanh()
             )
    return module
def conv_keep_all(indepth,outdepth,k=(3,3),s=(1,1),p=(1,1)):
#output width=((W-F+2*P )/S)+1
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    module = nn.Sequential(
             nn.Conv2d(indepth, outdepth,k, s, p, bias=False),          
             nn.BatchNorm2d(outdepth),
             nn.LeakyReLU(0.1,inplace=True)
             )
    return module
def conv_dv_2(indepth,outdepth,k=(6,6),s=(2,2),p=(2,2)):
#output width=((W-F+2*P )/S)+1
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    module = nn.Sequential(
             nn.Conv2d(indepth, outdepth,k, s, p, bias=False),          
             nn.BatchNorm2d(outdepth),
             nn.LeakyReLU(0.1,inplace=True)
             )
    return module
 
class _generator__(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self):
        super(_generator__, self).__init__()
         
        self.layer_num =4
        feature=np.power(2,7) # 2^7
         
        self.side_branch1  =  nn.ModuleList()    
        self.side_branch1.append( nn.Sequential(
              
             nn.Conv2d(self.layer_num, feature,(1,1), (1,1), (0,0), bias=False)         
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
                                                    )
                                 )
        #self.side_branch1.append(  conv_keep_W(feature,2*feature,k=(4,1),s=(1,1),p=(0,0))) # this is for 300
        #4*300
        self.side_branch1.append(  upconv_keep_W(feature, int(feature/2),k=(1,1),s=(1,1),p=(0,0)))
        feature = int(feature/2)

        #  4*300  9*300  -

        self.side_branch1.append(  upconv_keep_W(feature,int(feature/2)))
        feature = int(feature/2)
        #  9*300  18*300  -
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  upconv_keep_W(feature,int(feature/2)))
        feature = int(feature/2)
        #  18*300   37*300  -
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  upconv_keep_W(feature, int(feature/2)))
        feature = int(feature/2)
        # 75*300  - 37*300
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  upconv_keep_W(feature,int(feature/2)))
        feature = int(feature/2)
        #  75*300  150*300 -
        #self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  upconv_keep_W(feature,int(feature/2)))
        feature = int(feature/2)
        #a side branch predict with original iamge with rectangular kernel
        # - 150*300 300*300 
        #self.side_branch1.append(  conv_keep_all(feature, feature))    
        self.side_branch1.append(  upconv_keep_W_final(feature,1))
    
        #self.branch1LU = nn.Tanh( )
         
    def forward(self, x):
        #output = self.main(input)
        #layer_len = len(kernels)
        #for layer_point in range(layer_len):
        #    if(layer_len==0):
        #        output = self.layers[layer_point](input)
        #    else:
        #        output = self.layers[layer_point](output)
        #for i, name in enumerate(self.layers):
        #    x = self.layers[i](x)
        bz,D,W = x.size() 
        side_out = torch.zeros([bz,D,1,W], dtype=torch.float)
        side_out=side_out.cuda()
        side_out[:,:,0,:] =x
        for j, name in enumerate(self.side_branch1):
            side_out = self.side_branch1[j](side_out)
             
        

        #side_out2 =x
        #for j, name in enumerate(self.side_branch2):
        #    side_out2 = self.side_branch2[j](side_out2)
             

        ##fusion
        #fuse1=self.branch1LU(side_out)
        #side_out2 = nn.functional.interpolate(side_out2, size=(1, Path_length), mode='bilinear') 

        #fuse2=self.branch2LU(side_out2)

        #fuse=torch.cat((fuse1,fuse2),1)
        #fuse=self.fusion_layer(fuse)
        ##local_bz,_,_,local_l = fuse.size() 

        #side_out = side_out.view(-1,self.layer_num,Path_length).squeeze(1)# squess before fully connected
        #side_out2 = side_out2.view(-1,self.layer_num,Path_length).squeeze(1)# squess before fully connected

        #out  = fuse.view(-1,self.layer_num,Path_length).squeeze(1)# squess before fully connected
        #final =out
        ## change the predcition as delta of layers
        #for k in range(1,self.layer_num):
        #    final[:,k,:]=final[:,k,:] +  final[:,k-1,:]

        return side_out#,side_out,side_out2

 