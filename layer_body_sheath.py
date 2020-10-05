# the NET dbody for the sheath contour tain  upadat 5th octo 2020
import torch
import torch.nn as nn
import  arg_parse
from arg_parse import kernels, strides, pads
from  dataset_sheath import Path_length,Batch_size,Resample_size
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

class _netD_8_multiscal_fusion300(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self):
        super(_netD_8_multiscal_fusion300, self).__init__()
        kernels = [6, 4, 4, 4, 2,2,4]
        strides = [2, 2, 2, 2, 2,2,1]
        pads =    [2, 1, 1, 1, 0,0,0]
        self.fully_connect_len  =1000
        layer_len = len(kernels)

        #a side branch predict with original iamge with rectangular kernel
        # 300*300 - 150*300
        feature = 6
        self.side_branch1  =  nn.ModuleList()    
        self.side_branch1.append(  conv_keep_W(3,feature))
        # 150*300 - 75*300
        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        # 75*300  - 37*300
        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        # 37*300  - 18*300
        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        # 18*300  - 9*300
        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        # 9*300  - 4*300

        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        self.side_branch1.append(  conv_keep_W(feature,2*feature,k=(4,1),s=(1,1),p=(0,0)))
         
        feature = feature *2
        self.side_branch1.append( nn.Sequential(
              
             nn.Conv2d(feature, 1,(1,1), (1,1), (0,0), bias=False)         
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
                                                    )
                                 )

        #create the layer list
        self.layers = nn.ModuleList()
        for layer_pointer in range(layer_len):
             # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
             # input is (nc) x 128 x 128
            if  layer_pointer ==0:
                this_input_depth = 3
                this_output_depth = 8
            else:
                this_input_depth = this_output_depth
                this_output_depth = this_output_depth*2


    
            if (layer_pointer == (layer_len-1)):
                #self.layers = nn.Sequential(
                #nn.Conv2d(this_input_depth, Path_length, kernels[layer_len -layer_pointer-1], strides[layer_len -layer_pointer-1], pads[layer_len -layer_pointer-1], bias=False),
                #nn.Sigmoid()
                # )
                self.layers.append(
                nn.Conv2d(this_input_depth, self.fully_connect_len, kernels[layer_pointer], strides[layer_pointer], pads[layer_pointer], bias=False),          
                 )
                #self.layers.append (
                #nn.BatchNorm2d(1000),
                #   )
                #self.layers.append(
                #nn. AdaptiveAvgPool2d(output_size=(1, 1)),    
                # )
                self.layers.append (
                nn.LeakyReLU(0.2, inplace=False) #1
                )
                self.layers.append(
                nn.Linear(self.fully_connect_len, Path_length, bias=False),   #2       
                 )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
                #self.layers.append (
                #nn.BatchNorm2d(Path_length),
                #   )
                #self.layers.append(
                #nn.Sigmoid()
                # )
            else:
                  # input is (nc) x 64 x 64
                self.layers.append (
                nn.Conv2d(this_input_depth, this_output_depth, kernels[layer_pointer], strides[layer_pointer], pads[layer_pointer], bias=False),
                )
                self.layers.append (
                nn.BatchNorm2d(this_output_depth),
                   )
                self.layers.append (
                nn.LeakyReLU(0.2, inplace=False)
                )
            #self.layers.append(this_layer)
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
        side_out =x
        for j, name in enumerate(self.side_branch1):
            side_out = self.side_branch1[j](side_out)
             
        side_out = side_out.view(-1,Path_length).squeeze(1)# squess before fully connected

        for i, name in enumerate(self.layers):
           x = self.layers[i](x)
           if i == (len(self.layers)-2) :
               x = x.view(-1,self.fully_connect_len).squeeze(1)# squess before fully connected 

         
        out  = 0.6*side_out+ 0.4 *x
        #out  = side_out
        # return x
        # return side_out
        return out,side_out,x
# mainly based on the resnet  
class _netD_8_multiscal_fusion300_layer(nn.Module):
#output width=((W-F+2*P )/S)+1

    def __init__(self):
        super(_netD_8_multiscal_fusion300_layer, self).__init__()
         
        self.layer_num =2
        #a side branch predict with original iamge with rectangular kernel
        # 300*300 - 150*300
        feature = 12
        self.side_branch1  =  nn.ModuleList()    
        self.side_branch1.append(  conv_keep_W(3,feature))
        # 150*300 - 75*300
        self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        # 75*300  - 37*300
        self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        # 37*300  - 18*300
        self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        # 18*300  - 9*300
        self.side_branch1.append(  conv_keep_all(feature, feature))
        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        # 9*300  - 4*300

        self.side_branch1.append(  conv_keep_W(feature,2*feature))
        feature = feature *2
        self.side_branch1.append(  conv_keep_W(feature,2*feature,k=(4,1),s=(1,1),p=(0,0)))
         
        feature = feature *2
        self.side_branch1.append( nn.Sequential(
              
             nn.Conv2d(feature, self.layer_num,(1,1), (1,1), (0,0), bias=False)         
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
                                                    )
                                 )


        #a side branch predict with original iamge with rectangular kernel
        # 300*300 - 150*300
        feature = 12
        self.side_branch2  =  nn.ModuleList()    
        self.side_branch2.append(  conv_keep_W(3,feature))
        # 150*300 - 75*150

        self.side_branch2.append(  conv_dv_2(feature,2*feature))#
        # 75*150  - 37*150
        feature = feature *2
        self.side_branch2.append(  conv_keep_all(feature, feature))

        self.side_branch2.append(  conv_keep_W(feature,2*feature))
        # 37*150  - 18*75
        feature = feature *2
        self.side_branch2.append(  conv_keep_all(feature, feature))

        self.side_branch2.append(  conv_dv_2(feature,2*feature))
        # 18*75  - 9*75
        feature = feature *2
        self.side_branch2.append(  conv_keep_all(feature, feature))

        self.side_branch2.append(  conv_keep_W(feature,2*feature))
        # 9*75  - 4*75
        feature = feature *2
        self.side_branch2.append(  conv_keep_all(feature, feature))

        self.side_branch2.append(  conv_keep_W(feature,2*feature))

        # 4*75  - 1*75
        feature = feature *2
        self.side_branch2.append(  conv_keep_W(feature,2*feature,k=(4,1),s=(1,1),p=(0,0)))
        # 1*75  - 1*300
         
        feature = feature *2
        self.side_branch2.append( nn.Sequential(
              
             nn.Conv2d(feature, self.layer_num,(1,1), (1,1), (0,0), bias=False)         
             #nn.BatchNorm2d(1),
             #nn.LeakyReLU(0.1,inplace=True)
                                                    )
                                 )
        self.branch1LU = nn.LeakyReLU(0.1,inplace=True)
        self.branch2LU = nn.LeakyReLU(0.1,inplace=True)
        self.fusion_layer = nn.Conv2d(2*self.layer_num,self.layer_num,(1,3), (1,1), (0,1), bias=False)       

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
        side_out =x
        for j, name in enumerate(self.side_branch1):
            side_out = self.side_branch1[j](side_out)
             
        

        side_out2 =x
        for j, name in enumerate(self.side_branch2):
            side_out2 = self.side_branch2[j](side_out2)
             

        #fusion
        fuse1=self.branch1LU(side_out)
        side_out2 = nn.functional.interpolate(side_out2, size=(1, Path_length), mode='bilinear') 

        fuse2=self.branch2LU(side_out2)

        fuse=torch.cat((fuse1,fuse2),1)
        fuse=self.fusion_layer(fuse)
        #local_bz,_,_,local_l = fuse.size() 

        side_out = side_out.view(-1,self.layer_num,Path_length).squeeze(1)# squess before fully connected
        side_out2 = side_out2.view(-1,self.layer_num,Path_length).squeeze(1)# squess before fully connected

        out  = fuse.view(-1,self.layer_num,Path_length).squeeze(1)# squess before fully connected
        

        return out,side_out,side_out2

        #return out,side_out,side_out2
# mainly based on the resnet  
