import torch.nn as nn
import torch.nn.functional as F
import  torch
class MTL_loss(object):
   def __init__(self,Loss ="L1"):
       if (Loss == "L1"):
            self.criterion = nn.L1Loss()
       else:
            self.criterion = nn.BCELoss()
   def one_loss(self,output1,target1):
       b, _, len = output1.size()
       target_scaled = F.interpolate(target1, size=len, mode='area')
       this_loss = self.criterion(output1, target_scaled)
       return this_loss
   def multi_loss(self,output,target):
       num = len(output)
       loss = [None]*num
       for i in  range(num):
           loss[i] = self.one_loss(output[i],target)
       return loss  
   def one_loss_exi(self,output1,target1,exist1,Reverse_existence): # multiply the the xistence to dilute the loss of no back scattering
           b, _, len = output1.size()
           target_scaled = F.interpolate(target1, size=len, mode='area')
           backgroud_beta = 0.1
           if (Reverse_existence == True):
               this_loss = self.criterion(output1 * (1-exist1+backgroud_beta), target_scaled*(1-exist1+backgroud_beta))
           else:
               this_loss = self.criterion(output1*(exist1+backgroud_beta), target_scaled*(exist1+backgroud_beta))
           return this_loss
   def multi_loss_contour_exist(self,output,target,outexist,Reverse_existence):
       
       num = len(output)
       loss = [None]*num
       for i in  range(num):
           loss[i] = self.one_loss_exi(output[i],target,outexist[i],Reverse_existence)
       return loss
   # PyTorch


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs) # TODO: comment/uncomment here to change sigmoid function
        # inputs=torch.atan(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice