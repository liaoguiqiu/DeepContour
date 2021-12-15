import torch.nn as nn
import torch.nn.functional as F
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
   def one_loss_exi(self,output1,target1,exist1): # multiply the the xistence to dilute the loss of no back scattering
           b, _, len = output1.size()
           target_scaled = F.interpolate(target1, size=len, mode='area')
           this_loss = self.criterion(output1*exist1, target_scaled*exist1)
           return this_loss
   def multi_loss_contour_exist(self,output,target,outexist):
       
       num = len(output)
       loss = [None]*num
       for i in  range(num):
           loss[i] = self.one_loss_exi(output[i],target,outexist[i])
       return loss       
