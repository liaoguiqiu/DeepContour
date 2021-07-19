
#used python packages
import cv2
import math
import numpy as np
import os
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
 
import arg_parse
#import imagenet
from analy import MY_ANALYSIS
from analy import Save_signal_enum
import cv2
import numpy
from image_trans import BaseTransform  
from generator_contour import Generator_Contour,Save_Contour_pkl
import matplotlib.pyplot as plt
from scipy import signal 
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Basic_oper(object):
    def transfer_img_to_tensor(color,outH,outW,depth=3):
        # this function trasfer a image to a tensro (normaly )
        #this_gray  =   cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        this_gray= color
        img_piece = cv2.resize(this_gray, (outW,outH), interpolation=cv2.INTER_AREA)
            
        #img_piece = cv2.medianBlur(img_piece,5)
        if depth == 3:
        #mydata_loader .read_a_batch()
        #change to 3 chanels
            np_input = numpy.zeros((1,3,outH,outW)) # a batch with piece num
            np_input[0,0,:,:] = (img_piece - 104.0)/104.0
            np_input[0,1,:,:] = (img_piece - 104.0)/104.0
            np_input[0,2,:,:] = (img_piece - 104.0)/104.0
        
  

        input = torch.from_numpy(numpy.float32(np_input)) 
        #input = input.to(device) 
        #input = torch.from_numpy(numpy.float32(mydata_loader.input_image[0,:,:,:])) 
        input = input.to(device) 
        return input
         

    def tranfer_frome_cir2rec(gray):
        H,W = gray.shape
        value = np.sqrt(((H/2.0)**2.0)+((W/2.0)**2.0))

        polar_image = cv2.linearPolar(gray,(W/2, H/2), value, cv2.WARP_FILL_OUTLIERS)

        polar_image = polar_image.astype(np.uint8)
        polar_image=cv2.rotate(polar_image,rotateCode = 0) 
        return polar_image
    def tranfer_frome_rec2cir(gray):
        H,W = gray.shape
        value = np.sqrt(((H/2.0)**2.0)+((W/2.0)**2.0))
        gray=cv2.rotate(gray,rotateCode = 2) 
        #value = 200
        #circular = cv2.linearPolar(new_frame, (new_frame.shape[1]/2 , new_frame.shape[0]/2), 
        #                               200, cv2.WARP_INVERSE_MAP)
        circular = cv2.linearPolar(gray,(W/2, H/2), value, cv2.WARP_INVERSE_MAP)

        circular = circular.astype(np.uint8)
        #polar_image=cv2.rotate(polar_image,rotateCode = 0) 
        return circular
    def tranfer_frome_rec2cir2(color, padding_H =58):
        H,W_ini,_ = color.shape
        padding = np.zeros((padding_H,W_ini,3))
         
        color  = np.append(padding,color,axis=0)
        H,W,_ = color.shape
        value = np.sqrt(((H/4.2)**2.0)+((W/4.2)**2.0))
        color=cv2.rotate(color,rotateCode = 2) 
        #value = 200
        #circular = cv2.linearPolar(new_frame, (new_frame.shape[1]/2 , new_frame.shape[0]/2), 
        #                               200, cv2.WARP_INVERSE_MAP)
        circular = cv2.linearPolar(color,(W/2, H/2), value, cv2.WARP_INVERSE_MAP)

        circular = circular.astype(np.uint8)
        #polar_image=cv2.rotate(polar_image,rotateCode = 0) 
        return circular


