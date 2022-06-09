
# update 4:38 2nd OCt 2020
# this is to modify/opy a exiting Json file to generate the contour of the theatht 
#!!! this auto json is to generate json for images without label file , this willl generate a lot of json file 
import json as JSON
import cv2
import math
import numpy as np
import os
import random 
from zipfile import ZipFile
import scipy.signal as signal
import pandas as pd
from dataTool.generator_contour import Save_Contour_pkl
#from seg_one_1 import Seg_One_1
import codecs
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import base64
import io

import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
#from model import cGAN_build2 # the mmodel
from model import CE_build3 # the mmodel
# the model
import arg_parse
import cv2
import numpy
import rendering
from dataTool.generator_contour import Generator_Contour,Save_Contour_pkl,Communicate
from time import time
import os
from dataset_ivus import myDataloader,Batch_size,Resample_size, Path_length
from deploy import basic_trans
from scipy.interpolate import interp1d
from working_dir_root import Dataset_root,Output_root
from deploy.basic_trans import Basic_oper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def resample(x,n,kind='nearest'):
            factor = float(x.size/n)
            f = interp1d(np.linspace(0,1,x.size),x,kind)
            return f(np.linspace(0,1,n))
def draw_coordinates_color(img1,vy,color):
        if color ==0:
           painter  = [254,0,0]
        elif color ==1:
           painter  = [0,254,0]
        elif color ==2:
           painter  = [0,0,254]
        else :
           painter  = [0,0,0]
                    #path0  = signal.resample(path0, W)
        H,W,_ = img1.shape
        for j in range (W):
                #path0l[path0x[j]]
                dy = numpy.clip(vy[j],2,H-2)
            

                img1[int(dy)+1,j,:]=img1[int(dy),j,:]=painter
                img1[int(dy)-1,j,:]=img1[int(dy)-2,j,:]=painter

                #img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter


        return img1
def encodeImageForJson(image):
    img_pil = PIL.Image.fromarray(image, mode='RGB')
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    return encData
def encode_path_as_coordinates(path,h,w,H,W):
    points = 150
    y = path*H
    add_3   = np.append(y[::-1],y,axis=0) # cascade
    add_3   = np.append(add_3,y[::-1],axis=0) # cascade
    d3 = signal.resample(add_3, 3*points)
    d3 = signal.medfilt(d3,5)

    y = d3[points:2*points]

    x = np.arange(0, W)
    add_3   = np.append(x[::-1],x,axis=0) # cascade
    add_3   = np.append(add_3,x[::-1],axis=0) # cascade
    d3 = signal.resample(add_3, 3*points)

    x=d3[points:2*points]
    array = np.zeros((points,2))
    array[:,0] = x.astype(int)
    array[:,1] = y.astype(int)
    coordinates = array.tolist()
    return coordinates
def encode_as_coordinates_padding(path,h,w,H,W,rate,points = 150):


    r = rate/(2*rate+1) # the rate is calculated from original padding rate

    y = path*H
    #add_3   = np.append(y[::-1],y,axis=0) # cascade
    #add_3   = np.append(add_3,y[::-1],axis=0) # cascade
    left = int(W * r)
    right = int(W*(1 - r))

    d3 = resample(y,  W, kind='linear')
    #d3 = signal.medfilt(d3,5)

    y = resample(d3,points)
    l = len(y)

    #x0 = np.arange(0, W+1, int((W+1) / l))
    x0 = np.arange(0, W)

    x=  resample(x0,l)
    #x = x0[0:l]
    x[l-1] = W-1


    # add_3   = np.append(x[::-1],x,axis=0) # cascade
    # add_3   = np.append(add_3,x[::-1],axis=0) # cascade
    # d3 = signal.resample(add_3, 3*l)

    # x=d3[l:2*l]
    array = np.zeros((l,2))
    array[:,0] = x.astype(int)
    array[:,1] = y.astype(int)
    coordinates = array.tolist()
    return coordinates
def encode_as_original_y_padding_exv(path,exv,h,w,H,W,rate,points = 150):


    r = rate/(2*rate+1) # the rate is calculated from original padding rate
    mask = exv <0.5
    y = path*H* mask + (1-mask)*H
    #add_3   = np.append(y[::-1],y,axis=0) # cascade
    #add_3   = np.append(add_3,y[::-1],axis=0) # cascade
    left = int(W * r)
    right = int(W*(1 - r))

    d3 = resample(y,  W, kind='linear')
    #d3 = signal.medfilt(d3,5)
    #add cutt
    #y = resample(d3,points)
    y = resample(d3[left:right],W) # cut from the left to the right
     
    return y


class  Auto_json_label(object):
    def __init__(self ):
        #self.image_dir   = "../../OCT/beam_scanning/Data set/pic/NORMAL-BACKSIDE-center/"
        #self.roi_dir =  "../../OCT/beam_scanning/Data set/seg label/NORMAL-BACKSIDE-center/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE-center/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE/"
        # check the cuda device 
        pth_save_dir = "../../../out/sheathCGAN_coordinates3/"
        pth_save_dir = Output_root + "CEnet_trained/"

        
        # the portion of attated image to 2     sides
        self.attatch_rate  = 0.1

        #jason_tmp_dir  =  "D:/Deep learning/dataset/original/animal_tissue/1/label/100.json"
        jason_tmp_dir  = Dataset_root+ "label data/config/example.json"

        # read th jso fie in hte start :
        with open(jason_tmp_dir) as dir:
            self.jason_tmp = JSON.load(dir)
        self.shapeTmp  = self.jason_tmp["shapes"]
        self.coordinates0 = self.jason_tmp["shapes"] [1]["points"] # remember add finding corred label 1!!!
        self.co_len = len (self.coordinates0) 

        #self.database_root = "D:/Deep learning/dataset/original/phantom/2/"
        #self.database_root = "D:/Deep learning/dataset/original/dots/3/"
        # self.database_root = "D:/Deep learning/dataset/original/new_catheter_ruler/2/"
        # self.database_root = "D:/Deep learning/dataset/original/phantom_2th_march_2021/1/"
        # self.database_root = "D:/Deep learning/dataset/original/paper_with_strong_shadow/1/"
        self.database_root = Dataset_root +   "original/phantom_feed_back/"
        self.database_root = "E:/database/feed back experiment/"
        self.operatedir =   self.database_root + "resize/no fd syn with a video/"

        #self.database_root = "D:/Deep learning/dataset/original/animal_tissue/1/"
        #self.database_root = "D:/Deep learning/dataset/original/IVOCT/1/"
        base_dir =  os.path.basename(os.path.normpath(self.operatedir))
        self.save_correct_orien_dir =  self.database_root + "correct/"  + base_dir + "/"+"correct_orien/"
        self.save_align_surf_dir =   self.database_root + "correct/"  + base_dir + "/" + "align_surf/"
        self.save_correct_orien_cir_dir =  self.database_root + "correct/"  + base_dir + "/"+"correct_orien_cir/"
        self.save_align_surf_cir_dir =   self.database_root + "correct/"  + base_dir + "/" + "align_surf_cir/"
       
        self.refH = 300 # manuall set
        self.refW = 0
        try:
            os.stat(self.save_correct_orien_dir)
        except:
            os.makedirs(self.save_correct_orien_dir)
        try:
            os.stat(self.save_correct_orien_cir_dir)
        except:
            os.makedirs(self.save_correct_orien_cir_dir)
        try:
            os.stat(self.save_align_surf_dir )
        except:
            os.makedirs(self.save_align_surf_dir )
        try:
            os.stat(self.save_align_surf_cir_dir)
        except:
            os.makedirs(self.save_align_surf_cir_dir )

        self.f_downsample_factor = 30
        self.all_dir = self.database_root + "pic_all/"
        #self.image_dir   = self.database_root + "img/"
        self.image_dir   = self.database_root + "pic/"

        self.json_dir =  self.database_root + "label/" # for this class sthis dir ist save the modified json
        self.json_save_dir  = self.database_root + "label_generate/"
        self.img_num = 0
         
        self.contours_x =  [] # no predefines # predefine there are 4 contours
        self.contours_y =  [] # predefine there are 4 contours
        #self.seger = Seg_One_1()
        self.saver = Save_Contour_pkl()
        self.display_flag = True

        # deep learning model
        print(torch.cuda.current_device())
        print(torch.cuda.device(0))
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.is_available())
        torch.set_num_threads(2)

        Model_creator = CE_build3.CE_creator() # the  CEnet trainer with CGAN
#   Use the same arch to create two nets 
        self.CE_Nets= Model_creator.creat_nets()   # one is for the contour cordinates
        
        # for the detection just use the Gnets
        self.CE_Nets.netG.load_state_dict(torch.load(pth_save_dir+'cGANG_epoch_1.pth'))
        self.CE_Nets.netG.cuda()
       
        self.CE_Nets.netG.Unet_back.eval()
        self.croptop = 0
        #self.CE_Nets.netE.load_state_dict(torch.load(pth_save_dir+'cGANG_epoch_5.pth'))
        #self.CE_Nets.netE.cuda()

    def downsample_folder(self):#this is to down sample the image in one folder
        read_sequence = os.listdir(self.all_dir) # read all file name
        seqence_Len = len(read_sequence)    # get all file number 
          
        for sequence_num in range(0,seqence_Len):
        #for i in os.listdir("E:/estimagine/vs_project/PythonApplication_data_au/pic/"):
            if (sequence_num%self.f_downsample_factor == 0):
                img_path = self.all_dir + str(sequence_num) + ".tif"
                #jason_path  = self.json_dir + a + ".json"
                img1 = cv2.imread(img_path)
                
                if img1 is None:
                    print ("no_img")
                else:
                    # write this one into foler
                    cv2.imwrite(self.image_dir  + str(sequence_num) +".tif",img1 )
                    print("write  " + str(sequence_num))
                    pass

            sequence_num+=1
            pass

    def draw_coordinates_color(self,img1,vx,vy,color):
        
        if color ==0:
           painter  = [254,0,0]
        elif color ==1:
           painter  = [0,254,0]
        elif color ==2:
           painter  = [0,0,254]
        else :
           painter  = [0,0,0]
                    #path0  = signal.resample(path0, W)
        H,W,_ = img1.shape
        for j in range (len(vx)):
                #path0l[path0x[j]]
                dy = np.clip(vy[j],2,H-2)
                dx = np.clip(vx[j],2,W-2)

                img1[int(dy)+1,int(dx),:]=img1[int(dy),int(dx),:]=painter
                #img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter


        return img1
    def cut_path_edge(self,pathes):
        num,l = pathes.shape
        #transfer from the attatch portion of initial image 
        rate = self.attatch_rate/(2*self.attatch_rate+1)
        # resize first 
        L = int((2*self.attatch_rate+1)*l)
        new_p = np.zeros((num,L))

        for i in range(num):
            new_p[i]= signal.resample(pathes[i], L)
        # cut
            





        new_p = pathes[:,int(rate*l):int((1-rate)*l)]
       
    def predict_tissue_contour(self,gray,H_s, W_s , attatch_rate=0.1,points = 64):
        #gray  =   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H,W   = gray.shape
        gray = gray[self.croptop:H,:]
        extend = np.append(gray[:,int((1-attatch_rate)*W):W],gray,axis=1) # cascade
        extend = np.append(extend,gray[:,0:int(attatch_rate*W)],axis=1) # cascade
        img_piece = cv2.cvtColor(extend, cv2.COLOR_GRAY2RGB)

        img_piece = cv2.resize(img_piece, (W_s,H_s), interpolation=cv2.INTER_AREA)
        #inputV =  basic_trans.Basic_oper.transfer_img_to_tensor(img1,Resample_size,Resample_size)
        inputV =  basic_trans.Basic_oper.transfer_img_to_tensor(extend,H_s,W_s)

        self.CE_Nets.set_GE_input(inputV) 
        self.CE_Nets.forward(validation_flag = True, one_hot_render = False) # predict the path 
        pathes  =  self.CE_Nets.out_pathes0 [0].cpu().detach().numpy()
        pathes = pathes + self.croptop/H # shift back
        existences = self.CE_Nets.out_exis_v0 [0].cpu().detach().numpy() #self.out_exis_v0
        mask = existences<0.5
        draw_path0= pathes[0]*H_s * mask[0] + (1-mask[0])* H_s
        draw_path1= pathes[1]*H_s * mask[1] + (1-mask[1])* H_s

        img_draw =  draw_coordinates_color(img_piece.astype(np.uint8),draw_path0,0) 
        img_draw =  draw_coordinates_color(img_draw,draw_path1,1) 
        cv2.imshow('predicit_auto json',img_draw.astype(np.uint8)) 
        cv2.waitKey(1)
        
        
        #pathes = numpy.clip(pathes,0,1)
        #pathes = pathes*H/Resample_size
        #coordinates1 = encode_path_as_coordinates(pathes[0],Resample_size,Resample_size,H,W)
        #coordinates2 = encode_path_as_coordinates(pathes[1],Resample_size,Resample_size,H,W)
        #coordinates1 = encode_as_coordinates_padding(pathes[0],H_s,W_s,H,W,
        #                                                attatch_rate,points )

        #coordinates2 = encode_as_coordinates_padding(pathes[1],H_s,W_s,H,W,
        #                                                attatch_rate,points )
        #coordinates1 = encode_as_coordinates_padding_exv(pathes[0],existences[0],H_s,W_s,H,W,
                                                        #attatch_rate,points )

        Y = encode_as_original_y_padding_exv(pathes[1],existences[1],H_s,W_s,H,W,
                                                        attatch_rate,points )

     #encode_as_coordinates_padding_exv

        return Y

    def check_one_folder (self):
        imagelist = os.listdir(self.operatedir)
        _,image_type  = os.path.splitext(imagelist[0]) # first image of this folder 
        #for i in os.listdir(self.image_dir): # star from the image folder
        #for i in os.listdir(self.operatedir): # star from the image folder
        folder_l = len(os.listdir(self.operatedir))
        for i in  range(folder_l):


    #for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
        # separath  the name of json 
            a=i
            #a, b = os.path.splitext(i)
            flag = True
            # if it is a img it will have corresponding image 
            if flag == True :
                img_path = self.operatedir + str(a) + image_type
                #jason_path  = self.json_dir + a + ".json"
                img1 = cv2.imread(img_path)
                
                if img1 is None:
                    print ("no_img")
                else:
                    gray  =   cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    H,W   = gray.shape
                    
                    # modified to only predict the tissue contour
                    Y = self.predict_tissue_contour(gray,Resample_size, Resample_size,attatch_rate=self.attatch_rate,points=40 )
                    max_idex = np.argmin(Y)
                    #if a==0:
                    #    self.shiftW_l = shiftW
                    #    used_shift_W = shiftW
                    #else:
                    #    used_shift_W = shiftW*0.2 + self.shiftW_l*0.8
                    #    self.shiftW_l = shiftW
                    shiftW = (self.refW - max_idex)
                    Align_orin = np.roll(gray, int( shiftW) , axis = 1)
                    cv2.imwrite(self.save_correct_orien_dir  + str(a) + image_type,Align_orin )

                    shiftH = (self.refH - Y[max_idex])      
                    Align_sur = np.roll(Align_orin, int( shiftH) , axis = 0)
                    cv2.imwrite(self.save_align_surf_dir  + str(a) + image_type,Align_sur )

                   

                    

                    Align_orin_cir = Basic_oper.tranfer_frome_rec2cir2(Align_orin) 
                    cv2.imwrite(self.save_correct_orien_cir_dir  + str(a) + image_type,Align_orin_cir )
                    Align_sur_cir = Basic_oper.tranfer_frome_rec2cir2(Align_sur) 
                    cv2.imwrite(self.save_align_surf_cir_dir  + str(a) + image_type,Align_sur_cir )
                    #extend = np.append(gray[:,int((1-self.attatch_rate)*W):W],gray,axis=1) # cascade
                    #extend = np.append(extend,gray[:,0:int(self.attatch_rate*W)],axis=1) # cascade

                    ##inputV =  basic_trans.Basic_oper.transfer_img_to_tensor(img1,Resample_size,Resample_size)
                    #inputV =  basic_trans.Basic_oper.transfer_img_to_tensor(extend,Resample_size,Resample_size)

                    #self.CE_Nets.set_G_input(inputV) 
                    #self.CE_Nets.forward() # predict the path 
                    #pathes  =  self.CE_Nets.out_pathes0 [0].cpu().detach().numpy()
                    ##pathes = numpy.clip(pathes,0,1)
                    ##pathes = pathes*H/Resample_size
                    ##coordinates1 = encode_path_as_coordinates(pathes[0],Resample_size,Resample_size,H,W)
                    ##coordinates2 = encode_path_as_coordinates(pathes[1],Resample_size,Resample_size,H,W)
                    #coordinates1 = encode_as_coordinates_padding (pathes[0],Resample_size,Resample_size,H,W,
                    #                                              self.attatch_rate,points = 100)
                    #coordinates2 = encode_as_coordinates_padding(pathes[1],Resample_size,Resample_size,H,W,
                    #                                             self.attatch_rate,points = 100)
                    # thsi json dir will be used to save  the generated json
                     
                    # the start should be choose larger than 50 , here it is 100
                    #sheath_contour  = self.seger.seg_process(img1,100)
                    
                    print (a )

                    #num_line  = len(shape)
                    #len_list=  num_line
                    #with open(json_dir) as f_dir:
                    #    data = JSON.load(f_dir)
if __name__ == '__main__':
        labeler  = Auto_json_label()
        labeler.check_one_folder() 
        #labeler.downsample_folder()