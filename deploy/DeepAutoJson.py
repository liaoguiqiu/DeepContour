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
from generator_contour import Save_Contour_pkl
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
from model import cGAN_build2 # the mmodel
# the model
import arg_parse
import cv2
import numpy
import rendering
from generator_contour import Generator_Contour,Save_Contour_pkl,Communicate,Generator_Contour_layers,Generator_Contour_sheath
from time import time
import os
from dataset_sheath import myDataloader,Batch_size,Resample_size, Path_length
from deploy import basic_trans

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    left = int(points * r)
    right = int(points*(1 - r))

    d3 = signal.resample(y,  points)
    d3 = signal.medfilt(d3,5)

    y = d3[left:right]
    l= len(y)

    x = np.arange(0, W)
    add_3   = np.append(x[::-1],x,axis=0) # cascade
    add_3   = np.append(add_3,x[::-1],axis=0) # cascade
    d3 = signal.resample(add_3, 3*l)

    x=d3[l:2*l]
    array = np.zeros((l,2))
    array[:,0] = x.astype(int)
    array[:,1] = y.astype(int)
    coordinates = array.tolist()
    return coordinates



class  Auto_json_label(object):
    def __init__(self ):
        #self.image_dir   = "../../OCT/beam_scanning/Data set/pic/NORMAL-BACKSIDE-center/"
        #self.roi_dir =  "../../OCT/beam_scanning/Data set/seg label/NORMAL-BACKSIDE-center/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE-center/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE/"
        # check the cuda device 
        pth_save_dir = "../out/sheathCGAN_coordinates3/"
        # the portion of attated image to 2     sides
        self.attatch_rate  = 0.2 

        jason_tmp_dir  =  "D:/Deep learning/dataset/original/phantom/1/label/0.json"
        # read th jso fie in hte start :
        with open(jason_tmp_dir) as dir:
            self.jason_tmp = JSON.load(dir)
        self.shapeTmp  = self.jason_tmp["shapes"]
        self.coordinates0 = self.jason_tmp["shapes"] [1]["points"] # remember add finding corred label 1!!!
        self.co_len = len (self.coordinates0) 

        #self.database_root = "D:/Deep learning/dataset/original/phantom/2/"
        #self.database_root = "D:/Deep learning/dataset/original/dots/3/"
        self.database_root = "D:/Deep learning/dataset/original/new_catheter_ruler/2/"
        self.database_root = "D:/Deep learning/dataset/original/phantom_2th_march_2021/1/"
        self.database_root = "D:/Deep learning/dataset/original/paper_with_strong_shadow/1/"

         


        #self.database_root = "D:/Deep learning/dataset/original/animal_tissue/1/"
        #self.database_root = "D:/Deep learning/dataset/original/IVOCT/1/"


        self.f_downsample_factor = 8
        self.all_dir = self.database_root + "pic_all/"
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
        gancreator = cGAN_build2.CGAN_creator() # the Cgan for the segmentation 
        self.GANmodel= gancreator.creat_cgan()  #  G and D are created here 
        # for the detection just use the Gnets
        self.GANmodel.netG.load_state_dict(torch.load(pth_save_dir+'cGANG_epoch_2.pth'))
        self.GANmodel.netG.cuda()
    def downsample_folder(self):#this is to down sample the image in one folder
        read_sequence = os.listdir(self.all_dir) # read all file name
        seqence_Len = len(read_sequence)    # get all file number 
          
        for sequence_num in range(0,seqence_Len):
        #for i in os.listdir("E:/estimagine/vs_project/PythonApplication_data_au/pic/"):
            if (sequence_num%self.f_downsample_factor == 0):
                img_path = self.all_dir + str(sequence_num) + ".jpg"
                #jason_path  = self.json_dir + a + ".json"
                img1 = cv2.imread(img_path)
                
                if img1 is None:
                    print ("no_img")
                else:
                    # write this one into foler
                    cv2.imwrite(self.image_dir  + str(sequence_num) +".jpg",img1 )
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
       
    def predict_contour(self,gray,H_s, W_s , attatch_rate=0.5,points = 100):
        #gray  =   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H,W   = gray.shape
        extend = np.append(gray[:,int((1-attatch_rate)*W):W],gray,axis=1) # cascade
        extend = np.append(extend,gray[:,0:int(attatch_rate*W)],axis=1) # cascade

        #inputV =  basic_trans.Basic_oper.transfer_img_to_tensor(img1,Resample_size,Resample_size)
        inputV =  basic_trans.Basic_oper.transfer_img_to_tensor(extend,H_s,W_s)

        self.GANmodel.set_G_input(inputV) 
        self.GANmodel.forward() # predict the path 
        pathes  =  self.GANmodel.out_pathes0 [0].cpu().detach().numpy()
        #pathes = numpy.clip(pathes,0,1)
        #pathes = pathes*H/Resample_size
        #coordinates1 = encode_path_as_coordinates(pathes[0],Resample_size,Resample_size,H,W)
        #coordinates2 = encode_path_as_coordinates(pathes[1],Resample_size,Resample_size,H,W)
        coordinates1 = encode_as_coordinates_padding(pathes[0],H_s,W_s,H,W,
                                                        attatch_rate,points )
        coordinates2 = encode_as_coordinates_padding(pathes[1],H_s,W_s,H,W,
                                                        attatch_rate,points )

     

        return coordinates1,coordinates2

    def check_one_folder (self):
        #for i in os.listdir(self.image_dir): # star from the image folder
        for i in os.listdir(self.image_dir): # star from the image folder

    #for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
        # separath  the name of json 
            a, b = os.path.splitext(i)
            # if it is a img it will have corresponding image 
            if b == ".jpg" :
                img_path = self.image_dir + a + ".jpg"
                #jason_path  = self.json_dir + a + ".json"
                img1 = cv2.imread(img_path)
                
                if img1 is None:
                    print ("no_img")
                else:
                    gray  =   cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    H,W   = gray.shape
                    coordinates1,coordinates2  = self.predict_contour(gray,Resample_size, Resample_size,points=256 )
                    
                    #extend = np.append(gray[:,int((1-self.attatch_rate)*W):W],gray,axis=1) # cascade
                    #extend = np.append(extend,gray[:,0:int(self.attatch_rate*W)],axis=1) # cascade

                    ##inputV =  basic_trans.Basic_oper.transfer_img_to_tensor(img1,Resample_size,Resample_size)
                    #inputV =  basic_trans.Basic_oper.transfer_img_to_tensor(extend,Resample_size,Resample_size)

                    #self.GANmodel.set_G_input(inputV) 
                    #self.GANmodel.forward() # predict the path 
                    #pathes  =  self.GANmodel.out_pathes0 [0].cpu().detach().numpy()
                    ##pathes = numpy.clip(pathes,0,1)
                    ##pathes = pathes*H/Resample_size
                    ##coordinates1 = encode_path_as_coordinates(pathes[0],Resample_size,Resample_size,H,W)
                    ##coordinates2 = encode_path_as_coordinates(pathes[1],Resample_size,Resample_size,H,W)
                    #coordinates1 = encode_as_coordinates_padding (pathes[0],Resample_size,Resample_size,H,W,
                    #                                              self.attatch_rate,points = 100)
                    #coordinates2 = encode_as_coordinates_padding(pathes[1],Resample_size,Resample_size,H,W,
                    #                                             self.attatch_rate,points = 100)
                    # thsi json dir will be used to save  the generated json
                    save_json_dir = self.json_save_dir + a + ".json"
                    #copy the temp json
                    #with open(json_dir) as dir:
                    #        this_json= JSON.load(dir)
                    this_json = self.jason_tmp
                    this_coodinates = self.coordinates0
                    this_shape = self.shapeTmp
                    shape_temp = self.shapeTmp
                    # the start should be choose larger than 50 , here it is 100
                    #sheath_contour  = self.seger.seg_process(img1,100)
                    for iter  in range(2): #  2 contours here : iter is tje contoru index
                        if  shape_temp[iter]["label"]  =="1":
                            this_coodinates = shape_temp[iter]["points"] # remember add finding corred label 1!!!
                            this_coodinates = coordinates1
                            #co_len = len (this_coodinates) 
                            #for iter2 in range (co_len):
                            #    this_px  = this_coodinates[iter2][0] 
                            #    this_coodinates[iter2][1] = sheath_contour[int(this_px)]
                            shape_temp[iter] ["points"] = this_coodinates   #modify the shape temp
                        else : 
                            this_coodinates = shape_temp[iter]["points"] # remember add finding corred label 1!!!
                            this_coodinates = coordinates2
                            #co_len = len (this_coodinates) 
                            #for iter2 in range (co_len):
                            #    this_px  = this_coodinates[iter2][0] 
                            #    this_coodinates[iter2][1] = sheath_contour[int(this_px)]
                            shape_temp[iter] ["points"] = this_coodinates   #modify the shape temp
                            #for iter2 in range (len(this_shape)):
                            #    if  shape_temp[iter]["label"] == this_shape[iter2]["label"]:
                            #        shape_temp[iter] ["points"] = this_shape[iter2] ["points"]

                    # modify the imag name and the height and width next time

                    
                    #for iter in range (self.co_len):
                    #    this_px  = this_coodinates[iter][0] 
                    #    this_coodinates[iter][1] = sheath_contour[int(this_px)]

                    this_json ["shapes"]   = shape_temp
                    this_json ["imageHeight"] = H
                    this_json ["imageWidth"] = W
                    this_json ["imagePath"] = a+ ".jpg"
                    this_json [ "imageData"]  = encodeImageForJson(img1)
                    #shape  = data["shapes"]
                    with open(save_json_dir, "w") as jsonFile:
                        JSON.dump(this_json, jsonFile)
                    print (a )

                    #num_line  = len(shape)
                    #len_list=  num_line
                    #with open(json_dir) as f_dir:
                    #    data = JSON.load(f_dir)
if __name__ == '__main__':
        cheker  = Auto_json_label()
        #cheker.check_one_folder() 
        cheker.downsample_folder()