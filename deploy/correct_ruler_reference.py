import cv2
import math
import numpy as np
import os
import random 
from zipfile import ZipFile
import scipy.signal as signal
import pandas as pd
from DeepAutoJson import Auto_json_label
import matplotlib
matplotlib.use('TkAgg')
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt

from dataset_sheath import myDataloader,Batch_size,Resample_size, Path_length
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
                dy = np.clip(vy[j],2,H-2)
            

                img1[int(dy)+1,j,:]=img1[int(dy),j,:]=painter
              

                #img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter


        return img1
class Find_shadow(object):
    def __init__(self):
        self.folder_num = 0
        self.database_root = " "
        #self.save_root  = "D:/PhD/trying/tradition_method/OCT/sheath registration/pairB/with ruler/correct2/"
        self.save_root  = "D:/PhD/trying/tradition_method/OCT/sheath registration/pairC/ruler/2_correct/"

        #self.database_root = "D:/Deep learning/dataset/original/animal_tissue/1/"
        #self.database_root = "D:/Deep learning/dataset/original/IVOCT/1/"
        self.auto_label = Auto_json_label()

        self.f_downsample_factor = 93
        self.all_dir = "D:/PhD/trying/tradition_method/OCT/sheath registration/pairC/ruler/2_/"
        self.image_dir   = self.database_root + "pic/"
        self.json_dir =  self.database_root + "label/" # for this class sthis dir ist save the modified json 
        self.json_save_dir  = self.database_root + "label_generate/"
        self.img_num = 0
        self.last_est =0
    def find_the_peak ( self,c2,gray): # crop the ROI based on the contour
        shift=10
        rate0=0.1
        rate =0.3
        y1  = np.array(c2) # transfer list to array
        y = y1[:,1] 
        y=signal.resample(y,  Resample_size)
        y = gaussian_filter1d(y,5) # smooth the path 

        max_idex = np.argmin(y)
        y = y.astype(int)

         

        return max_idex,y
        


    def deal_pic(self,img,num):
        ref_position= 18  # this is calculated with img 5, so all img will be shift to this posion
        original =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        H_i,W_i  = original.shape
        img  = cv2.resize(img, ( Resample_size,Resample_size) )
        cv2.imshow('real',img ) 

        #img  = cv2.resize(img, ( Resample_size,Resample_size), interpolation=cv2.INTER_AREA)
        cv2.waitKey(10)  

        gray  =   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         



        H,W   = gray.shape
        coordinates1,coordinates2  = self.auto_label.predict_contour(gray,Resample_size, Resample_size,points=300 )
        max_idex,y=self.find_the_peak (coordinates2,gray)
        #cv2.drawContours(img, coordinates1, -1, (0, 255, 0), 3) 
        draw_coordinates_color(img,y,1)
        shift = (ref_position - max_idex)* W_i/Resample_size
        shift = 0.3*shift + 0.7*self.last_est
        self.last_est = shift

        New = np.roll(original, int( shift) , axis = 1)
        # save this to new folder 
        cv2.imwrite(self.save_root  + str(num) +".jpg",New )
        cv2.imshow('seg',img ) 
  
        cv2.waitKey(10)  


        pass
        return y,img
    def deal_folder(self):
        #for i in os.listdir(self.image_dir): # star from the image folder
        #for i in os.listdir(self.all_dir): # star from the image folder
        for a in range(80,894):
    #for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
        # separath  the name of json 
            #a, b = os.path.splitext(i)
            # if it is a img it will have corresponding image 
            #if b == ".jpg" :
                img_path = self.all_dir + str(a)+ ".jpg"
                #jason_path  = self.json_dir + a + ".json"
                img1 = cv2.imread(img_path)
                
                if img1 is None:
                    print ("no_img")
                else:
                    y,img = self.deal_pic(img1,a)
        
        return 0

if __name__ == '__main__':
        cheker  = Find_shadow()
        img_path = cheker.all_dir + "80" + ".jpg"
        img1 = cv2.imread(img_path)

                #jason_path  = self.json_dir + a + ".json"
        cheker.deal_pic(img1,80)
        #img1 = cv2.imread(img_path)
        cheker.deal_folder()




