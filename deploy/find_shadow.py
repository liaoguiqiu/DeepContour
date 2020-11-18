import cv2
import math
import numpy as np
import os
import random 
from zipfile import ZipFile
import scipy.signal as signal
import pandas as pd
from DeepAutoJson import Auto_json_label

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
        self.database_root = "D:/Deep learning/dataset/original/dots/2/"

        #self.database_root = "D:/Deep learning/dataset/original/animal_tissue/1/"
        #self.database_root = "D:/Deep learning/dataset/original/IVOCT/1/"
        self.auto_label = Auto_json_label()

        self.f_downsample_factor = 93
        self.all_dir = self.database_root + "pic_all/"
        self.image_dir   = self.database_root + "pic/"
        self.json_dir =  self.database_root + "label/" # for this class sthis dir ist save the modified json 
        self.json_save_dir  = self.database_root + "label_generate/"
        self.img_num = 0
    def crop_patch ( self,c2,gray): # crop the ROI based on the contour
        y1  = np.array(c2)
        y = y1[:,1]
        y=signal.resample(y,  Resample_size)
        #y = y.astype(int)

        max_idex =np.argmin(y)
        i=0
        find_flag=10
        while (i):
            i+=1
            if (i>=Resample_size):
                i=0

            if(find_flag==10 and y[i]>(Resample_size-10)):
                find_flag=0
                continue
            if(find_flag==0 and y[i]<(Resample_size-10)):
                find_flag==1

                




        draw_coordinates_color(img,y,1)


        pass


    def deal_pic(self,img):

        img  = cv2.resize(img, ( Resample_size,Resample_size) )
        cv2.imshow('real',img ) 

        #img  = cv2.resize(img, ( Resample_size,Resample_size), interpolation=cv2.INTER_AREA)
        cv2.waitKey(10)  

        gray  =   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray,60,255,cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


        cv2.imshow('2',opening ) 
        cv2.waitKey(10)  
        
         
        #imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #ret,thresh = cv2.threshold(imgray,127,255,0)
        #image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
        #cv2.imshow('real',img ) 



        H,W   = gray.shape
        coordinates1,coordinates2  = self.auto_label.predict_contour(gray,Resample_size, Resample_size,points=Resample_size )
        self.crop_patch (coordinates2,gray)
        #cv2.drawContours(img, coordinates1, -1, (0, 255, 0), 3) 
        cv2.imshow('real',img ) 
  
        cv2.waitKey(10)  


        pass
        return 0
    def deal_folder(self):
        #for i in os.listdir(self.image_dir): # star from the image folder
        for i in os.listdir(self.all_dir): # star from the image folder

    #for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
        # separath  the name of json 
            a, b = os.path.splitext(i)
            # if it is a img it will have corresponding image 
            if b == ".jpg" :
                img_path = self.all_dir + a + ".jpg"
                #jason_path  = self.json_dir + a + ".json"
                img1 = cv2.imread(img_path)
                
                if img1 is None:
                    print ("no_img")
                else:
                    coordinates1,coordinates2  = self.auto_label .predict_contour(img1,Resample_size, Resample_size,points=256 )
                    gray  =   cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    H,W   = gray.shape
        
        return 0

if __name__ == '__main__':
        cheker  = Find_shadow()
        img_path = cheker.all_dir + "777" + ".jpg"
        img1 = cv2.imread(img_path)

                #jason_path  = self.json_dir + a + ".json"
        cheker.deal_pic(img1)
        img1 = cv2.imread(img_path)
        #cheker.deal_folder




