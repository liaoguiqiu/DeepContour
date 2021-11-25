# this scrip is written for the dots video

import cv2
import numpy as np
import os
import scipy.signal as signal
from deploy.DeepAutoJson import Auto_json_label
from dataTool.matlab import Save_Signal_matlab
import matplotlib
matplotlib.use('TkAgg')
from scipy.ndimage import gaussian_filter1d

from dataset_sheath import Resample_size


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
        self.database_root = "D:/Deep learning/dataset/original/dots/1/"
        self.save_dir   =  "D:/PhD/trying/tradition_method/saved_processed_polar/"
        #self.database_root = "D:/Deep learning/dataset/original/animal_tissue/1/"
        #self.database_root = "D:/Deep learning/dataset/original/IVOCT/1/"
        self.auto_label = Auto_json_label()

        self.f_downsample_factor = 93
        self.all_dir = self.database_root + "pic_all/"
        self.image_dir   = self.database_root + "pic/"
        self.json_dir =  self.database_root + "label/" # for this class sthis dir ist save the modified json 
        self.json_save_dir  = self.database_root + "label_generate/"
        self.img_num = 0

         # from pix to distance  for the dot videos 1
          #  dot videos 1 the sheath has 19-20 pixes, represent 0.5mm
        self.pix2dis  = 1.00  # from pix to distance  for the dot videos 1
         #  dot videos 1 the sheath has 19-20 pixes, represent 0.5mm
        self.pix2dis  = 0.5/19  # from pix to distance  for the dot videos 1

        self.matlab = Save_Signal_matlab()
    def convert_coordinate ( self,c2 ): # crop the ROI based on the contour
       
        
        y1  = np.array(c2) # transfer list to array
        y = y1[:,1] 
        y=signal.resample(y,  Resample_size)
        y = gaussian_filter1d(y,5) # smooth the path 
        return y

    def find_closet(self, y1,y2):
        dis = y2  - y1
        min_dis = np.min(dis)
         

        return  min_dis
    def crop_patch2signal ( self,c2,gray): # crop the ROI based on the contour
        shift=10
        rate0=0.1
        rate =0.3
        y1  = np.array(c2)
        y = y1[:,1]
        y=signal.resample(y,  Resample_size)

        max_idex =np.argmin(y)
        y = y.astype(int)

        high = Resample_size-y[max_idex] # calculate the max height 
        source_line = gray [int(y[max_idex]+rate0*high) : int(y[max_idex]+rate*high ) , max_idex]
        L  = len (source_line)
        dots = np.sum(source_line)/L

        longPic = np.append(gray,gray,axis=1)
        longPic = np.append(longPic,gray,axis=1)
        add_3   = np.append(y,y,axis=0) # cascade
        add_3   = np.append(add_3,y,axis=0) # cascade
        max_idex = max_idex + Resample_size

        # acculate to the left 
        for i in range(Resample_size):
            ind= max_idex -i-1
            if (add_3[ind] > (Resample_size -40) ):
                break
            high= Resample_size-add_3[ind] # calculate the max height 
            source_line = longPic [int(add_3[ind]+rate0*high) : int(add_3[ind]+rate*high) , ind]
            L  = len (source_line)
            this_dots = np.sum(source_line)/L
            dots=  np.append(this_dots,dots)
         # acculate to the right 
        for i in range(Resample_size):
            ind= max_idex +i+1
            if (add_3[ind] > (Resample_size -40) ):
                break
            high= Resample_size-add_3[ind] # calculate the max height 
            source_line = longPic [int(add_3[ind]+rate0*high) : int(add_3[ind]+rate*high) , ind]
            L  = len (source_line)
            this_dots = np.sum(source_line)/L
            dots=  np.append(dots,this_dots )

        #cv2.waitKey(10)  
        
        #plt.plot(dots)
        #plt.show()
        #cv2.waitKey(10)  

        return dots,y
        #find_flag=10
        #while (i):
        #    i+=1
        #    if (i>=Resample_size):
        #        i=0

        #    if(find_flag==10 and y[i]>(Resample_size-10)):
        #        find_flag=0
        #        continue
        #    if(find_flag==0 and y[i]<(Resample_size-10)):
        #        find_flag==1

                




        #draw_coordinates_color(img,y,1)


        pass


    def deal_pic(self,img):

        img  = cv2.resize(img, ( Resample_size,Resample_size) )
        cv2.imshow('real',img ) 

        #img  = cv2.resize(img, ( Resample_size,Resample_size), interpolation=cv2.INTER_AREA)
        cv2.waitKey(10)  

        gray  =   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #ret,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)
        #kernel = np.ones((3,3),np.uint8)
        #opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        ##closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


        #cv2.imshow('2',opening ) 
        #cv2.waitKey(10)  
        
         
        #imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #ret,thresh = cv2.threshold(imgray,127,255,0)
        #image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
        #cv2.imshow('real',img ) 



        H,W   = gray.shape
        coordinates1,coordinates2  = self.auto_label.predict_contour(gray,Resample_size, Resample_size,points=300 )
        y1 = self.convert_coordinate(coordinates1)
        y2 = self.convert_coordinate(coordinates2)
        min_dis  = self.find_closet(y1,y2)
        #dots,y=self.crop_patch2signal (coordinates2,gray)
        #cv2.drawContours(img, coordinates1, -1, (0, 255, 0), 3) 
        draw_coordinates_color(img,y1.astype(int),0)
        draw_coordinates_color(img,y2.astype(int),1)

        cv2.imshow('seg',img ) 
  
        cv2.waitKey(10)  
  
        return min_dis,y2,img
    def deal_folder(self):
        #for i in os.listdir(self.image_dir): # star from the image folder

        flen  = len(os.listdir(self.all_dir))
        #for i in os.listdir(self.all_dir): # star from the image folder

    #for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
        # separath  the name of json 
        for a in range(0,flen):
        # if it is a img it will have corresponding image 
            
            img_path = self.all_dir + str( a) + ".jpg"
            #jason_path  = self.json_dir + a + ".json"
            img1 = cv2.imread(img_path)
                
            if img1 is None:
                print ("no_img")
            else:
                print ("img" + str( a))

                dis,y,img = self.deal_pic(img1)
                cv2.imwrite(self.save_dir  + str(a) +".jpg",img )
                dis  = self.pix2dis * dis
                self. matlab . buffer1(dis)
                self. matlab .save_mat()
        return 0

if __name__ == '__main__':
        cheker  = Find_shadow()
        img_path = cheker.all_dir + "0" + ".jpg"
        img1 = cv2.imread(img_path)

                #jason_path  = self.json_dir + a + ".json"
        cheker.deal_pic(img1)
        img1 = cv2.imread(img_path)
        cheker.deal_folder()




