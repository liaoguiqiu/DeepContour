import cv2
import numpy as np
import os
from analy import MY_ANALYSIS
from generator_contour import  Generator_Contour,Save_Contour_pkl
from analy import Save_signal_enum
from scipy import signal 
from image_trans import BaseTransform  
from random import seed
from random import random
import pickle

seed(1)
Batch_size = 1
Resample_size =300
Path_length = 300
Augment_limitation_flag = False
Augment_add_lines = False
Clip_mat_flag = False
random_clip_flag = False
transform = BaseTransform(  Resample_size,[104])  #gray scale data

class myDataloader(object):
    def __init__(self, batch_size,image_size,path_size):
        self.dataroot = "../dataset/For_contour_train/pic/"
        self.signalroot ="../dataset/For_contour_train/label/" 
        self.read_all_flag=0
        self.read_record =0
        self.folder_pointer = 0
        self.batch_size  = batch_size
        self.img_size  = image_size
        self.path_size  = path_size


        self.input_image = np.zeros((batch_size,1,image_size,image_size))
        self.input_path = np.zeros((batch_size,path_size))
        self.all_dir_list = os.listdir(self.dataroot)
        self.folder_num = len(self.all_dir_list)
        # create the buffer list
        self.folder_list = [None]*self.folder_num
        self.signal = [None]*self.folder_num

        # create all  the folder list and their data list

        number_i = 0
        # all_dir_list is subfolder list 
        #creat the image list point to the STASTICS TIS  list
        saved_stastics = Generator_Contour()
        #read all the folder list
        for subfold in self.all_dir_list:
            #if(number_i==0):
            this_folder_list =  os.listdir(os.path.join(self.dataroot, subfold))
            this_folder_list2 = [ self.dataroot +subfold + "/" + pointer for pointer in this_folder_list]
            self.folder_list[number_i] = this_folder_list2

            #change the dir firstly before read
            #saved_stastics.all_statics_dir = os.path.join(self.signalroot, subfold, 'contour.pkl')
            this_contour_dir =  self.signalroot+ subfold+'/'+ 'contours.pkl' # for both linux and window

            self.signal[number_i]  =  self.read_data(this_contour_dir)
            number_i +=1
            #read the folder list finished  get the folder list and all saved path
    def read_data(self,root):
        data = pickle.load(open(root,'rb'),encoding='iso-8859-1')
        return data
    def gray_scale_augmentation(self,orig_gray) :
        random_scale = 0.7 + (1.5  - 0.7) * random()
        aug_gray = orig_gray * random_scale
        aug_gray = np.clip(aug_gray, a_min = 1, a_max = 254)

        return aug_gray
    def random_min_clip_by_row(self,min1,min2,mat):
         rand= np.random.random_sample()
         rand = rand * (min2-min1) +min1
         H,W = mat.shape
         for i in np.arange(W):
             rand= np.random.random_sample()
             rand = rand * (min2-min1) +min1
             mat[:,i] = np.clip(mat[:,i],rand,254)
         return mat
    def add_lines_to_matrix(self,matrix):
        value  = 128
        H,W = matrix.shape
        line_positions = np.arange(H+10,W-2*H,30)
        for lines in line_positions:
            for i  in np.arange (0, H):
                matrix[i,lines-i] =value
                matrix[i,lines-i+3] =value


        return matrix  
    def read_a_batch(self):
        read_start = self.read_record
        #read_end  = self.read_record+ self.batch_size
        thisfolder_len =  len (self.folder_list[self.folder_pointer])
        
            #return self.input_image,self.input_path# if out this folder boundary, just returen
        this_pointer=0
        i=read_start
        while (1):
        #for i in range(read_start, read_end):
            #this_pointer = i -read_start
            # get the all the pointers 
            #Image_ID , b = os.path.splitext(os.path.dirname(self.folder_list[self.folder_pointer][i]))
            Path_dir,Image_ID =os.path.split(self.folder_list[self.folder_pointer][i])
            Image_ID_str,jpg = os.path.splitext(Image_ID)
            Image_ID = int(Image_ID_str)
            #start to read image and paths to fill in the input bach
            this_image_path = self.folder_list[self.folder_pointer][i] # read saved path
            this_img = cv2.imread(this_image_path)

            #resample 
            #this_img = cv2.resize(this_img, (self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
           
            #get the index of this Imag path
            Path_Index_list = self.signal[self.folder_pointer].img_num[:]
            #Path_Index_list = Path_Index_list.astype(int)
            #Path_Index_list = Path_Index_list.astype(str)

            try:
                Path_Index = Path_Index_list.index(Image_ID)
            except ValueError:
                print(Image_ID_str + "not path exsting")

            else:             
                Path_Index = Path_Index_list.index(Image_ID)            
                this_pathx = self.signal[self.folder_pointer].contoursx[Path_Index]
                this_pathy = self.signal[self.folder_pointer].contoursy[Path_Index]

                #path2 =  signal.resample(this_path, self.path_size)#resample the path
                # concreate the image batch and path
                this_gray  =   cv2.cvtColor(this_img, cv2.COLOR_BGR2GRAY)
                
                # imag augmentation
                if Augment_limitation_flag== True:
                    if  random_clip_flag == True:
            #Costmatrix = np.clip(Costmatrix, 20,254)
                        this_gray=self.random_min_clip_by_row(15,30,this_gray)
                    else:
                        clip_limitation = np.random.random_sample()*20
                        this_gray  = np . clip( this_gray , clip_limitation,255) # change the clip value depend on the ID
                if Augment_add_lines== True:
                    this_gray  = self.add_lines_to_matrix( this_gray  )
                #this_gray = self.gray_scale_augmentation(this_gray)
                H,W = this_gray.shape
                clen = len(this_pathx)
                #img_piece = this_gray[:,this_pathx[0]:this_pathx[clen-1]]
                # no crop blank version 
                factor=self.img_size/W
                img_piece = this_gray 
                img_piece = cv2.resize(img_piece, (self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
                this_pathy =  signal.resample(this_pathy, int(clen*factor))#resample the path
                #resample 
                this_pathy =  this_pathy*self.img_size/H#resample the path

                len1 = len(this_pathy)
                pathl = np.zeros(int(this_pathx[0]*factor))+ 1.5*self.img_size
                len2 = len(pathl)
                pathr = np.zeros(self.img_size-len1-len2) + 1.5*self.img_size
                path_piece = np.append(pathl,this_pathy,axis=0)
                path_piece = np.append(path_piece,pathr,axis=0)

                #path_piece   = np.clip(path_piece,0,self.img_size)
                
                self.input_image[this_pointer,0,:,:] = transform(img_piece)[0]/104.0
                self.input_path [this_pointer , :] = path_piece
                this_pointer +=1
                #if(this_pointer>=self.batch_size): # this batch has been filled
                #        break


            i+=1
            if (i>=thisfolder_len):
                i=0
                self.read_record =0
                self.folder_pointer+=1
                if (self.folder_pointer>= self.folder_num):
                    self.read_all_flag =1
                    self.folder_pointer =0
            if(this_pointer>=self.batch_size): # this batch has been filled
                break
            pass
        self.read_record=i # after reading , remember to  increase it 
        return self.input_image,self.input_path


##test read 
#data  = myDataloader (Batch_size,Resample_size,Path_length)

#for  epoch in range(500):

#    while(1):
#        data.read_a_batch()


