import cv2
import numpy as np
import os
from analy import MY_ANALYSIS
from generator_contour import  Generator_Contour,Save_Contour_pkl
#,Generator_Contour_layers
from analy import Save_signal_enum
from scipy import signal 
from image_trans import BaseTransform  
from random import seed
from random import random
import pickle

seed(1)
Batch_size = 5
Resample_size =300
Path_length = 300
Augment_limitation_flag = False
Augment_add_lines = False
Clip_mat_flag = False
random_clip_flag = False
transform = BaseTransform(  Resample_size,[104])  #gray scale data

class myDataloader(object):
    def __init__(self, batch_size,image_size,path_size):
        self.dataroot = "../dataset/For_layers_train/pic/"
        self.signalroot ="../dataset/For_layers_train/label/" 
        self.noisyflag = False
        self.read_all_flag=0
        self.read_record =0
        self.folder_pointer = 0
        self.batch_size  = batch_size
        self.img_size  = image_size
        self.path_size  = path_size


        self.input_image = np.zeros((batch_size,1,image_size,image_size))
        self.input_path = np.zeros((batch_size,4,path_size))
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
    def noisy(self,noise_typ,image):
           if noise_typ == "none":
              return image
           if noise_typ == "blur":
              blur =cv2.GaussianBlur(image,(5,5),0)
              return blur
           if noise_typ == "blur2":
              blur =cv2.GaussianBlur(image,(7,7),0)
              return blur
           if noise_typ == "gauss_noise":
              row,col = image.shape
              mean = 0
              var = 50
              sigma = var**0.5
              gauss = np.random.normal(mean,sigma,(row,col )) 
              gauss = gauss.reshape(row,col ) 
              noisy = image + gauss
              return np.clip(noisy,0,254)
           elif noise_typ == 's&p':
              row,col  = image.shape
              s_vs_p = 0.5
              amount = 0.004
              out = np.copy(image)
              # Salt mode
              num_salt = np.ceil(amount * image.size * s_vs_p)
              coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
              out[coords] = 1

              # Pepper mode
              num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
              coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
              out[coords] = 0
              return np.clip(out,0,254)
           elif noise_typ == 'poisson':
              vals = len(np.unique(image))
              vals = 2 ** np.ceil(np.log2(vals))
              noisy = np.random.poisson(image * vals) / float(vals)
              return np.clip(noisy,0,254)
           elif noise_typ =='speckle':
              row,col  = image.shape
              gauss = np.random.randn(row,col )
              gauss = gauss.reshape(row,col )        
              noisy = image + image * gauss
              return np.clip(noisy,0,254)
    def read_data(self,root):
        data = pickle.load(open(root,'rb'),encoding='iso-8859-1')
        return data
    def gray_scale_augmentation(self,orig_gray) :
        random_scale = 0.3 + (1.5  - 0.3) * np.random.random_sample()
        aug_gray = orig_gray * random_scale
        aug_gray = np.clip(aug_gray, a_min = 0, a_max = 254)

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
            Path_Index_list = self.signal[self.folder_pointer].img_num[:]

            #Image_ID = int(Image_ID_str)
            if type(Path_Index_list[0]) is str: 
                Image_ID = str(Image_ID_str)
            else:
                Image_ID = int(Image_ID_str)

            #start to read image and paths to fill in the input bach
            this_image_path = self.folder_list[self.folder_pointer][i] # read saved path
            this_img = cv2.imread(this_image_path)

            #resample 
            #this_img = cv2.resize(this_img, (self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
           
            #get the index of this Imag path
            #Path_Index_list = Path_Index_list.astype(int)
            #Path_Index_list = Path_Index_list.astype(str)

            try:
                Path_Index = Path_Index_list.index(Image_ID)
            except ValueError:
                print(Image_ID_str + "not path exsting")

            else:             
                Path_Index = Path_Index_list.index(Image_ID)  
                #for layers train alll  the x and y are list
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
                #if Augment_add_lines== True:
                #    this_gray  = self.add_lines_to_matrix( this_gray  )
                #this_gray = self.gray_scale_augmentation(this_gray)
                H,W = this_gray.shape
                c_num = len(this_pathx)

                clen = len(this_pathx[0])
                #img_piece = this_gray[:,this_pathx[0]:this_pathx[clen-1]]
                # no crop blank version 
                factor=self.img_size/W
                img_piece = this_gray 
                img_piece = cv2.resize(img_piece, (self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
                #img_piece = self.gray_scale_augmentation(img_piece)
                if self.noisyflag == True:

                    img_piece  = self . noisy( "gauss_noise" ,  img_piece )
                    img_piece  = self . noisy( "s&p" ,  img_piece )

                    img_piece  = self . noisy( "poisson" ,  img_piece )
                 


                for iter in range(c_num):
                    # when consider about  the blaank area :
                        #pathyiter =  signal.resample(this_pathy[iter], int(clen*factor))#resample the path
                        ##resample 
                        #pathyiter =  pathyiter*self.img_size/H#resample the path

                        ## set the blank area ( left or right) of the contour to a specific high value 
                        #pathl = np.zeros(int(this_pathx[iter][0]*factor))+ 2*self.img_size
                        #len1 = len(pathyiter)

                        #len2 = len(pathl)
                        #pathr = np.zeros(self.img_size-len1-len2) + 2*self.img_size
                        #path_piece = np.append(pathl,pathyiter,axis=0)
                        #path_piece = np.append(path_piece,pathr,axis=0)
                    pathyiter =  signal.resample(this_pathy[iter], self.img_size)#resample the path
                    pathyiter =  pathyiter*self.img_size/H
                    
                    self.input_path [this_pointer ,iter, :] = pathyiter
                #path_piece   = np.clip(path_piece,0,self.img_size)
                
                self.input_image[this_pointer,0,:,:] = transform(img_piece)[0]/104.0
                
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


