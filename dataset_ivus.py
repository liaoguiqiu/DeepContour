#THe data set read the PKL file for Contour with sheath so the contact can be detected
import cv2
import numpy as np
import os
from analy import MY_ANALYSIS
from dataTool import generator_contour 

from dataTool.generator_contour import  Generator_Contour,Save_Contour_pkl,Communicate

from analy import Save_signal_enum
from scipy import signal 
from image_trans import BaseTransform  
from random import seed
from random import random
import pickle
from basic_operator import Basic_Operator
from scipy.interpolate import interp1d

seed(1)
Batch_size = 1
Resample_size =256 # the input and label will be resampled 
Path_length = 256
Augment_limitation_flag = False
Augment_add_lines = False
Clip_mat_flag = False
random_clip_flag = False
Random_rotate = True
transform = BaseTransform(  Resample_size,[104])  #gray scale data

class myDataloader(object):
    def __init__(self, batch_size,image_size,path_size,validation= False,OLG=False):
        self.OLG_flag = OLG
        self.GT = True
        self.save_id =0
        #Guiqiu modified for my computer
        self.com_dir = "../../dataset/telecom/" # this dir is for the OLG
         # initial lizt the 
        self.talker = Communicate()
        self.talker=self.talker.read_data(self.com_dir)
        if self.talker.writing==2:
            self.talker.training =1
        else:
            self.talker.training =2


        self.talker.pending =0 # no pending so all folder can be writed
        #self.talker.writing =2 
        self.talker.save_data(self.com_dir) # save
        root = "D:/Deep learning/dataset/For IVUS/"

        self.dataroot = root + "train/img/"
        self.signalroot =root + "train/label/"
        if self.OLG_flag == True:
           self.dataroot = root + "train_OLG/img/"
           self.signalroot =root + "train_OLG/label/" 


        if validation  == True :
            self.OLG_flag = False
            self.dataroot = root + "test/img/"
            self.signalroot =root + "test/label/" 
        else: 
            self.GT = True  # for  trianing the GT should always be true
                            # FOR TEST IT COULD BE TRUE 
        



        self.noisyflag = False
        self.read_all_flag=0
        self.read_record =0
        self.folder_pointer = 0
        self.batch_size  = batch_size
        self.img_size  = image_size
        self.path_size  = path_size
        self.obj_num = 2 # take num of objects from vectors

        self.input_image = np.zeros((batch_size,1,image_size,image_size))
        # the number of the contour has been increased, and another vector has beeen added
        self.input_path = np.zeros((batch_size,self.obj_num,path_size)) # predifine the path number is 2
        self.exis_vec = np.zeros((batch_size,self.obj_num,path_size)) # predifine the existence vector number is 2

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
              gauss = 0.5*np.random.randn(row,col )
              gauss = gauss.reshape(row,col )        
              noisy = image + image * gauss
              return np.clip(noisy,0,254)
    def read_data(self,root):
        data = pickle.load(open(root,'rb'),encoding='iso-8859-1')
        return data
    def gray_scale_augmentation(self,orig_gray) :
        random_scale = 0.3 + (1.5  - 0.3) * np.random.random_sample()
        random_bias =   (np.random.random_sample() -0.5)*40
        aug_gray = orig_gray * random_scale +random_bias
        aug_gray = np.clip(aug_gray, a_min = 0, a_max = 254)

        return aug_gray
    def nonelinear_scale_augmentation(self,orig_gray) :
        gamma  = np.random.random_sample()*0.6 +0.4

        mask = (orig_gray >50)* gamma
        mask2=  (orig_gray <50) *1
        
         
        aug_gray =  mask * orig_gray + mask2 * orig_gray 

        #aug_gray = np.array(255*(orig_gray.astype(float) / 255) ** gamma, dtype = 'uint8') 
        #random_scale = 0.7 + (1.0  - 0.7) * np.random.random_sample()
        ##aug_gray = orig_gray.astype(float) * orig_gray.astype(float)/100.0*random_scale
        #aug_gray = orig_gray.astype(float) /1.3

        aug_gray = np.clip(aug_gray/2, a_min = 0, a_max = 254)
        Dice = int( np.random.random_sample()*10)
        if Dice % 10 ==0 :
            return orig_gray

            

        else:
            return  aug_gray
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
    def rolls(self,image,pathes,exis_vec):
 
        H,W = image.shape
        roller = np.random.random_sample() * W
        roller = int (roller)
        image = np.roll(image, roller, axis = 1)
        pathes = np.roll(pathes, roller, axis = 1)
        exis_vec =  np.roll(exis_vec, roller, axis = 1)


        return image,pathes,exis_vec
    def flips(self,image,pathes):    #upside down
 
        H,W = image.shape
        fliper = np.random.random_sample() * 10
        dice = int (fliper)%3 #reduce the possibility of the flips 
        if dice ==0:
            image=cv2.flip(image, 0) # flip upside down
            #image = np.roll(image, roller, axis = 1)
            pathes =H/2+(H/2-pathes)



        return image,pathes 
    def flips2(self,image,pathes,exs_p):  # left to right
 
        H,W = image.shape
        fliper = np.random.random_sample() * 10
        dice = int (fliper)%3 #reduce the possibility of the flips 
        if dice ==0:
            image=cv2.flip(image, 1) # flip  horizon
            #image = np.roll(image, roller, axis = 1)
            pathes =np.flip(pathes, 1)
            exs_p = np.flip(exs_p, 1)


        return image,pathes, exs_p 
    def disc_vector_resample(self,py,exis,px,H,W,H2,W2):
        def resample(x,n,kind='nearest'):
            factor = float(x.size/n)
            f = interp1d(np.linspace(0,1,x.size),x,kind)
            return f(np.linspace(0,1,n))
        # use cv2 resampling wiht   nearest interpolation
        clen = len(px)
                #img_piece = this_gray[:,this_pathx[0]:this_pathx[clen-1]]
                # no crop blank version 
        factor=W/W2
        #factor=W2/W
        #nearest_py = 
        #xp = np.arange(0, len(a), factor)
        #nearest_py = interp1d(np.arange(clen), py, kind='nearest')
        #nearest_exi = interp1d(np.arange(clen), exis, kind='nearest')

        this_pathy = resample(py,W2)
        #
        # #this_pathy =  signal.resample(py, int(clen*factor))#resample the path
        # #! resample binary vextor need to be checked
        existnence = resample(exis,W2)
        path_piece =  this_pathy*H2/H
        return path_piece,existnence
        
    def coordinates_and_existence(self,py,exis,px,H,W,H2,W2):
        # this function input the original coordinates of contour x and y, orginal image size and out put size

        clen = len(px)
                #img_piece = this_gray[:,this_pathx[0]:this_pathx[clen-1]]
                # no crop blank version 
        factor=W2/W
       

         
        this_pathy =  signal.resample(py, int(clen*factor))#resample the path
        #! resample binary vextor need to be checked
        existnence =  signal.resample(exis, int(clen*factor))# ! resample binary vextor need to be checked
        #resample 
        this_pathy =  this_pathy*H2/H#unifrom
        # first determine the lef piece
        pathl = np.zeros(int(px[0]*factor))+ 1.11*H2
        len1 = len(this_pathy)
        len2 = len(pathl)
        pathr = np.zeros(W2-len1-len2) + 1.11*H2
        path_piece = np.append(pathl,this_pathy,axis=0)
        path_piece = np.append(path_piece,pathr,axis=0)

        #in the down sample pathy function the dot with no label will be add a number of Height, 
        #however because the label software can not label the leftmost and the rightmost points,
        #so it will be given a max value,  I crop the edge of the label, remember to crop the image correspondingly .

         # convert the blank value to extrem high value
        mask = path_piece >(H2-5)
        path_piece = path_piece + mask * H2*0.2
        #existnence = mask * 1.0
        #path_piece = signal.resample(path_piece[3:W2-3], W2)
        return path_piece,existnence

    def read_a_batch(self):
        read_start = self.read_record
        
        #return self.input_image,self.input_path# if out this folder boundary, just returen
        this_pointer=0
        i=read_start
        this_folder_list  = self.folder_list[self.folder_pointer]
        #read_end  = self.read_record+ self.batch_size
        this_signal = self.signal[self.folder_pointer]
        if self.OLG_flag ==True:
            # check
            self.talker=self.talker.read_data(self.com_dir)

            if self.talker.training ==1:
               this_folder_dir = self.dataroot+"1/"
               this_folder_list =  os.listdir(self.dataroot+"1/")
               # convert subfolder list to full folder listr
               this_folder_list2 = [ self.dataroot +"1/" + "/" + pointer for pointer in this_folder_list]
               this_folder_list = this_folder_list2
               this_contour_dir =  self.signalroot+ "1/"+ 'contours.pkl' # for both linux and window
               this_signal  =  self.read_data(this_contour_dir)
               pass
            elif self.talker.training ==2:
               this_folder_dir = self.dataroot+"2/"
               this_folder_list =  os.listdir(self.dataroot+"2/")
               # convert subfolder list to full folder listr
               this_folder_list2 = [ self.dataroot +"2/" + "/" + pointer for pointer in this_folder_list]
               this_folder_list = this_folder_list2
               this_contour_dir =  self.signalroot+ "2/"+ 'contours.pkl' # for both linux and window
               this_signal  =  self.read_data(this_contour_dir)
               pass

        #thisfolder_len =  len (this_signal.img_num)
        thisfolder_len =  len (this_folder_list)


        while (1):
            if self.OLG_flag ==False:
                this_folder_list  = self.folder_list[self.folder_pointer]
        #read_end  = self.read_record+ self.batch_size
                this_signal = self.signal[self.folder_pointer]
                #thisfolder_len =  len (this_signal.img_num)
                thisfolder_len =  len (this_folder_list)


        #for i in range(read_start, read_end):
            #this_pointer = i -read_start
            # get the all the pointers 
            #Image_ID , b = os.path.splitext(os.path.dirname(self.folder_list[self.folder_pointer][i]))
            Path_dir,Image_ID =os.path.split(this_folder_list[i])
            Image_ID_str,jpg = os.path.splitext(Image_ID)
            Path_Index_list = this_signal.img_num[:]

            #Image_ID = int(Image_ID_str)
            if type(Path_Index_list[0]) is str: 
                Image_ID = str(Image_ID_str)
            else:
                Image_ID = int(Image_ID_str)
            self.save_id= Image_ID
            #start to read image and paths to fill in the input bach
            this_image_path = this_folder_list[i] # read saved path
            this_img = cv2.imread(this_image_path)

            #resample 
            #this_img = cv2.resize(this_img, (self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
           
            #get the index of this Imag path
            #Path_Index_list = Path_Index_list.astype(int)
            #Path_Index_list = Path_Index_list.astype(str)

            try:
                if self.GT == False:
                    Path_Index = 0 # just use the first path, that is fake ground truth, just for testing
                else:
                    Path_Index = Path_Index_list.index(Image_ID)
            except ValueError:
                print(Image_ID_str + "not path exsting")

            else:
                if self.GT == True:
                    Path_Index = Path_Index_list.index(Image_ID)  
                #for layers train alll  the x and y are list
                this_pathx = np.array(list(this_signal.contoursx[Path_Index].values())[0:self.obj_num])
                this_pathy = np.array(list(this_signal.contoursy[Path_Index].values())[0:self.obj_num])
                this_exist = np.array(list(this_signal.contours_exist[Path_Index].values())[0:self.obj_num])
                #path2 =  signal.resample(this_path, self.path_size)#resample the path
                # concreate the image batch and path
                this_gray  =   cv2.cvtColor(this_img, cv2.COLOR_BGR2GRAY)
                #this_gray  = this_gray[0:600,:]   # used to crop the image for IVUS image
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
                #this_gray = self.nonelinear_scale_augmentation(this_gray)
                H,W = this_gray.shape
                c_num = len(this_pathx)

                clen = len(this_pathx[0])
                #img_piece = this_gray[:,this_pathx[0]:this_pathx[clen-1]]
                # no crop blank version 
                factor=self.img_size/W
                img_piece = this_gray #[60:H,:] 
                img_piece = cv2.resize(img_piece, (self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
                if self.noisyflag == True:
                    img_piece = self.nonelinear_scale_augmentation(img_piece)

                    #img_piece = self.gray_scale_augmentation (img_piece)
                    #img_piece= Basic_Operator.add_speckle_or_not(img_piece)
                    #img_piece= Basic_Operator.add_noise_or_not(img_piece)
                    img_piece = Basic_Operator.add_gap_or_not(img_piece)
                    img_piece  = self . noisy( "gauss_noise" ,  img_piece)
                    #img_piece  = self . noisy( "s&p" ,  img_piece )

                    #img_piece  = self . noisy( "speckle" ,  img_piece )
                 


                for iter in range(c_num):
                    # when consider about  the blaank area :
                         
                    pathyiter  =  this_pathy[iter]
                    pathxiter  =  this_pathx [iter]
                    exis_iter= this_exist[iter]
                    # change the raw annotation into new perAline coordinates and existence vecor
                    path_piece,existence_p=self.disc_vector_resample(pathyiter,exis_iter,pathxiter,H,W,self.img_size,self.img_size)
                    # when consider about  the blaank area,and use the special resize :



                    #pathyiter =  signal.resample(this_pathy[iter], self.img_size)#resample the path
                    #pathyiter =  pathyiter*self.img_size/H
                    
                    #self.input_path [this_pointer ,iter, :] = pathyiter
                    self.input_path [this_pointer ,iter, :] = path_piece
                    self.exis_vec  [this_pointer ,iter, :] = existence_p
                #path_piece   = np.clip(path_piece,0,self.img_size)
                #////------test modification code should be modified after the true data ------------////////////
                # just force the last element of t he label to be the same as the second one
                #self.input_path [this_pointer ,2, :] = self.input_path [this_pointer ,1, :] 
                #self.exis_vec  [this_pointer ,2, :] = self.exis_vec  [this_pointer ,1, :]

                if Random_rotate == True:
                    img_piece, self.input_path [this_pointer ,:, :],self.exis_vec [this_pointer ,:, :] =self.rolls(img_piece,self.input_path [this_pointer ,:, :],self.exis_vec [this_pointer ,:, :])  
    
                #img_piece, self.input_path [this_pointer ,:, :] = self.flips(img_piece,self.input_path [this_pointer ,:, :])
                img_piece, self.input_path [this_pointer ,:, :],self.exis_vec[this_pointer ,:, :] =  self.flips2(img_piece,self.input_path [this_pointer ,:, :],self.exis_vec[this_pointer ,:, :])
                


                self.input_image[this_pointer,0,:,:] = transform(img_piece)[0]/104.0
                


                this_pointer +=1
                #if(this_pointer>=self.batch_size): # this batch has been filled
                #        break


            i+=1
            if (i>=thisfolder_len):
                i=0
                self.read_record =0
                self.read_all_flag =1
                if self.OLG_flag == True :
                    #check
                    self.talker=self.talker.read_data(self.com_dir)
                    if self.talker.pending ==1:# pending , finish writing, so switch!
                        self.talker.pending =0 # reset pending
                        if self.talker.training==1:
                            self.talker.training=2
                        else:
                            self.talker.training=1
                        # change writing target
                    
                        self.talker.save_data(self.com_dir)
                    pass

                else:
                    self.folder_pointer+=1
                    if (self.folder_pointer>= self.folder_num):
                        self.folder_pointer =0
                    pass
            #self.read_record=i # after reading , remember to  increase it 
            
            if(this_pointer>=self.batch_size): # this batch has been filled
                break
            pass
        self.read_record=i # after reading , remember to  increase it 

        # additionally returen the exsitence pssibility 
        return self.input_image,self.input_path, self.exis_vec


##test read 
#data  = myDataloader (Batch_size,Resample_size,Path_length)

#for  epoch in range(500):

#    while(1):
#        data.read_a_batch()


