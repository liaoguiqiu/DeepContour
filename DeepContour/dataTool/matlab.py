import pickle
from scipy import signal 
import scipy.io
import numpy as np
import os

########################class for signal##########################################

class Save_Signal_matlab(object):
      def __init__(self):
          self.flag  = True

          self.save_matlab_root = "D:/PhD/IPB meeting/Dots video dsitance calculation/"
          self.check_dir(self.save_matlab_root)
           
          self.value1=[]
          self.value2=[]
          self.value3=[]
          self.value4=[]

          pass
      def check_dir(self,this_path):
        if not os.path.exists(this_path):
            os.mkdir(this_path)
            print("Directory " , this_path ,  " Created ")
        else:    
            print("Directory " , this_path ,  " already exists")
      def buffer1(self,input ):
            self.value1.append(input)
            

         
            pass 
      def buffer2(self,input,input2):
            self.value1.append(input)
            self.value2.append(input2)
            scipy.io.savemat(self.save_matlab_root+'result.mat', mdict={'arr': self})

            pass 
      def buffer_4(self,id,truth,deep,tradition):
          self.label.append(id)
          self.truth.append(truth)
          self.deep_result.append(deep)
          self.tradition_result.append(tradition)
          pass
      
      def save_mat(self):
          scipy.io.savemat(self.save_matlab_root+'result.mat', mdict={'arr': self})
          pass


      #def save_mat_infor_of_over_allshift_with_NURD(self):
      #      scipy.io.savemat(self.save_matlab_root+'infor_shift_NURD.mat', mdict={'arr': self})
      #      pass
      #def save_pkl_infor_of_over_allshift_with_NURD(self):
      #    with open(self.save_matlab_root+'infor_shift_NURD.pkl', 'wb') as f:
      #      pickle.dump(self , f, pickle.HIGHEST_PROTOCOL)
      #def read_pkl_infor_of_over_allshift_with_NURD(self):
      #    result =     pickle.load(open(self.save_matlab_root+'infor_shift_NURD.pkl','rb'),encoding='iso-8859-1')
      #    #decode the mat data 
      #    nurd = result.NURD
      #    shift = result.overall_shift
      #    id = result.label
      #    return id,nurd,shift
#####################class for generat function#############################################
     

