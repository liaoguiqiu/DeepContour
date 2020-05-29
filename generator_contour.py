import cv2
import math
import numpy as np
import os
import random
from matplotlib.pyplot import *
#from mpl_toolkits.mplot3d import Axes3D


#PythonETpackage for xml file edition
try: 
    import xml.etree.cElementTree as ET 
except ImportError: 
    import xml.etree.ElementTree as ET 
import sys 
#GPU acceleration

from analy_visdom import VisdomLinePlotter
#from numba import vectorize
#from numba import jit
import pickle
from enum import Enum
class Save_Contour_pkl(object):
    def __init__(self ):
        #set = Read_read_check_ROI_label()
        #self.database_root = set.database_root
        #check or create this path
        #self.self_check_path_create(self.signal_data_path)
        self.img_num= []
        self.contoursx = []
        self.contoursy = []
    def self_check_path_create(self,directory):
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
   # add new step of all signals
    def append_new_name_contour(self,number,this_contoursx,this_contoursy,dir):
        #buffer
        self.img_num.append(number)
        self.contoursx.append(this_contoursx)
        self.contoursy.append(this_contoursy)

        #save the data 
        save_path = dir + "seg label pkl/"
        with open(save_path+'contours.pkl', 'wb') as f:
            pickle.dump(self , f, pickle.HIGHEST_PROTOCOL)
        pass
    #read from file
    def read_data(self,base_root):
        saved_path  = base_root  + "seg label pkl/"
        self = pickle.load(open(saved_path+'contours.pkl','rb'),encoding='iso-8859-1')
        return self
class Generator_Contour(object):
    def __init__(self ):
        #set = Read_read_check_ROI_label()
        #self.database_root = set.database_root
        #check or create this path
        #self.self_check_path_create(self.signal_data_path)
        self.img_num= []
        self.contoursx = []
        self.contoursy = []
    def read_data(self,root):
        data = pickle.load(open(root,'rb'),encoding='iso-8859-1')
        return data
    pass