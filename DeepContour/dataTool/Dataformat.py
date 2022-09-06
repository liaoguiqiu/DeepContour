import cv2
import numpy as np
import os
from analy import MY_ANALYSIS
# from dataTool import generator_contour
from dataTool import generator_contour_ivus

# from dataTool.generator_contour import  Generator_Contour,Save_Contour_pkl,Communicate
from  dataTool.generator_contour_ivus import  Generator_Contour_sheath,Communicate,Save_Contour_pkl
from working_dir_root import Dataset_root
from analy import Save_signal_enum
from scipy import signal
from image_trans import BaseTransform
from random import seed
from random import random
import  pickle5 as pickle
from distutils.dir_util import copy_tree
print("highest protocol: ", pickle.HIGHEST_PROTOCOL)
from basic_operator import Basic_Operator
from scipy.interpolate import interp1d

Base_dir = "E:/database/Soft phantom synchronized experiment/"
Data_root = Base_dir + "raw/"
target_root_img = Base_dir + "raw_reorganized/img/"
try:
    os.stat(target_root_img)
except:
    os.makedirs(target_root_img)
target_root_label = Base_dir + "raw_reorganized/label/"
try:
    os.stat(target_root_label)
except:
    os.makedirs(target_root_label)


all_dir_list = os.listdir(Data_root)
folder_num = len(all_dir_list)
        # create the buffer list
for subfold in all_dir_list:
    # if(number_i==0):
    this_folder =  os.path.join( Data_root , subfold)
    this_img_fold = this_folder +  '/'+ 'oct_only'
    this_label_fold = this_folder+  '/'+ 'label_generate'

    this_target_img = target_root_img + subfold
    this_target_label = target_root_label + subfold
    try:
        os.stat(this_target_img)
    except:
        os.makedirs(this_target_img)
    try:
        os.stat(this_target_label)
    except:
        os.makedirs(this_target_label)

    copy_tree(this_img_fold, this_target_img)
    copy_tree(this_label_fold, this_target_label)

    print(this_folder + "---- is done !")