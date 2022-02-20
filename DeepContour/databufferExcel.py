import json as JSON
import cv2
import math
import numpy as np
import os
import random 
from zipfile import ZipFile
import scipy.signal as signal
import pandas as pd
from pathlib import Path
class EXCEL_saver(object):
    def __init__(self, num = 10 ):
        #self.dir = "D:/Deep learning/out/1out_img/Ori_seg_rec_Unet/"
        self.plots = np.zeros((1,num))  
        self.plots[0,:] = np.arange(num)
        #self.firstflag = False
    def append_save(self,vector,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.plots = np.append(self.plots, [vector], axis=0)
        DF1 = pd.DataFrame(self.plots)
        DF1.to_csv(save_dir+"error_buff.csv")