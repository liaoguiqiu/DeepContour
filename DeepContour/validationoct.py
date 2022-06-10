"""
This version of validation takes he existence vector as individual instead of pair
"""

import torch
import  math
from model.base_model import BaseModel
import model.networks as  networks
from test_model import layer_body_sheath_res2
# from test_model import fusion_nets_ivus
import test_model.fusion_nets_multi as fusion_nets_ivus

from test_model.loss_MTL import MTL_loss,DiceLoss
import rendering
from dataset_ivus import myDataloader,Batch_size,Resample_size, Path_length,Reverse_existence,Existence_thrshold,Sep_Up_Low
from time import time
import torch.nn as nn
from torch.autograd import Variable
from databufferExcel import EXCEL_saver
#torch.autograd.set_detect_anomaly(True) # Fix for problem: RuntimeError: one of the variables needed for gradient
# computation has been modified by an inplace operation: [torch.cuda.FloatTensor [1024]] is at version 3;
# expected version 2 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient,
# with torch.autograd.set_detect_anomaly(True).
from working_dir_root import Dataset_root,Output_root
from scipy.spatial.distance import directed_hausdorff, cdist
from skimage import metrics
import numpy as np
Abasence_imageH = 0.5 # penalize the target
class Validation(object):
    def __init__(self):
        if Sep_Up_Low == False :
            self.metrics_saver= EXCEL_saver(14) # bound error 2, HDD 2, Prec 2, Recall 2, Jacard 3, Dice 3,
        else:
            self.metrics_saver = EXCEL_saver(24)  # bound error 4, HDD 4, Prec 4, Recall 4, Jacard 3, Dice 3,, overall J + D 2
    def error_calculation(self,MODEL,Model_key):
        # average Jaccard index J (IoU)
        def cal_J(true, predict,thre=0.5):
            predict = predict >  thre

            AnB = true * predict  # assume that the lable are all binary
            AuB = true + predict
            AuB = torch.clamp(AuB, 0, 1)
            s = 0.0001
            this_j = (torch.sum(AnB) +s) / (torch.sum(AuB) + s)
            return this_j

        # dice cofefficient
        def cal_D(true, predict,thre=0.5):
            predict = predict >  thre
            AB = true * predict  # assume that the lable are all binary
            # AuB = true+predict
            # AuB=torch.clamp(AuB, 0, 1)
            s = 0.0001

            this_d = (2 * torch.sum(AB) +s ) / (torch.sum(true) + torch.sum(predict) + s)
            return this_d

        def cal_L(true, predct):  # the L1 distance of one contour
            # calculate error
            true = torch.clamp(true, 0, 1)
            predct = torch.clamp(predct, 0, 1)

            error = torch.abs(true - predct)
            l = error.size()
            x = torch.sum(error) / l[0]
            return x
        def cal_P (true, predict): # calculate precision

            AnB = true * predict  #  True positive

            s = 0.0001
            this_p = (torch.sum(AnB) ) / (torch.sum(predict) + s)

            return this_p
        def cal_R (true, predict): # calculate Recall

            AnB = true * predict  #  True positive
            Fn =  (true) * (1-predict*1.0)  # false Negative
            s = 0.0001
            this_r = (torch.sum(AnB) ) / (torch.sum(AnB) + torch.sum(Fn) + s)

            return this_r
        def convert_ACE_2_CE(ac,p): # Aline hight vector to coordinates sequence with exsitence vector
            W = len(ac)
            x = np.arange(0, W)
            ac = ac.cpu().detach().numpy()
            p = p.cpu().detach().numpy()
            # select the valite coordiante
            array = np.zeros((W, 2))

            array[:, 0] = x
            array[:, 1] = ac

            # select the new x and y coordinates out with existence
            x_e = x[ np.where( p[:] >= 0.5)] # considering existence x
            c_e = ac[ np.where( p[:] >= 0.5)] # with existence

            W_e = len(x_e)
            array_e = np.zeros((W_e, 2))
            array_e[:, 0] = x_e.astype(int)
            array_e[:, 1] = c_e.astype(int)


            return array
        def cal_HDD (ct,pt,c,p ): #  Hausdorff distance
            # TODO: chose to mask out non existence and existence

            true =  convert_ACE_2_CE(ct,pt)
            predict = convert_ACE_2_CE(c,p)

            # result = metrics.hausdorff_distance (predict ,true )
            # result = max(directed_hausdorff(true, predict)[0], directed_hausdorff(predict, true)[0])
            result = cal_min_d (true,predict)
            return result
        def cal_min_d (true,predict ): #   based on spacial diatance matrix
            # TODO:  an M*N spacial distance matrix, pair-wise distance
            Matrix =  cdist(true, predict, 'euclidean')
            # symatric mean:
            Dtp = Matrix.min(axis=0)
            Dpt = Matrix.min(axis=1)
            result = max( np.max( Dtp), np.max(Dpt))
            return result



        # MODEL.set_requires_grad(MODEL.netG, False)  # D requires no gradients when optimizing G
        # out_pathes_real = MODEL.real_pathes
        if Reverse_existence == True:
            real_exv = 1 - MODEL.real_exv[0]
        real_pathes = MODEL.real_pathes[0]

        real_pathes = MODEL.real_pathes[0] *real_exv+ (1- real_exv)*Abasence_imageH
        # MODEL.validation_cnt += 1
        if MODEL.out_pathes is not None:
            out_pathes_all = MODEL.out_pathes[0]
            out_exv_all = MODEL.out_exis_vs[0]
            # out_pathes_all = MODEL.real_pathes.cpu().detach().numpy() * 0
            # out_exv_all = MODEL.real_exv
            out_pathes = out_pathes_all[0]
            # real_pathes = MODEL.real_pathes[0]
            if Reverse_existence == True:
                out_exv_all = 1 - out_exv_all
            out_exv_all = out_exv_all > Existence_thrshold
            out_exv = out_exv_all[0]
            # merge two exist
            # for i in range(0, len(out_pathes), 2):
            #     out_exv[i] = out_exv[i] * out_exv[i + 1]
            #     out_exv[i + 1] = out_exv[i] * out_exv[i + 1]

            for i in range(len(out_pathes)):
                This_non_exv = (~out_exv[i])
                This_non_exv.type(torch.FloatTensor)
                This_non_exv = This_non_exv.cuda()
                out_pathes[i] = out_pathes[i] * out_exv[i] + This_non_exv*Abasence_imageH
                # if Abasence_imageH == False:
                #     out_pathes[i] = out_pathes[i] * out_exv[i]
            # MODEL.real_pathes = pathes
            # MODEL.real_exv = exis_v

        else: # when there is not path, revise the none with the post-processed path
            # TODO: changed from up-lower to single bound
            if Sep_Up_Low == True:
                out_pathes , out_exv= rendering.onehot2layers_cut_bound(MODEL.fake_B_1_hot[0],Abasence_imageH)
            else:
                out_pathes, out_exv= rendering.onehot2layers(MODEL.fake_B_1_hot[0])
            MODEL.out_pathes=[None]*4
            MODEL.out_exis_vs = [None] * 4
            MODEL.out_pathes[0]=  out_pathes.unsqueeze(0)
            # out_exv  =
            if Reverse_existence == True:
                MODEL.out_exis_vs[0] =1- out_exv.unsqueeze(0)
            else:
                MODEL.out_exis_vs[0] = out_exv.unsqueeze(0)
        # out_pathes = rendering.onehot2layers_cut_bound(MODEL.fake_B_1_hot[0])
        # MODEL.out_pathes[0][0] = out_pathes
        # loss = MODEL.criterionMTL.multi_loss(MODEL.out_pathes,MODEL.real_pathes)
        # MODEL.error = 1.0*loss[0]
        # out_pathes[fusion_predcition][batch 0, contour index,:]
        # cutedge = 1
        # if Reverse_existence == True:
        #     exvP =    MODEL.out_exis_v0 <0.7
        #     exvT =    MODEL.real_exv<0.7
        # else:
        #     exvP = MODEL.out_exis_v0 > 0.7
        #     exvT = MODEL.real_exv > 0.7
        # MODEL.out_pathes[0] = MODEL.out_pathes[0] * exvP + (~exvP) # reverse the mask
        # MODEL.real_pathes = MODEL.real_pathes * exvT + (~exvT)
        # if Abasence_imageH == False:
        #     real_pathes = MODEL.real_pathes[0] * MODEL.real_exv[0]



        MODEL.L = np.zeros(len(out_pathes))
        for i in range(len(out_pathes)):
            MODEL.L[i] = cal_L(out_pathes[i], real_pathes[i]) * Resample_size
            print(" L " + str(i) + '=' + str(MODEL.L[i]))

        # MODEL.L1 = cal_L(MODEL.out_pathes[0][0,0,cutedge:Resample_size-cutedge],MODEL.real_pathes[0,0,cutedge:Resample_size-cutedge]) * Resample_size
        # MODEL.L2 = cal_L(MODEL.out_pathes[0][0,1,cutedge:Resample_size-cutedge],MODEL.real_pathes[0,1,cutedge:Resample_size-cutedge]) * Resample_size

        # Calculate the precision of presence
        MODEL.P = np.zeros(len(out_exv))
        for i in range(len(out_exv)):
            MODEL.P[i] = cal_P(real_exv[i], out_exv[i])
            print(" Precis" + str(i) + '=' + str(MODEL.P[i]))

        # Calculate the recall of presence
        MODEL.R = np.zeros(len(out_exv))
        for i in range(len(out_exv)):
            MODEL.R[i] = cal_R(real_exv[i], out_exv[i])
            print(" Recall" + str(i) + '=' + str(MODEL.R[i]))
        # print (" L1 =  "  + str(MODEL.L1))
        # print (" L2 =  "  + str(MODEL.L2))
        # Hausdorff distance
        MODEL.HDD = np.zeros(len(out_exv))
        for i in range(len(out_exv)):
            MODEL.HDD[i] = cal_HDD(real_pathes[i]*Resample_size,real_exv[i],out_pathes[i] * Resample_size, out_exv[i])
            print(" HDD" + str(i) + '=' + str(MODEL.HDD[i]))

        # calculate J (IOU insetion portion)
        real_b_hot = MODEL.real_B_one_hot[0]
        fake_b_hot = MODEL.fake_B_1_hot[0]
        MODEL.J = np.zeros(len(real_b_hot))
        for i in range(len(real_b_hot)):
            MODEL.J[i] = cal_J(real_b_hot[i], fake_b_hot[i])
            print(" J " + str(i) + '=' + str(MODEL.J[i]))

        MODEL.D = np.zeros(len(real_b_hot))
        for i in range(len(real_b_hot)):
            MODEL.D[i] = cal_D(real_b_hot[i], fake_b_hot[i])
            print(" D " + str(i) + '=' + str(MODEL.D[i]))
        MODEL.OverJD = np.zeros(2)
        MODEL.OverJD [0] = cal_J(real_b_hot, fake_b_hot)
        MODEL.OverJD[1] = cal_D(real_b_hot, fake_b_hot)
        # # this is the format of hot map
        # #out  = torch.zeros([bz,3, H,W], dtype=torch.float)
        # MODEL.J1 = cal_J(real_b_hot[0,0,:,cutedge:Resample_size-cutedge],fake_b_hot[0,0,:,cutedge:Resample_size-cutedge])
        # MODEL.J2 = cal_J(real_b_hot[0,1,:,cutedge:Resample_size-cutedge],fake_b_hot[0,1,:,cutedge:Resample_size-cutedge])
        # MODEL.J3 = cal_J(real_b_hot[0,2,:,cutedge:Resample_size-cutedge],fake_b_hot[0,2,:,cutedge:Resample_size-cutedge])
        # print (" J1 =  "  + str(MODEL.J1 ))
        # print (" J2 =  "  + str(MODEL.J2 ))
        # print (" J3 =  "  + str(MODEL.J3 ))
        #
        #
        #
        # MODEL.D1 = cal_D(real_b_hot[0,0,:,cutedge:Resample_size-cutedge],fake_b_hot[0,0,:,cutedge:Resample_size-cutedge])
        # MODEL.D2 = cal_D(real_b_hot[0,1,:,cutedge:Resample_size-cutedge],fake_b_hot[0,1,:,cutedge:Resample_size-cutedge])
        # MODEL.D3 = cal_D(real_b_hot[0,2,:,cutedge:Resample_size-cutedge],fake_b_hot[0,2,:,cutedge:Resample_size-cutedge])
        # print (" D1 =  "  + str(MODEL.D1 ))
        # print (" D2 =  "  + str(MODEL.D2 ))
        # print (" D3 =  "  + str(MODEL.D3 ))



        vector = np.append(MODEL.L,MODEL.P)

        vector=np.append(vector, MODEL.R )
        vector = np.append(vector,MODEL.HDD)

        vector=np.append(vector, MODEL.J )
        vector = np.append(vector, MODEL.D)
        vector = np.append(vector, MODEL.OverJD)


        # vector = [MODEL.L1,MODEL.L2,MODEL.J1, MODEL.J2,MODEL.J3,MODEL.D1,MODEL.D2,MODEL.D3]
        # vector = torch.stack(vector)
        # vector= vector.cpu().detach().numpy()
        save_dir = Output_root + "1Excel/"+Model_key+'/'
        self.metrics_saver.append_save(vector, save_dir)

        return MODEL

