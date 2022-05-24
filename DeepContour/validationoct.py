"""
This version of validation takes he existence vector as individual instead of pair
"""

import torch
from model.base_model import BaseModel
import model.networks as  networks
from test_model import layer_body_sheath_res2
# from test_model import fusion_nets_ivus
import test_model.fusion_nets_multi as fusion_nets_ivus

from test_model.loss_MTL import MTL_loss,DiceLoss
import rendering
from dataset_ivus import myDataloader,Batch_size,Resample_size, Path_length,Reverse_existence,Existence_thrshold
from time import time
import torch.nn as nn
from torch.autograd import Variable
from databufferExcel import EXCEL_saver
#torch.autograd.set_detect_anomaly(True) # Fix for problem: RuntimeError: one of the variables needed for gradient
# computation has been modified by an inplace operation: [torch.cuda.FloatTensor [1024]] is at version 3;
# expected version 2 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient,
# with torch.autograd.set_detect_anomaly(True).
from working_dir_root import Dataset_root,Output_root
import numpy as np
Abasence_imageH = 0.5 # penalize the target
class Validation(object):
    def __init__(self):

        self.metrics_saver= EXCEL_saver(8)

    def error_calculation(self,MODEL,Model_key):
        # average Jaccard index J
        def cal_J(true, predict):
            AnB = true * predict  # assume that the lable are all binary
            AuB = true + predict
            AuB = torch.clamp(AuB, 0, 1)
            s = 0.0001
            this_j = (torch.sum(AnB) + s) / (torch.sum(AuB) + s)
            return this_j

        # dice cofefficient
        def cal_D(true, predict):
            AB = true * predict  # assume that the lable are all binary
            # AuB = true+predict
            # AuB=torch.clamp(AuB, 0, 1)
            s = 0.0001

            this_d = (2 * torch.sum(AB) + s) / (torch.sum(true) + torch.sum(predict) + s)
            return this_d

        def cal_L(true, predct):  # the L1 distance of one contour
            # calculate error
            true = torch.clamp(true, 0, 1)
            predct = torch.clamp(predct, 0, 1)

            error = torch.abs(true - predct)
            l = error.size()
            x = torch.sum(error) / l[0]
            return x

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
            # out_pathes = rendering.onehot2layers_cut_bound(MODEL.fake_B_1_hot[0],Abasence_imageH)
            out_pathes = rendering.onehot2layers(MODEL.fake_B_1_hot[0])
            MODEL.out_pathes=[None]*4
            MODEL.out_pathes[0]= out_pathes


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

        # print (" L1 =  "  + str(MODEL.L1))
        # print (" L2 =  "  + str(MODEL.L2))

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
        vector = np.append(MODEL.L, MODEL.J)
        vector = np.append(vector, MODEL.D)

        # vector = [MODEL.L1,MODEL.L2,MODEL.J1, MODEL.J2,MODEL.J3,MODEL.D1,MODEL.D2,MODEL.D3]
        # vector = torch.stack(vector)
        # vector= vector.cpu().detach().numpy()
        save_dir = Output_root + "1Excel/"+Model_key+'/'
        self.metrics_saver.append_save(vector, save_dir)

        return MODEL

