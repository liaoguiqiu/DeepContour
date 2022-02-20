import torch.utils.data
from torch.autograd import Variable
from model import CE_build3  # the mmodel

# the model
import arg_parse
import cv2
import numpy
import rendering
# from dataTool import generator_contour
from dataTool import generator_contour_ivus

from dataTool.generator_contour import Generator_Contour, Save_Contour_pkl, Communicate
from dataTool.generator_contour_ivus import Generator_Contour_sheath
from dataset_ivus import myDataloader, Batch_size, Resample_size, Path_length,max_presence,Reverse_existence

import os
# from dataset_sheath import myDataloader,Batch_size,Resample_size, Path_length
# switch to another data loader for the IVUS, whih will have both the position and existence vector
from working_dir_root import Dataset_root, Output_root
from deploy.basic_trans import Basic_oper
from scipy import signal

def train_display(MODEL,realA,mydata_loader,Save_img_flag,read_id,infinite_save_id,Model_key):
    gray2 = realA[0, 0, :, :].cpu().detach().numpy() * 104 + 104
    show1 = gray2.astype(float)
    # path2 = mydata_loader.input_path[0,:]
    ##path2  = signal.resample(path2, Resample_size)
    # path2 = numpy.clip(path2,0,Resample_size-1)
    color1 = numpy.zeros((show1.shape[0], show1.shape[1], 3))
    color1[:, :, 0] = color1[:, :, 1] = color1[:, :, 2] = show1[:, :]

    oneHot = MODEL.fake_B_1_hot[0, :, :, :].cpu().detach().numpy()

    hot = numpy.ones((oneHot.shape[1], oneHot.shape[2], 3))
    # change the background to no back ground mask
    #dilute the mask
    hot[:, :, 0] = oneHot[0, :, :]
    hot[:, :, 1] = oneHot[1, :, :]*1.1+ oneHot[0, :, :]+ oneHot[2, :, :]*0.5
    hot[:, :, 2] = oneHot[2, :, :]*1.1+oneHot[0, :, :] + oneHot[1, :, :]*0.5

    oneHot_real = MODEL.real_B_one_hot[0, :, :, :].cpu().detach().numpy()

    hot_real = numpy.ones((oneHot.shape[1], oneHot.shape[2], 3))
    hot_real[:, :, 0] = oneHot_real[0, :, :]
    hot_real[:, :, 1] = oneHot_real[1, :, :]*1.1+oneHot_real[0, :, :] + oneHot_real[2, :, :]*0.5
    hot_real[:, :, 2] = oneHot_real[2, :, :]*1.1+oneHot_real[0, :, :]+oneHot_real[1, :, :]*0.5

    # saveout  = MODEL.fake_B # display encoding tranform
    saveout = MODEL.pix_wise  # middel feature pix encoding
    # saveout = MODEL.fake_B_1_hot  # middel feature pix encoding

    # saveout = rendering.onehot2integer(MODEL.real_B_one_hot)
    show2 = saveout[0, :, :, :].cpu().detach().numpy() * 255

    color = numpy.zeros((show2.shape[1], show2.shape[2], 3))
    color[:, :, 0] = color[:, :, 1] = color[:, :, 2] = numpy.clip(show2[0, :, :], 1, 254)

    # for i in range ( len(path2)):
    #    color = draw_coordinates_color(color,path2[i],i)

    # show3 = numpy.append(show1,show2,axis=1) # cascade
    show4 = numpy.append(color1, color, axis=1)  # cascade
    # the circular of the original image
    circ_original = Basic_oper.tranfer_frome_rec2cir2(color1)

    cv2.imshow('Original circular', circ_original.astype(numpy.uint8))
    if Save_img_flag == True:
        this_save_dir = Output_root + "1out_img/"+Model_key+"/original_circ/"
        if not os.path.exists(this_save_dir):
            os.makedirs(this_save_dir)
        cv2.imwrite(this_save_dir +
                    str(mydata_loader.save_id) + ".jpg", circ_original)
    # infinite_save_id

    cv2.imshow('Deeplearning one', show4.astype(numpy.uint8))
    if Save_img_flag == True:
        this_save_dir = Output_root + "1out_img/"+Model_key+"/Ori_seg_rec/"
        if not os.path.exists(this_save_dir):
            os.makedirs(this_save_dir)
        cv2.imwrite(this_save_dir +
                    str(infinite_save_id) + ".jpg", show4)
    real_label = MODEL.real_B
    real_label = rendering.onehot2integer(real_label)
    show5 = real_label[0, 0, :, :].cpu().detach().numpy() * 255
    cv2.imshow('real', show5.astype(numpy.uint8))
    if Save_img_flag == True:
        this_save_dir = Output_root + "1out_img/"+Model_key+"/ground_rec/"
        if not os.path.exists(this_save_dir):
            os.makedirs(this_save_dir)
        cv2.imwrite(this_save_dir +
                    str(infinite_save_id) + ".jpg", show5)

    # display_prediction(mydata_loader,  MODEL.out_pathes[0],hot)
    # display_prediction(mydata_loader,  MODEL.path_long3,hot)
    # display_prediction(mydata_loader,  MODEL.out_pathes3,hot)
    # display_prediction(read_id,mydata_loader,  MODEL.out_pathes0,hot,hot_real)


    display_prediction(read_id, mydata_loader, MODEL , hot, hot_real, Save_img_flag,Model_key)
    if (MODEL.out_exis_v0 is not None):
        display_prediction_exis(read_id, mydata_loader, MODEL.out_exis_v0)
    return
# 3 functions to drae the results in real time
def draw_coordinates_color(img1 ,vy ,color):
    color_list = [[75, 25, 230], [75, 180, 60], [25, 225, 255], [200, 130, 0], [48, 130, 245],
                  [180, 30, 145], [240, 240, 70], [230, 50, 240], [60, 245, 210], [212, 190, 250],
                  [128, 128, 0], [255, 190, 220], [40, 110, 170], [200, 250, 255], [0, 0, 128],
                  [195, 255, 170], [0, 128, 128], [180, 215, 255], [128, 0, 0]]
    painter = color_list[color]
    # if color ==0:
    #     painter  = [254 ,0 ,0]
    # elif color ==1:
    #     painter  = [0 ,254 ,0]
    # elif color ==2:
    #     painter  = [0 ,0 ,254]
    # else :
    #     painter  = [0 ,0 ,0]
        # path0  = signal.resample(path0, W)
    H ,W ,_ = img1.shape
    for j in range (W):
        # path0l[path0x[j]]
        dy = numpy.clip(vy[j] ,2 , H -2)


        img1[int(dy ) +1 ,j ,: ] =img1[int(dy) ,j ,: ] =painter
        img1[int(dy ) -1 ,j ,: ] =img1[int(dy ) -2 ,j ,: ] =painter

        # img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter


    return img1
def draw_coordinates_color_s(img1 ,vy0 ,vy1):


    H ,W ,_ = img1.shape
    for j in range (W):
        # path0l[path0x[j]]
        dy1 = numpy.clip(vy1[j] ,2 , H -2)
        dy0 = numpy.clip(vy0[j] ,2 , H -2)


        if (dy1 == H- 2):
            img1[int(dy1) + 1, j, :] = img1[int(dy1), j, :] = [0, 254, 254]
            img1[int(dy1) - 1, j, :] = img1[int(dy1) - 2, j, :] = [0, 254, 254]
        if (abs(dy0 - dy1) <= 5):
            img1[int(dy1) + 1, j, :] = img1[int(dy1), j, :] = [254, 0, 254]
            img1[int(dy1) - 1, j, :] = img1[int(dy1) - 2, j, :] = [254, 0, 254]
        # img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter

    return img1


def display_prediction_exis(read_id, mydata_loader, save_out):  # display in coordinates form
    gray2 = (mydata_loader.input_image[0, 0, :, :] * 104) + 104
    show1 = gray2.astype(float)
    path2 = mydata_loader.exis_vec[0, :] * Resample_size
    # path2  = signal.resample(path2, Resample_size)
    path2 = numpy.clip(path2, 0, Resample_size - 1)
    color1 = numpy.zeros((show1.shape[0], show1.shape[1], 3))
    color1[:, :, 0] = color1[:, :, 1] = color1[:, :, 2] = show1

    for i in range(len(path2)):
        color1 = draw_coordinates_color(color1, path2[i], int(i/max_presence))

    show2 = gray2.astype(float)
    save_out = save_out.cpu().detach().numpy()

    save_out = save_out[0, :] * (Resample_size)
    # save_out  = signal.resample(save_out, Resample_size)
    save_out = numpy.clip(save_out, 0, Resample_size - 1)
    color = numpy.zeros((show2.shape[0], show2.shape[1], 3))
    color[:, :, 0] = color[:, :, 1] = color[:, :, 2] = show2

    for i in range(len(save_out)):
        this_coordinate = signal.resample(save_out[i], Resample_size)
        color = draw_coordinates_color(color, this_coordinate, int(i/max_presence))

    # show3 = numpy.append(show1,show2,axis=1) # cascade
    show4 = numpy.append(color1, color, axis=1)  # cascade

    cv2.imshow('Deeplearning exitence 2', show4.astype(numpy.uint8))


def display_prediction(read_id, mydata_loader, MODEL , hot, hot_real,Save_img_flag,Model_key):  # display in coordinates form
    gray2 = (mydata_loader.input_image[0, 0, :, :] * 104) + 104
    show1 = gray2.astype(float)
    path2 = mydata_loader.input_path[0, :]
    # path2  = signal.resample(path2, Resample_size)
    path2 = numpy.clip(path2, 0, Resample_size - 1)
    color1 = numpy.zeros((show1.shape[0], show1.shape[1], 3))
    color1[:, :, 0] = color1[:, :, 1] = color1[:, :, 2] = show1

    for i in range(len(path2)):
        color1 = draw_coordinates_color(color1, path2[i], int(i/max_presence)) # draw duplicate the same color

    show2 = gray2.astype(float)
    if MODEL.out_pathes is not None:
        out_pathes_all= MODEL.out_pathes[0].cpu().detach().numpy()
        out_exv_all =  MODEL.out_exis_vs[0].cpu().detach().numpy()
        # self.real_pathes = pathes
        # self.real_exv = exis_v
    else:
        out_pathes_all = MODEL.real_pathes.cpu().detach().numpy()*0
        out_exv_all = MODEL.real_exv.cpu().detach().numpy()*0
    out_pathes = out_pathes_all[0] * (Resample_size)
    if Reverse_existence ==True:
        out_exv_all = 1-out_exv_all
    out_exv_all = out_exv_all>0.5
    out_exv = out_exv_all[0]
    # MODEL.out_pathes  = signal.resample(MODEL.out_pathes, Resample_size)
    out_pathes = numpy.clip( out_pathes, 0, Resample_size - 1)
    color = numpy.zeros((show2.shape[0], show2.shape[1], 3))
    color[:, :, 0] = color[:, :, 1] = color[:, :, 2] = show2
    colorhot_real = (color  ) * hot_real
    # sheath_real = signal.resample(path2[0], Resample_size)
    # tissue_real = signal.resample(path2[1], Resample_size)
    # colorhot_real = draw_coordinates_color_s(colorhot_real, sheath_real, tissue_real)
    circular_color_real = Basic_oper.tranfer_frome_rec2cir2(colorhot_real)
    cv2.imshow('color real cir', circular_color_real.astype(numpy.uint8))
    if Save_img_flag == True:
        this_save_dir = Output_root + "1out_img/"+Model_key+"/ground_circ/"
        if not os.path.exists(this_save_dir):
            os.makedirs(this_save_dir)
        cv2.imwrite(this_save_dir +
                    str(mydata_loader.save_id) + ".jpg", circular_color_real)


    for i in range(0,len( out_pathes),2):
        out_exv[i] =  out_exv[i] *  out_exv[i+1]
        out_exv[i+1] =  out_exv[i] *  out_exv[i+1]

    for i in range(len( out_pathes)):
        out_pathes[i] = out_pathes[i] * out_exv[i]
        this_coordinate = signal.resample( out_pathes[i], Resample_size)
        color = draw_coordinates_color(color, this_coordinate, int(i/max_presence)) # same color for duplication
    colorhot = (color ) * hot
    colorhot  =numpy.clip(colorhot,1,254)
    # sheath = signal.resample( out_pathes[0], Resample_size)
    # tissue = signal.resample( out_pathes[1], Resample_size)

    color2 = colorhot
    # this is just to draw contact quickly
    # color = draw_coordinates_color_s(color, sheath, tissue)
    # color2 = draw_coordinates_color_s(colorhot, sheath, tissue)
    #
    # color_real = draw_coordinates_color_s(colorhot_real, sheath, tissue)
    color_real = colorhot_real
    # show3 = numpy.append(show1,show2,axis=1) # cascade
    show4 = numpy.append(color1, color, axis=1)  # cascade
    circular1 = Basic_oper.tranfer_frome_rec2cir2(color)
    circular2 = Basic_oper.tranfer_frome_rec2cir2(color2)
    if Save_img_flag == True:
        this_save_dir = Output_root + "1out_img/"+Model_key+"/Ori_seg_rec_line/"
        if not os.path.exists(this_save_dir):
            os.makedirs(this_save_dir)
        cv2.imwrite(this_save_dir +
                    str(mydata_loader.save_id) + ".jpg", show4)

    cv2.imshow('Deeplearning one 2', show4.astype(numpy.uint8))

    cv2.imshow('Deeplearning circ', circular1.astype(numpy.uint8))
    cv2.imshow('Deeplearning circ2', circular2.astype(numpy.uint8))
    if Save_img_flag == True:
        this_save_dir = Output_root + "1out_img/"+Model_key+"/Ori_seg_rec_2/"
        if not os.path.exists(this_save_dir):
            os.makedirs(this_save_dir)
        cv2.imwrite(this_save_dir +
                    str(mydata_loader.save_id) + ".jpg", circular2)
    cv2.imshow('Deeplearning color', color2.astype(numpy.uint8))
    cv2.imshow('  color real', color_real.astype(numpy.uint8))
