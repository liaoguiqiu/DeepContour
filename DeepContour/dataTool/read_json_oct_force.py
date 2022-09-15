# THis read Json will compute the tactile by the way
# When the data is saved to PKL, two values: force and tactile will be added
# TODO: From the identification, the force read from the scale is about 3 Frames behind the OCT
# Log of modification cre
# this  is used  to read json files and transfer into a pkl file
# Guiqiu modify this for spliting the data for train validation
# Guiqiu modify this to have both upper and lower
import json as JSON
import cv2
import math
import numpy as np
import os
import random
from zipfile import ZipFile
import scipy.signal as signal
import pandas as pd
from collections import OrderedDict
from dataTool.generator_contour_ivus import Save_Contour_pkl,Save_force_contour_pkl
from working_dir_root import Dataset_root
from databufferExcel import EXCEL_saver
from deploy.basic_trans import Basic_oper

import csv
Train_validation_split = True  # flag for devide the data
Train_validation_devi = 3  # all data are equally devided by thsi number
Test_fold = 0  # use the 0 st for training, the other for validation
Delete_outsider_flag = False
Consider_overlapping = False
Process_all_folder_flag = True
class Read_read_check_json_label(object):
    def __init__(self,sub_folder="Endoscopic Phantom No trqns -110 0 alpha 995 +stab/"):
        # self.image_dir   = "../../OCT/beam_scanning/Data set/pic/NORMAL-BACKSIDE-center/"
        # self.roi_dir =  "../../OCT/beam_scanning/Data set/seg label/NORMAL-BACKSIDE-center/"
        # self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL/"
        # self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE-center/"
        # self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE/"
        #self.database_root = "../../dataset/ivus/"
        self.database_root = Dataset_root + "label data/"
        self.excel_saver = EXCEL_saver(4) # save the distance, contact region, and merged value,and force
        # self.database_root = "D:/Deep learning/dataset/label data/"

#<<<<<<< HEAD:DeepContour/dataTool/read_json_ivus.py
        #sub_folder = "animal2/"
#=======
        # sub_folder = "capsule_sample/"

        # Tania's computer

#>>>>>>> b8bb1d19b916df000a1ab2c21c7474cf6fa38b44:dataTool/read_json_ivus.py
        self.max_presence = 9
        self.image_dir = self.database_root + "img/" + sub_folder
        self.json_dir = self.database_root + "label/" + sub_folder
        self.save_dir = self.database_root + "seg label pkl/" + sub_folder
        self.save_dir_train = self.database_root + "seg label pkl train/" + sub_folder
        self.save_dir_test = self.database_root + "seg label pkl test/" + sub_folder
        self.save_dir_origin_cir = self.database_root + "img origin cir/" + sub_folder
        self.save_dir_seg_rect = self.database_root + "img seg rect/" + sub_folder
        self.save_dir_seg_cir = self.database_root + "img seg cir/" + sub_folder

        self.excel_force_dir = self.database_root + "excel_force_singnals/" + sub_folder
        self.save_excel_dir = self.database_root + "tactile_excel/" + sub_folder

        self.img_num = 0

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.save_dir_train):
            os.makedirs(self.save_dir_train)
        if not os.path.exists(self.save_dir_test):
            os.makedirs(self.save_dir_test)
        if not os.path.exists(self.save_excel_dir):
            os.makedirs(self.save_excel_dir)

        if not os.path.exists(self.save_dir_origin_cir):
            os.makedirs(self.save_dir_origin_cir)
        if not os.path.exists(self.save_dir_seg_rect):
            os.makedirs(self.save_dir_seg_rect )
        if not os.path.exists(self.save_dir_seg_cir):
            os.makedirs(self.save_dir_seg_cir)



        # self.contours_x = []  # no predefines # predefine there are 4 contours
        # self.contours_y = []  # predefine there are 4 contours

        self.saver = Save_force_contour_pkl()
        self.saver_train = Save_force_contour_pkl()
        self.saver_test = Save_force_contour_pkl()
        self.display_flag = True
        # hard ecodeing the index number to ensure the index is correspondng tp a specificlly value
        self.labels_lists = {
            'catheter': ['1', 'catheter','Catheter', 'test'],
            'lumen': ['2', 'vessel', 'lumen','tissue'],
            # 'wire': ['3','guide-wire', 'guidewire'],
            # 'media': ['4','vessel (media)', 'vessel(media)', 'media'],
            # 'branch': ['5','vessel(side-branch)', 'vessel (side-branch)', 'vessel(sidebranch)', 'vessel (sidebranch)',
            #            'side-branch', 'sidebranch', 'bifurcation'],
            # 'stent': ['6','stent'],
            # 'plaque': ['7','plaque'],
            # 'calcium': ['8','calcification', 'calcium'],
        }

        # hard ecodeing the index number to ensure the index is correspondng tp a specificlly value
        # self.labels_lists = {
        #     # 'catheter_u': ['1', 'catheter_u', 'test'],
        #     # 'catheter_l': ['2', 'catheter_l', 'test'],
        #     # 'lumen_u': ['3', 'lumen_u', 'lumen'],
        #     # 'lumen_l': ['4', 'lumen_l', 'lumen'],
        #     # 'wire_u': ['5', 'wire_u', 'guidewire'],
        #     # 'wire_l': ['6', 'wire_l', 'guidewire'],
        #     # 'media_u': ['7', 'media_u', 'vessel(media)', 'media'],
        #     # 'media_l': ['8', 'media_l', 'vessel(media)', 'media'],
        #     # 'branch_u': ['9', 'branch_u', 'vessel (side-branch)', 'vessel(sidebranch)', 'vessel (sidebranch)',
        #     #            'side-branch', 'sidebranch', 'bifurcation'],
        #     # 'branch_l': ['10', 'branch_l', 'vessel (side-branch)', 'vessel(sidebranch)', 'vessel (sidebranch)',
        #     #            'side-branch', 'sidebranch', 'bifurcation'],
        #     'plaque_u': ['1', 'plaque_u'],
        #     'plaque_l': ['2', 'plaque_l'],
        #     'calcium_u': ['3', 'calcium_u', 'calcium'],
        #     'calcium_l': ['4', 'calcium_l', 'calcium'],
        # }

        self.disease_labels = ['plaque', 'calcium']

        # BGR because of OpenCV
        self.color_list = [[75, 25, 230], [75, 180, 60], [25, 225, 255], [200, 130, 0], [48, 130, 245],
                           [180, 30, 145], [240, 240, 70], [230, 50, 240], [60, 245, 210], [212, 190, 250],
                           [128, 128, 0], [255, 190, 220], [40, 110, 170], [200, 250, 255], [0, 0, 128],
                           [195, 255, 170], [0, 128, 128], [180, 215, 255], [128, 0, 0]]

    def draw_coordinates_color(self, _img, vx, vy, color_idx):

        painter = self.color_list[color_idx]
        h, w, _ = _img.shape

        for j in range(len(vx)):
            dy = np.clip(vy[j], 1, h - 2)  # clip in case coordinate y is at the border (to paint the y +1 and -1)
            # dx = np.clip(vx[j], 0, w - 1)  # x coordinates are already one per A-line and within the image boundaries
            dx = int(vx[j])

            # _img[int(dy) + 1, int(dx), :] = _img[int(dy), int(dx), :] = painter
            _img[int(dy) + 1, dx, :] = _img[int(dy) - 1, dx, :] = _img[int(dy), dx, :] = painter

        return _img
    def tactile_compute (self,y1,y2,H,W):
        y = y2 - y1
        min_idex = np.argmin(y)
        min_dis = y[min_idex]
        # min_dis_l2 = np.append(min_dis_l2, [min_dis], axis=0)
        # min_dis_l2 = np.delete(min_dis_l2, 0, axis=0)
        # min_dis = sorted(min_dis_l2)[len(min_dis_l2)//2]
        # msg = Floats()
        if min_dis < 10: # if min dix is clos use it as threshould
            thres = min_dis+5
        else:
            thres = 10

        contact_contour = y2 * (y < thres)
        contact = sum(y < thres)  # the size of the contact region
        # contact_l = np.append(contact_l, [contact], axis=0)
        # contact_l = np.delete(contact_l, 0, axis=0)
        # contact = sorted(contact_l)[len(contact_l)//2]

        distance_uni = min_dis/H
        contact_uni = contact/W
        integrate = distance_uni - contact_uni
        # input the contour of catheter sheath and the tissue contour, img H and W
        return distance_uni,contact_uni,integrate,contact_contour
    def check_one_folder(self):
        #read the excel force signal
        if not os.path.exists(self.excel_force_dir +'error_buff.csv'):
            return
        force_signal = pd.read_csv (self.excel_force_dir +'error_buff.csv')
        filter_force = force_signal.values[:,3] # last column is the filtered force
        index_shift = 4 # actually 3, but read from excel fist row is empty
        imagelist = os.listdir(self.image_dir)
        _, b_i = os.path.splitext(imagelist[0])  # first image of this folder
        within_folder_i = 0
        for i in os.listdir(self.json_dir):
            # for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
            # separate the name of json
            a, b = os.path.splitext(i)

            img_index=int(a)

            # if it is a json it will have corresponding image
            if b == ".json":
                # with ZipFile(self.image_dir, 'r') as zipObj:
                #     listOfFiles = zipObj.namelist()
                # TODO: Extract image ext automatically
                # img_path = self.image_dir + a + ".tif"
                img_path = self.image_dir + a + b_i
                img1 = cv2.imread(img_path)
                if img1 is None:
                    print('No img with path: {0}'.format(img_path))
                elif (img_index + index_shift)>(len(filter_force)-1) or (img_index)<10:
                    print('No  force with path: {0}'.format(img_path))
                else:
                    this_force = filter_force[img_index + index_shift]
                    json_dir = self.json_dir + a + b
                    with open(json_dir) as f_dir:
                        data = JSON.load(f_dir)
                    shape = data['shapes']
                    num_labels = len(shape)
                    # len_labels_list = len(self.labels_lists)
                    # len_labels_list = 2
                    # with ZipFile(json_dir, 'r') as zipObj:
                    #       # Get list of files names in zip
                    #       #listOfiles = zipObj.namelist()
                    #       # this line of code is important since the the former one will change the sequence
                    #       listOfiles = zipObj.infolist()
                    #       len_list = len(listOfiles)

                    # rois = read_roi_zip(roi_dir) # get all the coordinates
                    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # transfer into gray image
                    H, W = gray.shape

                    #### Using OrderedDict for backward compatibility since index is important
                    # contours_x is initialized to match the image width for all labels (we need a value per A-line)
                    # contours_x = OrderedDict({key: np.arange(0, W) for key in self.labels_lists.keys()})
                    # Guiqiu update using the array <- multi-dimention list
                    num_label_list = len(self.labels_lists)
                    contours_x = np.zeros((num_label_list, self.max_presence,W))
                    contours_x[0:num_label_list,0:self.max_presence,:] = np.arange(0, W) # copy this one D array to the 3D array
                    # contours_y is initialized to the image height for all A-lines
                    # contours_y = OrderedDict({key: np.ones(W) * H - 1 for key in self.labels_lists.keys()})
                    contours_y = np.zeros((num_label_list, self.max_presence, W)) + H - 1
                    # Initialize existence vectors per label = zeros to all A-lines
                    # contours_exist = OrderedDict({key: np.zeros(W, dtype=int) for key in self.labels_lists.keys()})
                    contours_exist = np.zeros((num_label_list, self.max_presence, W))

                    obj_cnt_list = np.zeros(num_label_list) # this list is used to record count of each object that already appeared(duplicated count)
                    for idx in range(num_labels):

                        # TODO: consider add skip for specific labels
                        # if [key for key, value in self.labels_lists.items() if shape[idx]['label'] in value][0] \
                        #         in self.disease_labels:
                        #     continue  # skip to next iteration

                        coordinates = np.array(shape[idx]['points'])

                        # delete the coordinates outside the image boundaries
                        len_ori, _ = coordinates.shape
                        if Delete_outsider_flag:
                            target = 0
                            for j in range(len_ori):
                                # check the x and y coordinates outside the image boundaries -> delete if out
                                if (coordinates[target, 0] < 0 or coordinates[target, 0] > (W - 1)) \
                                        or (coordinates[target, 1] < 0 or coordinates[target, 1] > (H - 1)):
                                    coordinates = np.delete(coordinates, target, axis=0)
                                else:
                                    target += 1

                        path_x = coordinates[:, 0]
                        path_y = coordinates[:, 1]

                        num_points = len(path_x)
                        if int(path_x[num_points - 1] - path_x[0]) == 0:
                            path_w = 1  # when we have points too close to each other/same integer (short vectors)
                            # TODO: consider disease labels as they can cause this and they are not short vectors
                            #  (skipping now)
                        else:
                            path_w = int(path_x[num_points - 1] - path_x[0])

                        # sometimes the contour is in a reversed direction
                        if path_w < 0:
                            path_w = -path_w
                            path_y = path_y[::-1]
                            path_x = path_x[::-1]

                        path_yl = np.ones(int(path_w)) * np.nan

                        for j in range(num_points):
                            # important sometimes the start point is not the lestmost
                            this_index = np.clip(path_x[j] - path_x[0], 0, path_w - 1)
                            path_yl[int(this_index)] = float(path_y[j])

                        add_3 = np.append(path_yl[::-1], path_yl, axis=0)  # cascade
                        add_3 = np.append(add_3, path_yl[::-1], axis=0)  # cascade
                        s = pd.Series(add_3)
                        path_yl = s.interpolate(method='linear')

                        path_yl = path_yl[path_w:2 * path_w].to_numpy()
                        path_xl = np.arange(int(path_x[0]), int(path_x[0]) + path_w)

                        if len(path_xl) > 0.96 * W and len(path_xl) != W:  # correct the 'imperfect' label contours
                            # remember to add resacle later TODO: what is this resacle? ask Guiqiu
                            path_yl = signal.resample(path_yl, W)
                            path_xl = np.arange(0, W)

                        #### Fill OrderedDict for the three vectors depending on the label
                        # Note: Labels with no data (from the self.labels_lists) remain empty

                        if shape[idx]["label"].lower() in list(
                                label for sublist in list(self.labels_lists.values()) for label in sublist):
                            current_label = \
                                [key for key, value in self.labels_lists.items() if shape[idx]['label'] in value][0]

                            # # initialize overlap_x as empty in case there is no overlap (to use in display)
                            # overlap_x = np.array([])
                            # guiqiu used array <- list
                            # get the index number sub 1 if it is not start from 0
                            label_index_num = int( self.labels_lists[current_label][0]  ) -1
                            repeat = int(obj_cnt_list[label_index_num])
                            # count that this type of objects appear once again
                            obj_cnt_list[label_index_num] = obj_cnt_list[label_index_num] + 1

                            contours_exist[label_index_num][repeat][path_xl] = 1
                            contours_y[label_index_num][repeat][path_xl] = path_yl
                            # # Check if contour is close to the image height and set existence contour to 0
                            # # to handle when there is no back-scattering but the manual label was put close to H
                            contours_exist[label_index_num][repeat] = contours_exist[label_index_num][repeat] * (contours_y[label_index_num][repeat] < (0.95*H ))


                            # # Check if contour is close to the image height and set existence contour to 0
                            # # to handle when there is no back-scattering but the manual label was put close to H
                            # contour_y_close_h = np.where(contours_y[current_label] >= 0.96 * H)[0]
                            # contours_exist[current_label][contour_y_close_h] = 0

                        pass



                    # re-encode the annotation ad reduce the dimention (flaten)
                    # contours_y = np.zeros((num_label_list, self.max_presence, W)) + H - 1
                    # contours_exist = np.zeros((num_label_list, self.max_presence, W))
                    # contours_y_flat = np.zeros((num_label_list * self.max_presence, W))
                    # contours_exist_flat = np.zeros((num_label_list * self.max_presence, W))
                    if (Consider_overlapping == True): # if considering overlapping the contour will be encoded separetely
                        contours_y = contours_y.reshape(num_label_list * self.max_presence, W)
                        contours_exist = contours_exist.reshape(num_label_list * self.max_presence, W)
                        contours_x = contours_x.reshape(num_label_list * self.max_presence, W)
                    else:
                        contours_y = contours_y*contours_exist # change the border value to zero
                        contours_y = np.sum(contours_y,axis = 1)

                        # contours_y = contours_y.reshape(num_label_list * self.max_presence, W)
                        contours_exist = np.sum(contours_exist, axis = 1)
                        contours_exist = np.clip(contours_exist,0,1) # limit

                        contours_y = contours_y + (H-1)*(1-contours_exist)
                        contours_y = np.clip(contours_y, 0, H-1)  # limit

                        contours_x = contours_x[:,0,:] # just remain one dimention of x

                    distance_uni, contact_uni, integrate,contact_contour = self.tactile_compute(contours_y[0],contours_y[1],H,W)
                    excel_vector = [distance_uni,contact_uni,integrate,this_force]
                    self.excel_saver.append_save(excel_vector, self.save_excel_dir)

                    img_origin_cir = Basic_oper.tranfer_frome_rec2cir2(img1[0:H,:])
                    if self.display_flag:  # for loop for display out of previous loop in case of overlap of contours
                        for id in range(len(contours_exist[:,0])):
                            if 1 in contours_exist[id,:]:
                                clr_id  = id
                                if Consider_overlapping ==True:
                                    clr_id = int(id/ self.max_presence)
                                # draw duplicated as the same color
                                img1 = self.draw_coordinates_color(img1, contours_x[id] ,
                                                                   contours_y[id],
                                                                   int(clr_id))
                                # img1 = self.draw_coordinates_color(img1, contours_x[id][
                                #     np.where( contours_exist[id,:] == 1)], contours_y[id][np.where(contours_exist[id,:] == 1)], int(id/self.max_presence))
                        img1 = self.draw_coordinates_color(img1, contours_x[0],
                                                          contact_contour,
                                                          int(clr_id)+1)
                    img_seg_rectan = img1
                    img_seg_cir = Basic_oper.tranfer_frome_rec2cir2(img_seg_rectan[0:H,:])

                    # save this result
                    self.img_num = a  # TODO: why assigning a to another variable?

                    # TODO: check if append_new_name_contour is used anywhere else other than in this script
                    self.saver.append_new_name_contour(self.img_num, contours_x, contours_y, contours_exist,distance_uni, contact_uni, this_force,
                                                       self.save_dir)
                    # [distance_uni, contact_uni, integrate, this_force]
                    # [distance_uni, contact_uni, this_force]

                    if Train_validation_split:
                        if within_folder_i % Train_validation_devi == Test_fold:
                            self.saver_test.append_new_name_contour(self.img_num, contours_x, contours_y,
                                                                    contours_exist,distance_uni, contact_uni, this_force,
                                                                    self.save_dir_test)
                        else:
                            self.saver_train.append_new_name_contour(self.img_num, contours_x, contours_y,
                                                                     contours_exist,distance_uni, contact_uni, this_force,
                                                                     self.save_dir_train)
                    else:
                        self.saver_train.append_new_name_contour(self.img_num, contours_x, contours_y,
                                                                     contours_exist,distance_uni, contact_uni, this_force,
                                                                     self.save_dir_train)

                    cv2.imshow('Image with highlighted contours', img1)
                    print(str(a))
                    cv2.imwrite(self.save_dir_origin_cir +  str(a) + ".jpg", img_origin_cir)
                    cv2.imwrite(self.save_dir_seg_rect +  str(a) + ".jpg", img_seg_rectan)
                    cv2.imwrite(self.save_dir_seg_cir +  str(a) + ".jpg", img_seg_cir)



                    cv2.waitKey(10)
                    within_folder_i += 1  # this index is used to determine the train validation split


if __name__ == '__main__':

    if (Process_all_folder_flag == False): # just  the default foler will be convert
        converter = Read_read_check_json_label()
        converter.check_one_folder()  # convert json files into pkl files
    else:
        converter = Read_read_check_json_label()
        all_img_folder_list = os.listdir(converter.database_root + "img/")

        # create the buffer list
        for sub_folder in all_img_folder_list:
            this_sub = sub_folder + "/"
            converter = Read_read_check_json_label(this_sub) # initialized the converter every round

            # if(number_i==0):
            # converter.image_dir = converter.database_root + "img/" + this_sub
            # converter.json_dir = converter.database_root + "label/" + this_sub
            # converter.save_dir = converter.database_root + "seg label pkl/" + this_sub
            # converter.save_dir_train = converter.database_root + "seg label pkl train/" + this_sub
            # converter.save_dir_test = converter.database_root + "seg label pkl test/" + this_sub
            #
            # converter.excel_force_dir = converter.database_root + "excel_force_singnals/" + this_sub
            # converter.save_excel_dir = converter.database_root + "tactile_excel/" + this_sub
            # if not os.path.exists(converter.save_dir):
            #     os.makedirs(converter.save_dir)
            # if not os.path.exists(converter.save_dir_train):
            #     os.makedirs(converter.save_dir_train)
            # if not os.path.exists(converter.save_dir_test):
            #     os.makedirs(converter.save_dir_test)
            # if not os.path.exists(converter.save_excel_dir):
            #     os.makedirs(converter.save_excel_dir)
            # converter.img_num = 0
            converter.check_one_folder() # check this folder iteratively
            print( this_sub + "---is done!")


