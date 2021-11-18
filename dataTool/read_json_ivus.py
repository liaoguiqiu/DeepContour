# Log of modification cre
# this  is used  to read json files and transfer into a pkl file
# Guiqiu modify this for spliting the data for train validation
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
from generator_contour_ivus import Save_Contour_pkl
Train_validation_split = True # flag for devide the data 
Train_validation_devi = 3 # all data are equally devided by thsi number
Test_fold = 0   # use the 0 st for training, the other for validation 
Delete_outsider_flag =False
class Read_read_check_json_label(object):
    def __init__(self):
        # self.image_dir   = "../../OCT/beam_scanning/Data set/pic/NORMAL-BACKSIDE-center/"
        # self.roi_dir =  "../../OCT/beam_scanning/Data set/seg label/NORMAL-BACKSIDE-center/"
        # self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL/"
        # self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE-center/"
        # self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE/"
        self.database_root = "../../dataset/ivus/"
        self.database_root = "D:/Deep learning/dataset/label data/"

        sub_folder = "phantom1/"

        self.image_dir = self.database_root + "img/" + sub_folder
        self.json_dir = self.database_root + "label/" + sub_folder
        self.save_dir = self.database_root + "seg label pkl/" + sub_folder
        self.save_dir_train = self.database_root + "seg label pkl train/" + sub_folder
        self.save_dir_test = self.database_root + "seg label pkl test/" + sub_folder


        self.img_num = 0

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.save_dir_train):
            os.makedirs(self.save_dir_train)
        if not os.path.exists(self.save_dir_test):
            os.makedirs(self.save_dir_test)

        # self.contours_x = []  # no predefines # predefine there are 4 contours
        # self.contours_y = []  # predefine there are 4 contours

        self.saver = Save_Contour_pkl()
        self.saver_train = Save_Contour_pkl()
        self.saver_test = Save_Contour_pkl()


        self.display_flag = True

        self.labels_lists = {
            'catheter': ['1', 'catheter', 'test'],
            'lumen': ['2', 'vessel', 'lumen'],
            'wire': ['guide-wire', 'guidewire'],
            'media': ['vessel (media)', 'vessel(media)', 'media'],
            'branch': ['vessel(side-branch)', 'vessel (side-branch)', 'vessel(sidebranch)', 'vessel (sidebranch)',
                       'side-branch', 'sidebranch', 'bifurcation'],
            'stent': ['stent'],
            'plaque': ['plaque'],
            'calcium': ['calcification', 'calcium'],
        }

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
            dx = vx[j]

            # _img[int(dy) + 1, int(dx), :] = _img[int(dy), int(dx), :] = painter
            _img[int(dy) + 1, dx, :] = _img[int(dy) - 1, dx, :] = _img[int(dy), dx, :] = painter

        return _img

    def check_one_folder(self):
        # check the image type:
        imagelist = os.listdir(self.image_dir)
        _,b_i  = os.path.splitext(imagelist[0]) # first image of this folder 
        within_folder_i = 0 
        for i in os.listdir(self.json_dir):
            # for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
            # separate the name of json
            a, b = os.path.splitext(i)
            # if it is a json it will have corresponding image 
            if b == ".json":
                # with ZipFile(self.image_dir, 'r') as zipObj:
                #     listOfFiles = zipObj.namelist()
                # TODO: Extract image ext automatically
                #img_path = self.image_dir + a + ".tif"
                img_path = self.image_dir + a + b_i 
                img1 = cv2.imread(img_path)
                if img1 is None:
                    print('No img with path: {0}'.format(img_path))
                else:
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
                    contours_x = OrderedDict({key: np.arange(0, W) for key in self.labels_lists.keys()})

                    # contours_y is initialized to the image height for all A-lines
                    contours_y = OrderedDict({key: np.ones(W) * H - 1 for key in self.labels_lists.keys()})

                    # Initialize existence vectors per label = zeros to all A-lines
                    contours_exist = OrderedDict({key: np.zeros(W, dtype=int) for key in self.labels_lists.keys()})

                    for idx in range(num_labels):
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

                            if 1 in contours_exist[current_label]:  # multiple vectors with same label
                                # necessary to check if there are overlapping contours
                                existing_contour_x = np.where(contours_exist[current_label] == 1)[0]
                                overlap_x = np.intersect1d(existing_contour_x, path_xl)  # overlapped x coordinates

                                # if there is an overlap --> keep contour points closer to scanning center
                                if len(overlap_x) != 0:
                                    new_y_overlapped = path_yl[overlap_x - path_xl[0]]
                                    existing_y_overlapped = contours_y[current_label][overlap_x]
                                    closer_to_center_y_overlapped = np.minimum(new_y_overlapped, existing_y_overlapped)

                                    # "safe" to add y coordinates from this vector to label with existing contour with
                                    # the same label
                                    contours_y[current_label][path_xl] = path_yl

                                    # overlap was determined before and the corresponding x coordinates can be adjusted
                                    contours_y[current_label][overlap_x] = closer_to_center_y_overlapped

                                    # change existence vector A-line for non-overlapping y coordinates
                                    contours_exist[current_label][path_xl] = 1  # overlapping A-lines are already 1
                                else:
                                    # Add 1 to the A-lines where there is contour as there is no overlap
                                    contours_exist[current_label][path_xl] = 1
                                    contours_y[current_label][path_xl] = path_yl

                            else:
                                # Add 1 to the A-lines where there is contour
                                contours_exist[current_label][path_xl] = 1
                                # contours_x[current_label] = path_xl
                                contours_y[current_label][path_xl] = path_yl

                        pass

                    if self.display_flag:  # for loop for display out of previous loop in case of overlap of contours
                        for label, exist_v in contours_exist.items():
                            if 1 in exist_v:
                                index = list(contours_exist.keys()).index(label)
                                img1 = self.draw_coordinates_color(img1, contours_x[label][
                                    np.where(exist_v == 1)[0]], contours_y[label][np.where(exist_v == 1)[0]], index)

                    # save this result
                    self.img_num = a  # TODO: why assigning a to another variable?

                    # TODO: check if append_new_name_contour is used anywhere else other than in this script
                    self.saver.append_new_name_contour(self.img_num, contours_x, contours_y, contours_exist,
                                                       self.save_dir)

                    if Train_validation_split == True: 
                       if within_folder_i%Train_validation_devi == Test_fold:
                           self.saver_test.append_new_name_contour(self.img_num, contours_x, contours_y, contours_exist,
                                                       self.save_dir_test)
                       else:
                           self.saver_train.append_new_name_contour(self.img_num, contours_x, contours_y, contours_exist,
                                                       self.save_dir_train)


                    cv2.imshow('Image with highlighted contours', img1)
                    print(str(a))
                    cv2.waitKey(10)
                    within_folder_i +=1 # this index is used to determine the train validation split 

if __name__ == '__main__':

    converter = Read_read_check_json_label()
    converter.check_one_folder()  # convert json files into pkl files
     
