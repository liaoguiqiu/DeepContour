# 7th October 2020
# update the gnerator to add the situation with sheath
import cv2
import numpy as np
import os
import random
from matplotlib.pyplot import *
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
# PythonETpackage for xml file edition
import pickle
from dataTool.operater import Basic_Operator
from dataTool.operator2 import Basic_Operator2
import scipy.signal as signal
from working_dir_root import Dataset_root
import time


# the generator and distribution monitoring

# this is used  to communicate with trainner py
class Communicate(object):
    def __init__(self):
        # set = Read_read_check_ROI_label()
        # self.database_root = set.database_root
        # check or create this path
        # self.self_check_path_create(self.signal_data_path)
        self.training = 1
        self.writing = 2
        self.pending = 1

    def change_state(self):
        if self.writing == 1:
            self.writing = 0
        pass

    def read_data(self, dir):
        saved_path = dir + 'protocol.pkl'
        try:
            self = pickle.load(open(saved_path, 'rb'), encoding='iso-8859-1')
        except:
            print("this is the first time of the generator")
            self.training = 1  # only use for the first ytime
            self.writing = 2  # only use for the first ytime
            self.pending = 0  # only use for the first ytime
            self.save_data(dir)
        return self

    def save_data(self, dir):
        # save the data
        save_path = dir + 'protocol.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        pass


# from ImgJ_ROI2 import Read_read_check_ROI_label
# for this function this is to save the laeyrers
class Save_Contour_pkl(object):
    def __init__(self):
        # set = Read_read_check_ROI_label()
        # self.database_root = set.database_root
        # check or create this path
        # self.self_check_path_create(self.signal_data_path)
        self.img_num = []
        self.contoursx = []
        self.contoursy = []
        self.contours_exist = []

    def self_check_path_create(self, directory):
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
            # add new step of all signals

    def append_new_name_contour(self, number, this_contours_x, this_contours_y, this_contours_exist, dir):
        # buffer
        self.img_num.append(number)
        self.contoursx.append(this_contours_x)
        self.contoursy.append(this_contours_y)
        self.contours_exist.append(this_contours_exist)
        # TODO: check if this function is used anywhere else other than read_json. Yes! inside the generator!

        # save the data
        save_path = dir

        with open(save_path + 'contours.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        pass

    # read from file
    def read_data(self, base_root):
        saved_path = base_root
        self = pickle.load(open(saved_path + 'contours.pkl', 'rb'), encoding='iso-8859-1')
        return self


class Generator_Contour_sheath(object):
    def __init__(self):
        self.OLG_flag = False
        self.cv_display = True
        self.dis_origin = True

        # BGR because of OpenCV
        self.color_list = [[75, 25, 230], [75, 180, 60], [25, 225, 255], [200, 130, 0], [48, 130, 245],
                           [180, 30, 145], [240, 240, 70], [230, 50, 240], [60, 245, 210], [212, 190, 250],
                           [128, 128, 0], [255, 190, 220], [40, 110, 170], [200, 250, 255], [0, 0, 128],
                           [195, 255, 170], [0, 128, 128], [180, 215, 255], [128, 0, 0]]

        self.disease_labels = ['plaque', 'calcium']

        self.origin_data = Save_Contour_pkl()
        # self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/VARY/"
        # data_root = "../../dataset/ivus/"
        data_root = Dataset_root + "label data/"
        self.image_dir = data_root + "img/"
        # self.pkl_dir = data_root +"seg label pkl train/"  #TODO: change this to actually use the flag!!!!
        self.pkl_dir = data_root + "seg label pkl/"
        # self.pkl_dir = data_root +"seg label pkl train/"

        # for normal generator
        self.save_image_dir = data_root + "img_generate/"  # this dir just save all together
        self.save_image_dir_devi = data_root + "img_genetate_devi/"  # this dir devide the generated images
        self.save_pkl_dir = data_root + "pkl_generate/"

        # for OLG on line generator
        self.com_dir = data_root + "telecom/"
        self.OLG_dir = Dataset_root + "For IVUS/train_OLG/"
        self.contour_saver = Save_Contour_pkl()
        # self.origin_data =self.origin_data.read_data(self.pkl_dir)
        self.image_type = ".jpg"
        if not os.path.exists(self.save_image_dir):
            os.makedirs(self.save_image_dir)

        if not os.path.exists(self.save_image_dir_devi):
            os.makedirs(self.save_image_dir_devi)

        if not os.path.exists(self.save_pkl_dir):
            os.makedirs(self.save_pkl_dir)

        self.back_ground_root = "../../" + "saved_background_for_generator/"

        # self.save_img_dir = "../../"     + "saved_generated_contour/"
        # self.save_contour_dir = "../../"     + "saved_stastics_coutour_generated/"
        self.display_flag = True
        self.img_num = []
        self.contoursx = []
        self.contoursy = []

        # get all the folder
        self.all_dir_list = os.listdir(self.pkl_dir)
        self.folder_num = len(self.all_dir_list)
        # create the buffer list
        self.folder_list = [None] * self.folder_num
        self.signal = [None] * self.folder_num

        # create a detail foldeer list to save the generated images

        for subfold in self.all_dir_list:
            save_sub = self.save_image_dir_devi + subfold + '/'
            if not os.path.exists(save_sub):
                os.makedirs(save_sub)
        # create all  the folder list and their data list

        # number_i = 0
        ## all_dir_list is subfolder list 
        ##creat the image list point to the STASTICS TIS  list
        ##saved_stastics = Generator_Contour()
        ##read all the folder list
        # for subfold in self.all_dir_list:
        #    #if(number_i==0):
        #    this_folder_list =  os.listdir(os.path.join(self.pkl_dir, subfold))
        #    this_folder_list2 = [ self.pkl_dir +subfold + "/" + pointer for pointer in this_folder_list]
        #    self.folder_list[number_i] = this_folder_list2

        #    #change the dir firstly before read
        #    #saved_stastics.all_statics_dir = os.path.join(self.signalroot, subfold, 'contour.pkl')
        #    this_contour_dir =  self.pkl_dir+ subfold+'/'+ 'contours.pkl' # for both linux and window

        #    self.signal[number_i]  =  self.read_data(this_contour_dir)
        #    number_i +=1
        # read the folder list finished  get the folder list and all saved path

        # check or create this path

    # def append_new_name_contour(self, number, this_contoursx, this_contoursy, dir):
    #    # buffer
    #    self.img_num.append(number)
    #    self.contoursx.append(this_contoursx)
    #    self.contoursy.append(this_contoursy)

    #    # save the data
    #    save_path = dir  # + "seg label pkl/"
    #    with open(save_path + 'contours.pkl', 'wb') as f:
    #        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    #    pass

    def check(self):
        saved_path = self.save_pkl_dir
        data = pickle.load(open(saved_path + 'contours.pkl', 'rb'), encoding='iso-8859-1')
        file_len = len(data.img_num)
        for num in range(file_len):
            name = data.img_num[num]
            img_path = self.save_image_dir + str(name) + self.image_type
            img_or = cv2.imread(img_path)
            img1 = cv2.cvtColor(img_or, cv2.COLOR_BGR2GRAY)
            H, W = img1.shape
            # just use the first contour
            contour0x = data.contoursx[num]
            contour0y = data.contoursy[num]
            # draw this original contour
            display = Basic_Operator.draw_coordinates_color(img_or, contour0x, contour0y, 1)
            cv2.imshow('origin', display.astype(np.uint8))
            cv2.waitKey(10)

        pass

    # display the the contour  gray
    def display_contour(self, img, contourx, contoury, title):
        if self.display_flag == True:
            display = img
            contour0 = contour
            for j in range(len(contoury)):
                # path0l[path0x[j]]
                display[int(contoury[j]) + 1, contourx[j]] = display[int(contoury[j]) - 1, contourx[j]] = display[
                    int(contoury[j]), contourx[j]] = 254
            cv2.imshow(title, display.astype(np.uint8))
            cv2.waitKey(10)
        pass

    # to calculate the statitic distrubution of the dast
    def stastics(self):
        # for num  in  self.origin_data.img_num:
        img_id = 1
        distance_ori = []
        distance_new = []
        contact_r_ori = []  # the ratio of the contact
        contact_r_new = []  # the ratio of the contact
        # other metrics
        # the tissue range or the length (with a ratio)
        # the blank area ratio 

        # number_i = 0
        for subfold in self.all_dir_list:

            # saved_stastics.all_statics_dir = os.path.join(self.signalroot, subfold, 'contour.pkl')
            this_contour_dir = self.pkl_dir + subfold + '/'  # for both linux and window

            self.origin_data = self.origin_data.read_data(this_contour_dir)  # this original data
            # number_i +=1
            file_len = len(self.origin_data.img_num)

            for num in range(file_len):
                name = self.origin_data.img_num[num]
                img_path = self.image_dir + subfold + '/' + name + self.image_type
                img_or = cv2.imread(img_path)
                img1 = cv2.cvtColor(img_or, cv2.COLOR_BGR2GRAY)
                H, W = img1.shape
                # just use the first contour
                # contour0x  = self.origin_data.contoursx[num][0]
                # contour0y  = self.origin_data.contoursy[num][0]
                contourx = self.origin_data.contoursx[num]
                contoury = self.origin_data.contoursy[num]

                # fill in the background area with H value when contour is not fully labeld ,
                contourx[0], contoury[0] = Basic_Operator2.re_fresh_path(contourx[0], contoury[0], H, W)
                contourx[1], contoury[1] = Basic_Operator2.re_fresh_path(contourx[1], contoury[1], H, W)

                # uniform the initial path 
                this_distance = (contoury[1] - contoury[0]) / H
                distance_ori.append(this_distance)

                # the original contact 
                cont_points = sum(this_distance < 0.006)
                contact_r_ori.append(cont_points / W)

                if self.dis_origin == True:
                    # draw this original contour
                    display = Basic_Operator.draw_coordinates_color(img_or, contourx[0], contoury[0],
                                                                    1)  # draw the sheath
                    display = Basic_Operator.draw_coordinates_color(img_or, contourx[1], contoury[1],
                                                                    2)  # draw the tissue
                    cv2.imshow('origin', display.astype(np.uint8))

                # new_contourx=contour0x +200
                # new_contoury=contour0y-200

                H_new = H
                W_new = W
                # genrate the new sheath contour
                sheath_x, sheath_y = Basic_Operator2.random_sheath_contour(H_new, W_new, contourx[0], contoury[0])

                # generate the signal
                dc1 = np.random.random_sample() * 100
                dc1 = int(dc1) % 2
                if dc1 != 0:
                    new_contourx, new_contoury = Basic_Operator2.random_shape_contour3(H, W, H_new, W_new, sheath_x,
                                                                                       sheath_y, contourx[1],
                                                                                       contoury[1])
                else:
                    new_contourx, new_contoury = Basic_Operator2.random_shape_contour3(H, W, H_new, W_new, sheath_x,
                                                                                       sheath_y, contourx[1],
                                                                                       contoury[1])

                # fill in the blank area 

                # aligh and cut the signal
                # left= max(sheath_x[0],new_contourx[0])
                # right=min(sheath_x[len(sheath_x)-1],  new_contourx[len(new_contourx)-1])
                # fill in the background area with H value when contour is not fully labeld , 
                sheath_x, sheath_y = Basic_Operator2.re_fresh_path(sheath_x, sheath_y, H_new, W_new)
                new_contourx, new_contoury = Basic_Operator2.re_fresh_path(new_contourx, new_contoury, H_new, W_new)
                new_distance = (new_contoury - sheath_y) / H_new
                distance_new.append(new_distance)
                # the contact raction new 
                # the original contact 
                cont_points_n = sum(new_distance < 0.006)
                contact_r_new.append(cont_points_n / W_new)

                # min_b  = int(np.max(contoury[0]))
                # max_b  = int(np.min(contoury[1]))

                # new_cx  = [None]*2
                # new_cy   = [None]*2
                # new_cx[0]  = sheath_x
                # new_cy[0]  = sheath_y
                # new_cx[1]  = new_contourx
                # new_cy[1]  = new_contoury

                print(str(name))
                # self.append_new_name_contour(img_id,new_cx,new_cy,self.save_pkl_dir)
                # save them altogether 
                # save them separetly 
                # if not os.path.exists(directory):
                # os.makedirs(directory)

                img_id += 1

                pass

            # time.sleep(1)

            # sns.set_style('darkgrid')
            # dis = np.concatenate(distance_ori).flat
            # plt.figure()

            # sns.distplot(dis)
            # time.sleep(0.1)
        # d =  np.random.sample(1000)*10
        # ---------------------------------
        # draw the distibution of the distance 
        time.sleep(0.1)
        sns.set_style('darkgrid')
        dis = np.concatenate(distance_ori).flat
        plt.figure()
        sns.distplot(dis)
        time.sleep(0.1)

        time.sleep(0.1)

        sns.set_style('darkgrid')
        plt.figure()

        dis = np.concatenate(distance_new).flat
        clear_list = [x for x in dis if (x > 0 and x <= 0.75)]

        sns.distplot(clear_list)

        plt.figure()
        sns.distplot(dis)
        time.sleep(0.1)
        # -------------------
        # draw somthing about the contact
        # draw the distibution of the distance 
        time.sleep(0.1)
        sns.set_style('darkgrid')
        # dis = np.concatenate(contact_r_ori).flat # original contact map
        dis = contact_r_ori  # original contact map

        plt.figure()
        sns.distplot(dis)
        time.sleep(0.1)

        time.sleep(0.1)
        sns.set_style('darkgrid')
        # dis = np.concatenate(contact_r_new).flat # original contact map
        dis = contact_r_new
        plt.figure()
        sns.distplot(dis)
        time.sleep(0.1)

    def generate(self):
        # for num  in  self.origin_data.img_num:
        img_id = 1

        for subfold in self.all_dir_list:

            imagelist = os.listdir(self.image_dir + subfold + '/')
            _, image_type = os.path.splitext(imagelist[0])  # first image of this folder
            self.image_type = image_type  # TODO: understand/ask Guiqiu. why making it self. ??
            # change the dir before reading
            this_contour_dir = self.pkl_dir + subfold + '/'  # for both linux and window

            self.origin_data = self.origin_data.read_data(this_contour_dir)  # this original data label - pkl format
            file_len = len(self.origin_data.img_num)

            # TODO: move update of generated images to somewhere easy to find
            repeat = int(200 / file_len)  # repeat to balance
            if repeat < 1:
                repeat = 1

            img_id_devi = 1  # this image id is for one category/subfolder

            for n in range(repeat):  # repeat this to all data

                time_start = time.process_time()

                for num in range(file_len):
                    name = self.origin_data.img_num[num]
                    img_path = self.image_dir + subfold + '/' + name + image_type

                    img_or = cv2.imread(img_path)
                    img1 = cv2.cvtColor(img_or, cv2.COLOR_BGR2GRAY)
                    H, W = img1.shape

                    # get all values from OrderedDict to np.array - at this point, existence is not considered
                    # possible (ordered) labels:
                    # ['catheter', 'lumen', 'wire', 'media', 'branch', 'stent', 'plaque', 'calcium']
                    contourx = np.array(list(self.origin_data.contoursx[num].values()))
                    contoury = np.array(list(self.origin_data.contoursy[num].values()))
                    existence = np.array(list(self.origin_data.contours_exist[num].values()))

                    # draw the original contours, if they exist
                    # do not consider disease labels - always placed at the end of the labels dict (origin_data)
                    # Update list in init
                    num_labels_consider = len(existence[:, 0]) - len(self.disease_labels)
                    for idx_color in range(num_labels_consider):
                        if 1 in existence[idx_color]:
                            display = Basic_Operator.draw_coordinates_color_multi(img_or, contourx[idx_color],
                                                                                  contoury[idx_color],
                                                                                  self.color_list[idx_color])

                    if self.dis_origin:
                        cv2.imshow('origin', display.astype(np.uint8))

                    H_new = H  # TODO: why is this H_new and W_new being used? we are not going to change the image size - NOT NECESSARY
                    W_new = W

                    gen_contourx = [None] * num_labels_consider
                    gen_contoury = [None] * num_labels_consider
                    gen_existence = [None] * num_labels_consider

                    # determine existing labels/contours indexes
                    index_exist_contours = [i for i, e in enumerate(existence) if 1 in existence[i]]

                    # Check if catheter contour exists in original image. If yes, generate a new contour
                    if 0 in index_exist_contours:  # label: catheter (Always the first, i.e. always index 0)
                        # generate the new/distorted sheath (aka catheter) contour
                        gen_contourx[0], gen_contoury[0] = Basic_Operator2.random_sheath_contour_ivus(H, W,
                                                                                                      contourx[0],
                                                                                                      contoury[0])
                        # Update existence vector
                        gen_existence[0] = gen_contoury[0] < (0.98 * H)

                        # create new image and mask with generated sheath/catheter contour
                        new_img, mask = Basic_Operator2.fill_sheath_with_contour_ivus(img1, H, W, contourx[0],
                                                                                      contoury[0], gen_contourx[0],
                                                                                      gen_contoury[0])

                        if self.cv_display:
                            cv2.imshow('sheath/catheter generated contour', new_img.astype(np.uint8))

                    # MORE CONTOURS ARE PROCESSED BEFORE THE GUIDE-WIRE BECAUSE IN CASE THESE ARE BOUNDING THE WIRE(S),
                    # THE GENERATED CONTOURS MUST BE USED TO CLIP THE SYNTHETIC SHIFT
                    # if more contours exist (i.e. vector not empty) other than catheter and/or wire, generate new ones
                    more_contours = [x for x in index_exist_contours if x not in [0, 2]]
                    if more_contours:
                        # Assumption: catheter is always the top contour for all contours if it exists
                        # (except possibly for the wire, which is handled separately)
                        if 0 not in index_exist_contours:
                            gen_top_contoury = np.zeros(W)
                            top_contoury = np.zeros(W)
                        else:  # the generated catheter bounds the multiple contours below in case of a shift > 0
                            gen_top_contoury = gen_contoury[0]
                            top_contoury = contoury[0]

                        #  If we lower the new contour and the shift for the lumen goes above this but below the
                        #  original, then we will cross contours. So, to clip the distorted/generated contours,
                        #  the new catheter contour must be used. Thus, here gen_top_contoury is used
                        #  and not top_contoury. top_contoury is used to fill the image in fill_patch_base_multi(.)
                        new_contourx, new_contoury = \
                            Basic_Operator2.random_shape_contour_ivus_multi(H, gen_top_contoury,
                                                                            contourx[more_contours],
                                                                            contoury[more_contours])

                        for i in range(len(more_contours)):
                            gen_contoury[more_contours[i]] = new_contoury[i]
                            gen_contourx[more_contours[i]] = new_contourx[i]
                            gen_existence[more_contours[i]] = gen_contoury[more_contours[i]] < (0.98 * H)

                        # get index of the top contour per Aline
                        if len(more_contours) == 1:  # only one contour needs to be considered per Aline
                            top_more_contoury = new_contoury
                            top_more_contourx = new_contourx
                            base_contoury = contoury[more_contours]
                            base_contourx = contourx[more_contours]
                        else:
                            # top_more_contours_idx = [more_contours[x] for x in new_contoury.argmin(axis=0)]
                            top_more_contoury = new_contoury.min(axis=0)
                            top_more_contourx = new_contourx[0]  # all Alines defined for all contours
                            base_contoury = np.array([contoury[more_contours[val], i] for i, val in
                                                      enumerate(new_contoury.argmin(axis=0))])
                            base_contourx = np.array([contourx[more_contours[val], i] for i, val in
                                                      enumerate(new_contoury.argmin(axis=0))])

                        # so the generated contour is randomly "sharp" or "soft" edged
                        dice = int(np.random.random_sample() * 10)
                        if dice % 2 == 0:  # sharp edge
                            sharp = True
                        else:
                            sharp = False

                        new_img, mask = Basic_Operator2.fill_patch_base_multi(img1, H, sharp, top_contoury,
                                                                              base_contourx, base_contoury,
                                                                              top_more_contourx, top_more_contoury,
                                                                              new_img, mask)

                        # get background mask from relevant original contours
                        # to extract the background contour, we need the original
                        back_mask = Basic_Operator2.set_background_mask(H, top_contoury, base_contoury)

                    else:  # otherwise, define relevant original contours to get the background mask
                        # Top as catheter if exists and bottom as image height

                        back_base_contoury = np.ones(H) * (H - 1)
                        if 0 not in index_exist_contours:
                            back_top_contoury = np.zeros(W)
                        else:
                            back_top_contoury = contoury[0]

                        back_mask = Basic_Operator2.set_background_mask(H, back_top_contoury, back_base_contoury)

                    if self.cv_display:
                        cv2.imshow('mask', mask.astype(np.uint8))
                        cv2.imshow('all generated contours, except wire', new_img.astype(np.uint8))

                    # MORE CONTOURS ARE PROCESSED BEFORE THE GUIDE-WIRE BECAUSE IN CASE THESE ARE BOUNDING THE WIRE(S),
                    # THE GENERATED CONTOURS MUST BE USED TO CLIP THE SYNTHETIC SHIFT
                    # After the catheter, check if guide-wire exists and if it is bounded by any other contour
                    # (because it is below it) than the catheter/sheath.
                    # This can happen if e.g. multiple guide-wires are present
                    if 2 in index_exist_contours:  # label: wire (Always the third, i.e. always index 2)
                        # determine which contour is bounding the wire contour:
                        # --- Alines with wire (existence[2] == 1)
                        lines_exist_wire = np.where(existence[2])[0]

                        # determine if there are multiple wires (apart from each other) in the original image.
                        # --- if yes, set flag to generate contours to 'True' to compute different distortions
                        # --- 'False' by default in operator2.py in random_wire_contour_ivus()
                        index_multiple = [i + 1 for i, e in enumerate(lines_exist_wire[1:])
                                          if e != lines_exist_wire[i] + 1]

                        # if list is not empty (empty lists are seen as false), we have multiple wires
                        if index_multiple:
                            multiple_wire = np.split(lines_exist_wire, index_multiple)
                        else:
                            multiple_wire = [lines_exist_wire]

                        # save index for all contours present in the Alines with wire that are above it (check y values)
                        same_x_top_contours = [i for i, e in enumerate(existence[:, lines_exist_wire]) if 1 in e
                                               and i != 2 and any(np.array(contoury[i, lines_exist_wire]
                                                                           <= contoury[2, lines_exist_wire]))]

                        # in case there is no top contour, just the image boundaries are considered
                        top_wire_contour_index = []

                        # possibly top bounded by another contour that is not the catheter and not empty
                        # multiple wires can have different contours on top
                        if same_x_top_contours:
                            if same_x_top_contours != [0]:

                                top_wire_contour_index = [None] * len(multiple_wire)
                                # for each wire, compute the contour closer to the wire to use as top bound
                                for i in range(len(multiple_wire)):  # TODO: what if index is more than one per i??
                                    top_wire_contour_index[i] = int(np.unique(np.argmax(
                                        contoury[np.ix_(same_x_top_contours, multiple_wire[i])] *
                                        existence[np.ix_(same_x_top_contours, multiple_wire[i])], axis=0)))

                            elif same_x_top_contours == [0]:  # if catheter exists, top contour is the catheter
                                top_wire_contour_index = [0]

                        # generate the new/distorted wire contour clipped by the distorted contour
                        gen_contourx[2], gen_contoury[2] = \
                            Basic_Operator2.random_wire_contour_ivus(H, contourx[2], contoury[2],
                                                                     np.vstack(np.array(
                                                                         gen_contoury)[top_wire_contour_index]),
                                                                     multiple_wire)

                        # Update existence vector
                        gen_existence[2] = gen_contoury[2] < (0.98 * H)

                        # create new image and mask with generated wire contour
                        new_img, mask = Basic_Operator2.fill_wire_ivus(img1, H, lines_exist_wire, contoury[2],
                                                                       gen_contoury[2], new_img, mask)

                        if self.cv_display:
                            cv2.imshow('Added wire generated contour', new_img.astype(np.uint8))

                    # TODO: why are we refreshing path if the path has to come in the generator already for the whole
                    #  image width? - REMOVE - DONE
                    # contourx[0], contoury[0] = Basic_Operator2.re_fresh_path(contourx[0], contoury[0], H, W)
                    #
                    # contourx[1], contoury[1] = Basic_Operator2.re_fresh_path(contourx[1], contoury[1], H, W)

                    # generate background randomly from a patch of original background(typically blood speckle for IVUS)
                    back_img = Basic_Operator2.generate_ivus_blood_background(img1, back_mask, H, W)

                    if self.cv_display:
                        cv2.imshow('background', back_img.astype(np.uint8))

                    combi_img = Basic_Operator2.add_original_back(new_img, back_img, mask)

                    RGB_img = Basic_Operator.gray2rgb(combi_img)

                    # fill vectors of 'unsed' contours with original values (y = image height and existence = 0)
                    for x in range(num_labels_consider):
                        if x not in index_exist_contours:
                            gen_contoury[x] = contoury[x]
                            gen_contourx[x] = contourx[x]
                            gen_existence[x] = gen_contoury[x] < (0.98 * H)

                    for idx_color in range(num_labels_consider):
                        if any(gen_existence[idx_color]):
                            display = Basic_Operator.draw_coordinates_color_multi(RGB_img, gen_contourx[idx_color],
                                                                                  gen_contoury[idx_color],
                                                                                  self.color_list[idx_color])

                    if self.dis_origin:
                        cv2.imshow('final', display.astype(np.uint8))

                    cv2.waitKey(10)
                    # new_cx = [None] * 2  # initialized as just two boundaries
                    # new_cy = [None] * 2
                    # new_ex = [None] * 2  # also have the existence
                    #
                    # new_cx[0] = sheath_x
                    # new_cy[0] = sheath_y
                    # new_cx[1] = new_contourx
                    # new_cy[1] = new_contoury
                    # new_ex[0] = sheath_y < (0.98 * H_new)
                    # new_ex[1] = new_contoury < (0.98 * H_new)

                    time_elapsed = (time.process_time() - time_start)  # seconds
                    print(str(name))
                    print('Time elapsed: {0} sec'.format(time_elapsed))

                    # TODO: check if append contour function can handle more than 2
                    # TODO: the order of the contours matters? or as long as it is always the same order it is okay?
                    self.contour_saver.append_new_name_contour(img_id, gen_contourx, gen_contoury, gen_existence,
                                                               self.save_pkl_dir)  # replace with existence

                    # save them altogether 
                    cv2.imwrite(self.save_image_dir + str(img_id) + image_type, combi_img)
                    # save them separately
                    cv2.imwrite(self.save_image_dir_devi + subfold + '/' + str(img_id_devi) + image_type, combi_img)

                    img_id += 1
                    img_id_devi += 1

                    pass


if __name__ == '__main__':

    generator = Generator_Contour_sheath()

    if generator.OLG_flag:
        talker = Communicate()
        com_dir = generator.com_dir

        talker = talker.read_data(com_dir)

        imgbase_dir = generator.OLG_dir + "img/"
        labelbase_dir = generator.OLG_dir + "label/"

        while 1:
            generator = Generator_Contour_sheath()

            talker = talker.read_data(com_dir)  # TODO: why do we run this in the while and before?
            # only use for the first time

            if talker.training == 1 and talker.writing == 2:  # check if 2 need writing
                if talker.pending == 0:
                    generator.save_image_dir = imgbase_dir + "2/"
                    generator.save_pkl_dir = labelbase_dir + "2/"

                    generator.generate()  # generate

                    talker.writing = 1
                    talker.pending = 1
                    talker.save_data(com_dir)
            # set break point here for first time
            if talker.training == 2 and talker.writing == 1:  # check if 2 need writing
                if talker.pending == 0:
                    generator.save_image_dir = imgbase_dir + "1/"
                    generator.save_pkl_dir = labelbase_dir + "1/"

                    generator.generate()  # generate

                    talker.writing = 2
                    talker.pending = 1
                    talker.save_data(com_dir)
            cv2.waitKey(1000)
            print("Waiting")

    # this is the identical function of the generator that warp the shape of the tissue of the sheath
    # which will only keep the contour generator(not generate the image actually ), this file is just 
    # run before the generate so that the distribution of original label
    # and the generate label can be visualized
    # generator.stastics()
    generator.generate()

    generator.check()

    # back = generator.generate_background_image1(3,1024,1024)
    # cv2.imshow('origin',back.astype(np.uint8))
    cv2.waitKey(10)
