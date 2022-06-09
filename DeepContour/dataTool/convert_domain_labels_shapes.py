# Script to convert labelme labels (in JSON files) from polar domain to cartesian domain and vice-versa
# for non-layer objects

import os
import warnings
import xml.etree.cElementTree as ET
from typing import List

import cv2
import json as JSON
import numpy as np
import codecs
import base64
import io
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import matplotlib.pyplot as plt
import pandas as pd
from dataTool.convert_domain_labels import labelsConverter
from working_dir_root import Dataset_root

# Flags
CONVERT_FLAG = True
DISPLAY_FLAG = True

def breakRegions(read_json_path, write_json_path):

    for img_id in os.listdir(read_json_path):
        # img_id = 'image0094.json'
        file, ext = os.path.splitext(img_id)

        if ext == '.json':
            json_file_path = os.path.join(read_json_path, img_id)
            with open(json_file_path) as f:
                json_data = JSON.load(f)

            shapes = json_data['shapes']
            num_labels = len(shapes)

            # image width that will be used to check if any target region is broken when it shouldn't
            w_idx = float(json_data['imageWidth'] - 1)

            #### Handle broken shapes with same label that should be one shape
            different_shapes_at_w = [x for x in range(num_labels) if w_idx in np.array(shapes[x]['points'])[:, 0]]
            different_shapes_at_0 = [x for x in range(num_labels) if 0.0 in np.array(shapes[x]['points'])[:, 0]]

            remove_shapes_idx = []
            joint_coordinates = []
            joint_labels = []

            # At image width
            if len(different_shapes_at_w) > 1:
                labels_at_w = [shapes[i]['label'] for i in different_shapes_at_w]
                counter_labels_at_w = [(x, labels_at_w.count(x)) for x in set(labels_at_w)]

                for i in range(len(counter_labels_at_w)):  # same label for the shapes at image width -> merge them
                    if counter_labels_at_w[i][1] > 1: # if multiple shapes, merge. If not, do nothing
                        # get indexes for the current label that is repeated
                        current_label_idx_at_w = [x for x in different_shapes_at_w
                                                  if counter_labels_at_w[i][0] == shapes[x]['label']]

                        # There will be a maximum of 2 contours per broken shape
                        current_max_w_idx = np.argmax([np.argmax(np.array(shapes[j]['points'])[:, 0])
                                                       for j in current_label_idx_at_w])
                        current_min_w_idx = np.argmin([np.argmax(np.array(shapes[j]['points'])[:, 0])
                                                       for j in current_label_idx_at_w])

                        current_joint_coordinates = np.append(np.array(shapes[current_label_idx_at_w[current_min_w_idx]]
                                                                       ['points']),
                                                              np.array(shapes[current_label_idx_at_w[current_max_w_idx]]
                                                                       ['points']),
                                                              axis=0)

                        remove_shapes_idx.extend(current_label_idx_at_w)  # add indexes to remove from shapes
                        joint_coordinates.append(current_joint_coordinates)  # add coordinates that will replace removed
                        # indexes coordinates
                        joint_labels.append(counter_labels_at_w[i][0])  # add label that will replace removed labels

            # At image 'start'
            if len(different_shapes_at_0) > 1:
                labels_at_0 = [shapes[i]['label'] for i in different_shapes_at_0]
                counter_labels_at_0 = [(x, labels_at_0.count(x)) for x in set(labels_at_0)]

                for i in range(len(counter_labels_at_0)):  # same label for the shapes at image 'start' -> merge them
                    if counter_labels_at_0[i][1] > 1: # if multiple shapes, merge. If not, do nothing
                        # get indexes for the current label that is repeated
                        current_label_idx_at_0 = [x for x in different_shapes_at_0
                                                  if counter_labels_at_0[i][0] == shapes[x]['label']]

                        # There will be a maximum of 2 contours per broken shape
                        current_max_0_idx = np.argmax([np.argmin(np.array(shapes[j]['points'])[:, 0])
                                                       for j in current_label_idx_at_0])
                        current_min_0_idx = np.argmin([np.argmin(np.array(shapes[j]['points'])[:, 0])
                                                       for j in current_label_idx_at_0])

                        current_joint_coordinates = np.append(np.array(shapes[current_label_idx_at_0[current_min_0_idx]]
                                                                       ['points']),
                                                              np.array(shapes[current_label_idx_at_0[current_max_0_idx]]
                                                                       ['points']),
                                                              axis=0)

                        remove_shapes_idx.extend(current_label_idx_at_0)  # add indexes to remove from shapes
                        joint_coordinates.append(current_joint_coordinates)  # add coordinates that will replace removed
                        # indexes coordinates
                        joint_labels.append(counter_labels_at_0[i][0])  # add label that will replace removed labels

            if remove_shapes_idx:  # list is not empty
                # sort indexes in descending order to remove merged shapes
                for index in sorted(remove_shapes_idx, reverse=True):
                    del shapes[index]

                # and then append the joint_coordinates to shapes using the example_shape
                for i in range(len(joint_coordinates)):
                    json_data_i = {
                        'label': joint_labels[i],
                        'points': joint_coordinates[i].tolist(),
                        'group_id': None,
                        'shape_type': 'linestrip',
                        'flags': {}
                    }
                    shapes.append(json_data_i)

                    # example_shape['points'] = joint_coordinates[i].tolist()
                    # example_shape['label'] = joint_labels[i]
                    #
                    # shapes.append(example_shape)

                # redefine number of labels in case shapes were merged
                num_labels = len(shapes)

            modified_shapes = []  # we will append an example shape for each label 2 times (for the upper and lower)

            for idx in range(num_labels):
                # use the 'simplified' disease labels and save them uniformly for the different json files
                current_label = [key for key, value in labels_lists.items() if shapes[idx]['label'] in value][0]

                if current_label in disease_labels:
                    label_coordinates = np.asarray(shapes[idx]['points'])

                    upper_label, lower_label, upper_coord, lower_coord = separateUpperLowerContour(current_label,
                                                                                                   label_coordinates)

                    # append upper contour
                    json_data_u = {
                        'label': upper_label,
                        'points': upper_coord.tolist(),
                        'group_id': None,
                        'shape_type': 'linestrip',
                        'flags': {}
                    }
                    modified_shapes.append(json_data_u)

                    # append lower contour
                    json_data_l = {
                        'label': lower_label,
                        'points': lower_coord.tolist(),
                        'group_id': None,
                        'shape_type': 'linestrip',
                        'flags': {}
                    }
                    modified_shapes.append(json_data_l)

            # export new labels with modified shapes
            json_data['shapes'] = modified_shapes
            json_file_out = os.path.join(write_json_path, file + '.json')
            with open(json_file_out, "w") as jsonFile:
                JSON.dump(json_data, jsonFile)

            print('File {0} exported.'.format(img_id))


def separateUpperLowerContour(label, coordinates):

    end_idx = len(coordinates)
    # find min and max x value in polar coordinates
    min_index = np.argmin(coordinates[:, 0])
    max_index = np.argmax(coordinates[:, 0])

    # if (min_index == 0 and max_index == end_idx - 1) or (min_index == 0 and max_index == end_idx - 1):
    #     a = 0  # case where the contour was separated for the conversion! need to handle this case FIRST
    #     # iterate through all the shapes with same label to find where it is open and close it!! DONE BEFORE

    if min_index > max_index:

        vec_end = coordinates[min_index:end_idx, :]

        # if max_index == 0:
        #     vec_start = coordinates[0:max_index, :]  # will be empty -> shape is close to the image boundaries
        # else:
        vec_start = coordinates[0:max_index + 1, :]

        sep_vec1 = np.append(vec_end, vec_start, axis=0)
        # if max_index == end_idx - 1:
        #     sep_vec2 = coordinates[max_index:min_index, :]
        # else:
        sep_vec2 = coordinates[max_index:min_index + 1, :]

    elif max_index > min_index:

        vec_end = coordinates[max_index:end_idx, :]

        # if min_index == 0:
        #     vec_start = coordinates[0:min_index, :]  # will be empty -> shape is close to the image boundaries
        # else:
        vec_start = coordinates[0:min_index + 1, :]

        sep_vec1 = np.append(vec_end, vec_start, axis=0)
        # if max_index == end_idx - 1:
        #     sep_vec2 = coordinates[min_index:max_index, :]
        # else:
        sep_vec2 = coordinates[min_index:max_index + 1, :]

    # find the mean y value of each vector to determine upper and lower contour
    if sep_vec2.mean(axis=0)[1] < sep_vec1.mean(axis=0)[1]:  # if true, sep_vec2 is the upper boundary and vice-versa
        upper_vec = sep_vec2
        lower_vec = sep_vec1
    else:
        upper_vec = sep_vec1
        lower_vec = sep_vec2

    upper_label = label + '_u'
    lower_label = label + '_l'

    return upper_label, lower_label, upper_vec, lower_vec


if __name__ == '__main__':
    global labels_lists
    global disease_labels

    if CONVERT_FLAG:
        converter = labelsConverter()

        # Read settings from config file:
        d = os.path.join('..', '..', 'config')
        configFile = os.path.join(d, 'settings_label_converter.xml')
        converter.loadXML(configFile)

        # convert to polar/cartesian domain and save as it would be done running convert_domain_labels.py
        converter.convertLabel()

        # set directory or image path to convert = out_path of the initial conversion
        if converter.params_convert['label_data']['target'] == 'dir':
            convert_labels_input_path = converter.params_convert['label_data']['out_dir_path']
        elif converter.params_convert['label_data']['target'] == 'singleimg':
            convert_labels_input_path = converter.params_convert['label_data']['out_one_img_dir_path']

    else:
        # set directory or image directory path to break shapes manually
        # convert_labels_input_path = Dataset_root + "label data/"
        convert_labels_input_path = 'C:/Users/u0132260/Documents/Data/ATLAS-0001/2021_10_20_case_7/' \
                                    'Labels_PDLDU4Z4_disease_rect/'

    # Possible labels that can be found in the JSON files
    labels_lists = {
        'catheter': ['1', 'catheter', 'test'],
        'lumen': ['2', 'vessel', 'lumen'],
        'wire': ['3', 'guide-wire', 'guidewire'],
        'media': ['4', 'vessel (media)', 'vessel(media)', 'media'],
        'branch': ['5', 'vessel(side-branch)', 'vessel (side-branch)', 'vessel(sidebranch)', 'vessel (sidebranch)',
                   'side-branch', 'sidebranch', 'bifurcation'],
        'stent': ['6', 'stent'],
        'plaque': ['7', 'plaque'],
        'calcium': ['8', 'calcification', 'calcium'],
    }

    disease_labels = ['plaque', 'calcium']

    # list possible names to consider to break contours in top and bottom -> IVUS disease related
    # shapes_labels_list = ['7', 'plaque', '8', 'calcification', 'calcium']

    # Output path = Input path + '_separated' -> at the same level of Input path
    dir_name = os.path.basename(os.path.normpath(convert_labels_input_path))
    dir_out_name = dir_name + '_separated'
    convert_labels_out_path = os.path.join(convert_labels_input_path, '..', dir_out_name)

    if not os.path.exists(convert_labels_out_path):
        os.makedirs(convert_labels_out_path)

    # break non-layer shapes
    print('Starting the merger of disease labels')
    breakRegions(convert_labels_input_path, convert_labels_out_path)
