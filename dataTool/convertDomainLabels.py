# Script to convert labelme labels (in JSON files) from polar domain to cartesian domain and vice-versa
# from bs4 import BeautifulSoup
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


class labelsConverter(object):
    """ Label conversion settings
        """

    def __init__(self):
        self.params_convert = {}  # To save parameters set in the config file

        # BGR because of OpenCV
        self.color_list = [[75, 25, 230], [75, 180, 60], [25, 225, 255], [200, 130, 0], [48, 130, 245],
                           [180, 30, 145], [240, 240, 70], [230, 50, 240], [60, 245, 210], [212, 190, 250],
                           [128, 128, 0], [255, 190, 220], [40, 110, 170], [200, 250, 255], [0, 0, 128],
                           [195, 255, 170], [0, 128, 128], [180, 215, 255], [128, 0, 0]]

        # self.disease_labels = ['plaque', 'calcium', 'calcification']  # maybe necessary to add case when ivus
        # # is not inside the main lumen but in the vessel wall (CTO recanalization technique)

    def loadXML(self, _fn):
        """ Load settings from .xml files
        :param _fn: filename of the file to load
        """

        _, ext = os.path.splitext(_fn)

        if ext != '.xml':
            raise NotImplementedError('Format {0} not implemented: only .xml config files are supported.'.format(ext))

        else:

            tree = ET.parse(_fn)
            root = tree.getroot()

            #### Node LabelData
            node_data = root.find('LabelData')
            node_src_dir = node_data.find('SourceDir')
            node_out_dir = node_data.find('OutputDir')
            node_src_img = node_data.find('SourceImg')
            node_out_img = node_data.find('OutputImg')

            self.params_convert['label_data'] = {
                'conv_type': node_data.attrib['conversionType'].lower(),
                'target': node_data.attrib['target'].lower(),
                'src_dir_path': node_src_dir.text,
                'out_dir_path': node_out_dir.text,
                'src_one_img_path': node_src_img.text
            }
            same_dir = node_out_img.attrib['sameDir'].lower()

            if self.params_convert['label_data']['target'] == 'singleimg':
                dir_one_img, file = os.path.split(self.params_convert['label_data']['src_one_img_path'])
                self.params_convert['label_data']['src_one_img_file_name'] = file
                if same_dir == 'true':
                    self.params_convert['label_data']['out_one_img_dir_path'] = \
                        os.path.join(dir_one_img, self.params_convert['label_data']['conv_type'])
                elif same_dir == 'false':
                    self.params_convert['label_data']['out_one_img_dir_path'] = node_out_img.find('fileDirPath').text
                else:
                    warnings.warn(
                        'Option "{0}" not implemented: only "true" or "false" are supported in sameDir attribute.'
                        ' \n Saving in same directory (sameDir = "true")'.format(same_dir))

                    self.params_convert['label_data']['out_one_img_dir_path'] = \
                        os.path.join(dir_one_img, self.params_convert['label_data']['conv_type'])

                if not os.path.exists(self.params_convert['label_data']['out_one_img_dir_path']):
                    os.makedirs(self.params_convert['label_data']['out_one_img_dir_path'])

            else:
                if not os.path.exists(self.params_convert['label_data']['out_dir_path']):
                    os.makedirs(self.params_convert['label_data']['out_dir_path'])

            #### Node ImageData
            node_image_data = root.find('ImageData')

            if self.params_convert['label_data']['target'] == 'dir':
                self.params_convert['image_data'] = {
                    'ext': node_image_data.find('ImagesDir').find('Ext').text,
                    'path': node_image_data.find('ImagesDir').find('Path').text
                }
            else:
                self.params_convert['image_data'] = {
                    'file_path': node_image_data.find('SingleImg').find('filePath').text
                }

            #### Node Display
            node_display = root.find('Display')
            self.params_convert['display'] = {}
            self.params_convert['display']['state'] = node_display.attrib['state'].lower()

    def convertLabel(self):
        """ Convert labels from JSON files (Labelling software: Label Me), depending on the source (dir or img)
        """
        if self.params_convert['label_data']['target'] == 'dir':
            self.convertLabelDir()
        elif self.params_convert['label_data']['target'] == 'singleimg':
            self.convertLabelImg()
        else:
            raise NotImplementedError('Format {0} not implemented: only Dir and SingleImg targets are supported.'
                                      .format(self.params_convert['target']))

    def convertLabelDir(self):
        """ Convert (polar/cartesian) entire directory with image labels in JSON files (Labelling software: Label Me)
        """

        if self.params_convert['display']['state'] == 'true':
            fig, axs = plt.subplots(1, 2, constrained_layout=True)
            fig.suptitle('(Directory) Label conversion to ' + self.params_convert['label_data']['conv_type'],
                         fontsize=24, fontweight="bold")
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()

        for img_id in os.listdir(self.params_convert['label_data']['src_dir_path']):  # every label/image in the dir
            file, ext = os.path.splitext(img_id)

            if ext == '.json':

                json_file_path = os.path.join(self.params_convert['label_data']['src_dir_path'], img_id)
                with open(json_file_path) as f:
                    data = JSON.load(f)

                labels = data['shapes']
                num_labels = len(labels)

                h_src = data['imageHeight']
                w_src = data['imageWidth']

                # Read imageData from JSON file
                img_json_data = data['imageData']
                img_json_color = self.decodeImageFromJson(img_json_data)
                if len(img_json_color.shape) < 3:  # for display
                    img_json_color = cv2.cvtColor(img_json_color, cv2.COLOR_GRAY2RGB)
                # img_json = cv2.cvtColor(img_json_color, cv2.COLOR_BGR2GRAY)

                # Read image for JSON file
                img_path = os.path.join(self.params_convert['image_data']['path'], file + '.' +
                                        self.params_convert['image_data']['ext'])

                img_color = cv2.imread(img_path)
                img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

                if img is None:
                    raise FileNotFoundError('File {0} not found in dir: {1}'
                                            .format(file + '.' +
                                                    self.params_convert['image_data']['ext'],
                                                    self.params_convert['image_data']['path']))

                h_dst, w_dst = img.shape

                contours_x, contours_y, label_coordinates_json_x, label_coordinates_json_y, adjusted_transitions = \
                    self.convertLabelPoints(labels, num_labels, h_src, w_src, h_dst, w_dst)

                separated_contours_x, separated_contours_y, separated_num_labels, separated_contours_labels = \
                    self._separateContours(labels, contours_x, contours_y, num_labels, adjusted_transitions)

                if self.params_convert['display']['state'] == 'true':
                    # fig, axs = plt.subplots(1, 2, constrained_layout=True)
                    # fig.suptitle('(Directory) Label conversion to ' + self.params_convert['label_data']['conv_type'],
                    #              fontsize=24, fontweight="bold")
                    # manager = plt.get_current_fig_manager()
                    # manager.window.showMaximized()
                    axs[0].imshow(img_json_color)
                    axs[0].axis('off')
                    axs[0].set_title('Source data', fontsize=20)
                    axs[1].imshow(img_color)
                    axs[1].axis('off')
                    axs[1].set_title('Converted data', fontsize=20)

                    labels_list = []

                    for idx in range(num_labels):
                        if labels[idx]['label'] not in labels_list:
                            labels_list.append(labels[idx]['label'])

                        label_idx = labels_list.index(labels[idx]['label'])
                        color_hex = '#%02x%02x%02x' % (self.color_list[label_idx][2], self.color_list[label_idx][1],
                                                       self.color_list[label_idx][0])  # BGR to HEX

                        axs[0].plot(label_coordinates_json_x[idx], label_coordinates_json_y[idx], color=color_hex,
                                    marker='o', linewidth=1, markersize=5)

                    handles, labels = axs[0].get_legend_handles_labels()
                    legend = pd.DataFrame(np.column_stack([handles, labels]),
                                          columns=['line', 'label']).drop_duplicates(
                        subset='label')

                    handles = legend.iloc[:, 0].tolist()
                    labels = legend.iloc[:, 1].tolist()

                    axs[0].legend(handles, labels, loc='lower right', ncol=len(labels_list), mode="expand",
                                  borderaxespad=0.)

                    for idx in range(separated_num_labels):
                        label_idx = labels_list.index(separated_contours_labels[idx])
                        color_hex = '#%02x%02x%02x' % (self.color_list[label_idx][2], self.color_list[label_idx][1],
                                                       self.color_list[label_idx][0])  # BGR to HEX

                        axs[1].plot(separated_contours_x[idx], separated_contours_y[idx], color=color_hex, marker='o',
                                    linewidth=1, markersize=5)

                    plt.gcf().canvas.draw()
                    plt.pause(0.1)
                    axs[0].cla()
                    axs[1].cla()

                json_file_out = os.path.join(self.params_convert['label_data']['out_dir_path'], file + '.json')
                self._exportNewLabel(data, img_color, separated_contours_x, separated_contours_y,
                                     separated_contours_labels, json_file_out)

        print('All labels converted to "{0}". Directory: {1}'.format(self.params_convert['label_data']['conv_type'],
                                                                     self.params_convert['label_data']['src_dir_path']))

    def convertLabelImg(self):
        """ Convert single image label from a JSON file (Labelling software: Label Me)
        """

        _, ext = os.path.splitext(self.params_convert['label_data']['src_one_img_file_name'])

        if ext == '.json':

            with open(self.params_convert['label_data']['src_one_img_path']) as f:
                data = JSON.load(f)

            labels = data['shapes']
            num_labels = len(labels)

            h_src = data['imageHeight']
            w_src = data['imageWidth']

            # Read imageData from JSON file
            img_json_data = data['imageData']
            img_json_color = self.decodeImageFromJson(img_json_data)
            if len(img_json_color.shape) < 3:  # for display
                img_json_color = cv2.cvtColor(img_json_color, cv2.COLOR_GRAY2RGB)
            # img_json = cv2.cvtColor(img_json_color, cv2.COLOR_BGR2GRAY)

            # Read image for JSON file
            img_color = cv2.imread(self.params_convert['image_data']['file_path'])
            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            if img is None:
                raise FileNotFoundError('File {0} not found.'
                                        .format(self.params_convert['image_data']['file_path']))

            h_dst, w_dst = img.shape

            contours_x, contours_y, label_coordinates_json_x, label_coordinates_json_y, adjusted_transitions = \
                self.convertLabelPoints(labels, num_labels, h_src, w_src, h_dst, w_dst)

            separated_contours_x, separated_contours_y, separated_num_labels, separated_contours_labels = \
                self._separateContours(labels, contours_x, contours_y, num_labels, adjusted_transitions)

            if self.params_convert['display']['state'] == 'true':
                fig, axs = plt.subplots(1, 2, constrained_layout=True)
                fig.suptitle('(Single Image) Label conversion to ' + self.params_convert['label_data']['conv_type'],
                             fontsize=24, fontweight="bold")
                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()
                axs[0].imshow(img_json_color)
                axs[0].axis('off')
                axs[0].set_title('Source data', fontsize=20)
                axs[1].imshow(img_color)
                axs[1].axis('off')
                axs[1].set_title('Converted data', fontsize=20)

                labels_list = []

                for idx in range(num_labels):
                    if labels[idx]['label'] not in labels_list:
                        labels_list.append(labels[idx]['label'])

                    label_idx = labels_list.index(labels[idx]['label'])
                    color_hex = '#%02x%02x%02x' % (self.color_list[label_idx][2], self.color_list[label_idx][1],
                                                   self.color_list[label_idx][0])  # BGR to HEX

                    axs[0].plot(label_coordinates_json_x[idx], label_coordinates_json_y[idx], color=color_hex,
                                marker='o', linewidth=1, markersize=5, label=labels[idx]['label'])

                handles, labels = axs[0].get_legend_handles_labels()
                legend = pd.DataFrame(np.column_stack([handles, labels]), columns=['line', 'label']).drop_duplicates(
                    subset='label')

                handles = legend.iloc[:, 0].tolist()
                labels = legend.iloc[:, 1].tolist()

                axs[0].legend(handles, labels, loc='lower right', ncol=len(labels_list), mode="expand",
                              borderaxespad=0.)

                for idx in range(separated_num_labels):
                    label_idx = labels_list.index(separated_contours_labels[idx])
                    color_hex = '#%02x%02x%02x' % (self.color_list[label_idx][2], self.color_list[label_idx][1],
                                                   self.color_list[label_idx][0])  # BGR to HEX

                    axs[1].plot(separated_contours_x[idx], separated_contours_y[idx], color=color_hex, marker='o',
                                linewidth=1, markersize=5)

            json_file_out = os.path.join(self.params_convert['label_data']['out_one_img_dir_path'],
                                         self.params_convert['label_data']['src_one_img_file_name'])
            self._exportNewLabel(data, img_color, separated_contours_x, separated_contours_y,
                                 separated_contours_labels, json_file_out)

            plt.show()  # Hold window

        else:
            raise ValueError('The label file must be .json.')

        print('Labels converted to "{0}". Single image label file: {1}'.format(
            self.params_convert['label_data']['conv_type'],
            self.params_convert['label_data']['src_one_img_path']))

    def convertLabelPoints(self, labels, num_labels, h_src, w_src, h_dst, w_dst):

        contours_x = [None] * num_labels
        contours_y = [None] * num_labels
        label_coordinates_json_x = [None] * num_labels
        label_coordinates_json_y = [None] * num_labels
        adjusted_transitions = [None] * num_labels

        for i in range(num_labels):
            # extract coordinates for each label
            label_coordinates = np.asarray(labels[i]['points'])

            if labels[i]['shape_type'] == 'circle':
                num_points = 35  # user-defined number of points on the circumference for the coordinates
                angle = np.linspace(0, 358, num=num_points)  # start in 1 to avoid having the 0 and 360 duplicates
                center = label_coordinates[0, :]  # first row of coordinates (X,Y) is by default the center (in labelme)
                radius = np.sqrt(
                    (label_coordinates[1, 0] - center[0]) ** 2 + (label_coordinates[1, 1] - center[1]) ** 2)

                label_coordinates = radius * np.cos(angle * np.pi / 180) + center[0]
                label_coordinates = np.c_[label_coordinates, radius * np.sin(angle * np.pi / 180) + center[1]]

            # delete the coordinates outside the boundaries of the image
            len_ori, _ = label_coordinates.shape
            target = 0
            for j in range(len_ori):
                # check the coordinates outside the image boundaries
                if label_coordinates[target, 0] < 0 or label_coordinates[target, 0] > (w_src - 1) or \
                        label_coordinates[target, 1] < 0 or label_coordinates[target, 1] > (h_src - 1):
                    label_coordinates = np.delete(label_coordinates, target, axis=0)
                else:
                    target += 1

            # delete duplicated points (very likely to happen when doing manual segmentation)
            label_coordinates = pd.DataFrame(label_coordinates).drop_duplicates(keep='first').values

            ### Conversion parameters
            min_len = float(np.minimum(h_src, w_src))
            # minimum value of src image height and width to use in scaling parameter
            max_radius = float(np.sqrt(((min_len / 2.0) ** 2.0) + ((min_len / 2.0) ** 2.0)))  # diagonal
            # scaling parameters in polar:
            kx = h_src / max_radius
            ky = w_src / 360

            label_coordinates_json_x[i] = label_coordinates[:, 0]
            label_coordinates_json_y[i] = label_coordinates[:, 1]

            if self.params_convert['label_data']['conv_type'] == 'cartesian':  # convert to cartesian

                x_conv_cart, y_conv_cart = self.polar2cart(label_coordinates[:, 1],
                                                           w_src - label_coordinates[:, 0],
                                                           (w_dst / 2, h_dst / 2), kx, ky)

                contours_x[i] = x_conv_cart
                contours_y[i] = y_conv_cart

            elif self.params_convert['label_data']['conv_type'] == 'polar':  # convert to polar

                x_conv_cart, y_conv_cart, transitions = self._cart2polar(label_coordinates[:, 0] - w_src / 2.0,
                                                                         label_coordinates[:, 1] - h_src / 2.0, kx, ky)

                # rotate because images are rotated to show horizontally
                contours_x[i] = w_src - y_conv_cart
                contours_y[i] = x_conv_cart
                adjusted_transitions[i] = transitions

                # If transitions are found (contour crosses x=0 when y>0 in circular),
                # connect with border and sort points
                if 1 in transitions:
                    contours_xy = np.c_[contours_x[i], contours_y[i]]
                    # Assumes horizontal polar image:
                    contours_x[i], contours_y[i], adjusted_transitions[i] = self._adjustPointsOnEdges(transitions,
                                                                                                      contours_xy,
                                                                                                      w_dst)

            else:
                raise NotImplementedError(
                    'Conversion type "{0}" not implemented: only "Polar" and "Cartesian" types are supported.'
                    .format(self.params_convert['label_data']['conv_type']))

        return contours_x, contours_y, label_coordinates_json_x, label_coordinates_json_y, adjusted_transitions

    @staticmethod
    def _adjustPointsOnEdges(_transitions, contours_xy, _w_dst):

        num_transitions = len(np.transpose(np.where(_transitions == 1)))
        adjusted_transitions = np.zeros(_transitions.shape[0] + num_transitions * 2)
        counter_new_points = 0
        contours_xy_adjusted = contours_xy.tolist()

        for i, idx_transition in enumerate(np.transpose(np.where(_transitions == 1))):
            p_closest_0 = contours_xy[np.where(
                contours_xy[:, 0] == min(contours_xy[idx_transition[0], 0], contours_xy[idx_transition[0] + 1, 0]))[0][
                                          0], :]
            p_closest_360 = contours_xy[np.where(
                contours_xy[:, 0] == max(contours_xy[idx_transition[0], 0], contours_xy[idx_transition[0] + 1, 0]))[0][
                                            0], :]

            m_at_0 = (p_closest_360[1] - p_closest_0[1]) / ((p_closest_360[0] - _w_dst) - p_closest_0[0])
            b_at_0 = p_closest_0[1] - m_at_0 * p_closest_0[0]
            # Y coordinate is the same for the points at the two edges
            p_0 = [0, b_at_0]
            p_360 = [_w_dst - 1, b_at_0]

            # close to 360 ---> close to 0
            if contours_xy[idx_transition[0], 0] - contours_xy[idx_transition[0] + 1, 0] > 0:
                contours_xy_adjusted.insert(idx_transition[0] + 1 + counter_new_points, p_360)
                contours_xy_adjusted.insert(idx_transition[0] + 2 + counter_new_points, p_0)

            # close to 0 ---> close to 360
            elif contours_xy[idx_transition[0], 0] - contours_xy[idx_transition[0] + 1, 0] <= 0:
                contours_xy_adjusted.insert(idx_transition[0] + 1 + counter_new_points, p_0)
                contours_xy_adjusted.insert(idx_transition[0] + 2 + counter_new_points, p_360)

            adjusted_transitions[idx_transition[0] + counter_new_points + 2] = 1
            counter_new_points += 2

        return np.asarray(contours_xy_adjusted)[:, 0], np.asarray(contours_xy_adjusted)[:, 1], adjusted_transitions
        # x, y, indexes of adjusted contours transitions

    @staticmethod
    def _separateContours(labels, contours_x, contours_y, num_labels, adjusted_transitions):

        separated_contours_x = []
        separated_contours_y = []
        separated_labels = []

        for i_label in range(num_labels):
            contour_transitions = np.where(adjusted_transitions[i_label] == 1)[0]

            split_contour_x = np.split(contours_x[i_label], contour_transitions)
            split_contour_y = np.split(contours_y[i_label], contour_transitions)

            separated_contours_x += split_contour_x
            separated_contours_y += split_contour_y

            for i_trans in range(len(contour_transitions) + 1):
                separated_labels.append(labels[i_label]['label'])

        num_separated_labels = len(separated_labels)

        return separated_contours_x, separated_contours_y, num_separated_labels, separated_labels

    @staticmethod
    def polar2cart(mag, angle, center, kx, ky):
        theta = angle / ky
        r = mag / kx
        x = r * np.cos(theta * np.pi / 180) + center[0]
        y = r * np.sin(theta * np.pi / 180) + center[1]

        return x, y

    @staticmethod
    def _cart2polar(x, y, kx, ky):

        r = [None] * len(x)
        theta = [None] * len(x)
        transitions = np.zeros(len(x))

        for idx, (value_x, value_y) in enumerate(zip(x, y)):
            r_i = np.sqrt(value_x ** 2 + value_y ** 2)
            theta_i = np.arctan(value_y / value_x) * (180 / np.pi)

            if -1e-8 <= value_x <= 1e-8:
                if -1e-8 <= value_y <= 1e-8:
                    theta_i = 0
                elif value_y > -1e-8:
                    theta_i = 90
                elif value_y < 1e-8:
                    theta_i = 270
            elif value_x > -1e-8 and value_y < 1e-8:
                theta_i = theta_i + 360
            elif value_x < 1e-8:
                theta_i = theta_i + 180

            r[idx] = r_i
            theta[idx] = theta_i

            # check if points suffer a transition of more than 270 deg (before scaling with kx and ky)
            if idx != 0 and abs(theta_i - theta[idx - 1]) >= 270:
                transitions[idx - 1] = 1  # indicate that there is a transition from the index flagged to the next
                # point on the vectors of contours

        r = np.asarray(r)
        theta = np.asarray(theta)

        mag = kx * r
        angle = ky * theta

        return mag, angle, transitions

    @staticmethod
    def encodeImageForJson(image):
        img_pil = PIL.Image.fromarray(image, mode='RGB')
        f = io.BytesIO()
        img_pil.save(f, format='PNG')
        data = f.getvalue()
        enc_data = codecs.encode(data, 'base64').decode()
        enc_data = enc_data.replace('\n', '')

        return enc_data

    @staticmethod
    def decodeImageFromJson(image_b64):
        image_data = base64.b64decode(image_b64)
        f = io.BytesIO()
        f.write(image_data)
        img_pil = PIL.Image.open(f)
        img_arr = np.asarray(img_pil)

        return img_arr

    @staticmethod
    def _exportNewLabel(_json_data, _img, _contours_x, _contours_y, _labels, _json_file_out):

        _, json_file = os.path.split(_json_file_out)
        num_labels = len(_contours_x)
        h, w, _ = _img.shape
        _json_data['shapes'].clear()  # remove all elements

        for i in range(num_labels):
            json_data_i = {
                'label': _labels[i],
                'points': np.column_stack((_contours_x[i], _contours_y[i])).tolist(),
                'group_id': None,
                'shape_type': 'linestrip',
                'flags': {}
            }
            _json_data['shapes'].append(json_data_i)

        _json_data['imageHeight'] = h
        _json_data['imageWidth'] = w
        _json_data["imageData"] = labelsConverter.encodeImageForJson(_img)

        with open(_json_file_out, "w") as jsonFile:
            JSON.dump(_json_data, jsonFile)

        print('File {0} exported.'.format(json_file))


if __name__ == '__main__':
    converter = labelsConverter()

    # Read settings from config file:
    d = os.path.join('..', '..', 'config')
    configFile = os.path.join(d, 'settings_label_converter.xml')
    converter.loadXML(configFile)
    converter.convertLabel()
