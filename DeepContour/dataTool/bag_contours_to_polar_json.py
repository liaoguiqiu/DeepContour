import numpy as np
import pandas as pd
import os
import json as JSON
import cv2
import codecs
import io
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import base64
import scipy.signal as signal
import matplotlib.pyplot as plt


def encodeImageForJson(image):
    img_pil = PIL.Image.fromarray(image, mode='RGB')
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    enc_data = codecs.encode(data, 'base64').decode()
    enc_data = enc_data.replace('\n', '')

    return enc_data


def decodeImageFromJson(image_b64):
    image_data = base64.b64decode(image_b64)
    f = io.BytesIO()
    f.write(image_data)
    img_pil = PIL.Image.open(f)
    img_arr = np.asarray(img_pil)

    return img_arr


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
            transitions[idx] = 1  # indicate that there is a transition from the index flagged to the next
            # point on the vectors of contours

    r = np.asarray(r)
    theta = np.asarray(theta)

    mag = kx * r
    angle = ky * theta

    return mag, angle, transitions


#### Settings
root_dir = 'C:/Users/u0132260/Documents/Data/CRAS 2022/PVA phantom/branch2_continuous_1V'
contours_json_dir = os.path.join(root_dir, 'labels_json_nosignal')  # include full path
images_dir = os.path.join(root_dir, 'branch2_continuous_1V_images_rect')  # include full path
images_ext = 'png'
output_dir = os.path.join(root_dir, 'labels_json_nosignal_rect')  # include full path

example_json = "../../config/example.json"  # Do not change
_px_to_mm = 0.0765306  # 12 x 5 mm / 784 px (IVUS)

_color_hex_catheter = '#%02x%02x%02x' % (230, 25, 75)  # BGR to HEX
_color_hex_lumen = '#%02x%02x%02x' % (60, 180, 75)  # BGR to HEX

interpolate_flag = True
adjust_opening_flag = False

#### Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#### Load example of a json file to dump correctly
with open(example_json) as f:
    data_json = JSON.load(f)

for label_id in os.listdir(contours_json_dir):  # every label/image in the dir

    json_file_path = os.path.join(contours_json_dir, label_id)
    file, ext = os.path.splitext(label_id)
    with open(json_file_path) as f:
        data = JSON.load(f)

    labels = data['shapes']
    num_labels = len(labels)

    h_src = data['imageHeight']
    w_src = data['imageWidth']

    # Read imageData from JSON file
    img_json_data = data['imageData']
    img_json_color = decodeImageFromJson(img_json_data)
    if len(img_json_color.shape) < 3:  # for display
        img_json_color = cv2.cvtColor(img_json_color, cv2.COLOR_GRAY2RGB)

    # Read image for JSON file
    img_path = os.path.join(images_dir, file + '.' + images_ext)

    img_color = cv2.imread(img_path)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h_dst, w_dst = img.shape

    ### Conversion parameters
    min_len = float(np.minimum(h_src, w_src))
    # minimum value of src image height and width to use in scaling parameter
    max_radius = float(np.sqrt(((min_len / 2.0) ** 2.0) + ((min_len / 2.0) ** 2.0)))  # diagonal
    # scaling parameters in polar:
    kx = h_src / max_radius
    ky = w_src / 360

    for idx in range(num_labels):
        if labels[idx]['label'] == 'catheter':
            catheter_coordinates = np.array(labels[idx]['points'])
        elif labels[idx]['label'] == 'lumen':
            lumen_coordinates = np.array(labels[idx]['points'])

    # Convert catheter
    x_conv_cart_catheter, y_conv_cart_catheter, transitions_catheter = \
        _cart2polar(catheter_coordinates[:, 0] - w_src / 2.0, catheter_coordinates[:, 1] - h_src / 2.0, kx, ky)

    # Convert lumen
    x_conv_cart_lumen, y_conv_cart_lumen, transitions_lumen = _cart2polar(lumen_coordinates[:, 0] - w_src / 2.0,
                                                                          lumen_coordinates[:, 1] - h_src / 2.0, kx, ky)

    list_catheter_x = w_src - y_conv_cart_catheter
    list_catheter_y = x_conv_cart_catheter

    list_lumen_x = w_src - y_conv_cart_lumen
    list_lumen_y = x_conv_cart_lumen

    # contour_transitions = np.where(transitions_lumen == 1)[0]
    #
    # split_contour_x = np.split(np.array(list_lumen_x), contour_transitions)
    # split_contour_y = np.split(np.array(list_lumen_y), contour_transitions)

    list_catheter_x, list_catheter_y = zip(*sorted(zip(list_catheter_x.tolist(), list_catheter_y.tolist())))
    list_lumen_x, list_lumen_y = zip(*sorted(zip(list_lumen_x.tolist(), list_lumen_y.tolist())))

    ### Adjust opening (side-branches, lack of signal, ect)
    if adjust_opening_flag:
        array_catheter_x = np.array(list_catheter_x)
        array_catheter_y = np.array(list_catheter_y)
        array_lumen_x = np.array(list_lumen_x)
        array_lumen_y = np.array(list_lumen_y)

        # Catheter interpolation
        num_points_catheter = len(array_catheter_x)
        path_w_catheter = int(array_catheter_x[num_points_catheter - 1] - array_catheter_x[0])
        array_catheter_yl = np.ones(int(path_w_catheter)) * np.nan

        for j in range(num_points_catheter):
            # important sometimes the start point is not the lestmost
            this_index = np.clip(array_catheter_x[j] - array_catheter_x[0], 0, path_w_catheter - 1)
            array_catheter_yl[int(this_index)] = float(array_catheter_y[j])

        add_3_catheter = np.append(array_catheter_yl[::1], array_catheter_yl, axis=0)  # cascade
        add_3_catheter = np.append(add_3_catheter, array_catheter_yl[::1], axis=0)  # cascade
        s = pd.Series(add_3_catheter)
        array_catheter_yl = s.interpolate(method='linear')

        array_catheter_yl = array_catheter_yl[path_w_catheter:2 * path_w_catheter].to_numpy()
        array_catheter_xl = np.arange(int(array_catheter_x[0]), int(array_catheter_x[0]) + path_w_catheter)

        catheter_x = np.arange(0, w_dst)
        catheter_y = np.ones(int(w_dst)) * h_dst
        catheter_y[array_catheter_xl] = array_catheter_yl

        array_catheter_xl = catheter_x
        array_catheter_yl = catheter_y

        if len(array_catheter_xl) > 0.96 * w_dst and len(
                array_catheter_xl) != w_dst:  # correct the 'imperfect' label contours
            # remember to add resacle later TODO: what is this resacle? ask Guiqiu
            array_catheter_yl = signal.resample(array_catheter_yl, w_dst)
            array_catheter_xl = np.arange(0, w_dst)

        list_catheter_x = array_catheter_xl[0:len(array_catheter_xl):27].tolist()
        list_catheter_y = array_catheter_yl[0:len(array_catheter_yl):27].tolist()

        # Lumen interpolation
        num_points_lumen = len(array_lumen_x)
        path_w_lumen = int(array_lumen_x[num_points_lumen - 1] - array_lumen_x[0])
        array_lumen_yl = np.ones(int(path_w_lumen)) * np.nan

        for j in range(num_points_lumen):
            # important sometimes the start point is not the lestmost
            this_index = np.clip(array_lumen_x[j] - array_lumen_x[0], 0, path_w_lumen - 1)
            array_lumen_yl[int(this_index)] = float(array_lumen_y[j])

        add_3_lumen = np.append(array_lumen_yl[::1], array_lumen_yl, axis=0)  # cascade
        add_3_lumen = np.append(add_3_lumen, array_lumen_yl[::1],
                                axis=0)  # cascade # TODO - CHANGED AND TELL GUIQIU THAT IT IS NOT BACKWARDS!
        s = pd.Series(add_3_lumen)
        array_lumen_yl = s.interpolate(method='linear')

        # array_lumen_yl = array_lumen_yl[(path_w_lumen - int(array_lumen_x[0])):(path_w_lumen - int(array_lumen_x[0])) +
        # w_dst].to_numpy()
        array_lumen_yl = array_lumen_yl[path_w_lumen:2 * path_w_lumen].to_numpy()
        array_lumen_xl = np.arange(int(array_lumen_x[0]), int(array_lumen_x[0]) + path_w_lumen)

        lumen_x = np.arange(0, w_dst)
        lumen_y = np.ones(int(w_dst)) * h_dst
        lumen_y[array_lumen_xl] = array_lumen_yl

        array_lumen_xl = lumen_x
        array_lumen_yl = lumen_y

        if len(array_lumen_xl) > 0.96 * w_dst and len(
                array_lumen_xl) != w_dst:  # correct the 'imperfect' label contours
            # remember to add resacle later TODO: what is this resacle? ask Guiqiu
            array_lumen_yl = signal.resample(array_lumen_yl, w_dst)
            array_lumen_xl = np.arange(0, w_dst)

        list_lumen_x = array_lumen_xl[0:len(array_lumen_xl):27].tolist()
        list_lumen_y = array_lumen_yl[0:len(array_lumen_yl):27].tolist()
        # list_lumen_x = array_lumen_xl.tolist()
        # list_lumen_y = array_lumen_yl.tolist()

    ### Interpolate to all A-lines
    if interpolate_flag:
        array_catheter_x = np.array(list_catheter_x)
        array_catheter_y = np.array(list_catheter_y)
        array_lumen_x = np.array(list_lumen_x)
        array_lumen_y = np.array(list_lumen_y)

        # Catheter interpolation
        num_points_catheter = len(array_catheter_x)
        # path_w_catheter = int(array_catheter_x[num_points_catheter - 1] - array_catheter_x[0])
        path_w_catheter = int(w_dst)
        array_catheter_yl = np.ones(int(path_w_catheter)) * np.nan

        for j in range(num_points_catheter):
            # important sometimes the start point is not the lestmost
            # this_index = np.clip(array_catheter_x[j] - array_catheter_x[0], 0, path_w_catheter - 1)
            this_index = np.clip(array_catheter_x[j], 0, path_w_catheter - 1)
            array_catheter_yl[int(this_index)] = float(array_catheter_y[j])

        add_3_catheter = np.append(array_catheter_yl[::1], array_catheter_yl, axis=0)  # cascade
        add_3_catheter = np.append(add_3_catheter, array_catheter_yl[::1], axis=0)  # cascade
        s = pd.Series(add_3_catheter)
        array_catheter_yl = s.interpolate(method='linear')

        array_catheter_yl = array_catheter_yl[path_w_catheter:2 * path_w_catheter].to_numpy()
        array_catheter_xl = np.arange(int(array_catheter_x[0]), int(array_catheter_x[0]) + path_w_catheter)

        if len(array_catheter_xl) > 0.96 * w_dst and len(
                array_catheter_xl) != w_dst:  # correct the 'imperfect' label contours
            # remember to add resacle later TODO: what is this resacle? ask Guiqiu
            array_catheter_yl = signal.resample(array_catheter_yl, w_dst)
            array_catheter_xl = np.arange(0, w_dst)

        list_catheter_x = array_catheter_xl[0:len(array_catheter_xl):27].tolist()
        list_catheter_y = array_catheter_yl[0:len(array_catheter_yl):27].tolist()

        # Lumen interpolation
        num_points_lumen = len(array_lumen_x)
        # path_w_lumen = int(array_lumen_x[num_points_lumen - 1] - array_lumen_x[0])
        path_w_lumen = int(w_dst)
        array_lumen_yl = np.ones(int(path_w_lumen)) * np.nan

        for j in range(num_points_lumen):
            # important sometimes the start point is not the lestmost
            # this_index = np.clip(array_lumen_x[j] - array_lumen_x[0], 0, path_w_lumen - 1)
            this_index = np.clip(array_lumen_x[j], 0, path_w_lumen - 1)
            array_lumen_yl[int(this_index)] = float(array_lumen_y[j])

        add_3_lumen = np.append(array_lumen_yl[::1], array_lumen_yl, axis=0)  # cascade
        add_3_lumen = np.append(add_3_lumen, array_lumen_yl[::1], axis=0)  # cascade # TODO - CHANGED AND TELL GUIQIU THAT IT IS NOT BACKWARDS!
        s = pd.Series(add_3_lumen)
        array_lumen_yl = s.interpolate(method='linear')

        # array_lumen_yl = array_lumen_yl[(path_w_lumen - int(array_lumen_x[0])):(path_w_lumen - int(array_lumen_x[0])) +
                                                                               # w_dst].to_numpy()
        array_lumen_yl = array_lumen_yl[path_w_lumen:2 * path_w_lumen].to_numpy()
        # array_lumen_xl = np.arange(int(array_lumen_x[0]), int(array_lumen_x[0]) + path_w_lumen)
        array_lumen_xl = np.arange(0, path_w_lumen)

        if len(array_lumen_xl) > 0.96 * w_dst and len(
                array_lumen_xl) != w_dst:  # correct the 'imperfect' label contours
            # remember to add resacle later TODO: what is this resacle? ask Guiqiu
            array_lumen_yl = signal.resample(array_lumen_yl, w_dst)
            array_lumen_xl = np.arange(0, w_dst)

        list_lumen_x = array_lumen_xl[0:len(array_lumen_xl):27].tolist()
        list_lumen_y = array_lumen_yl[0:len(array_lumen_yl):27].tolist()
        # list_lumen_x = array_lumen_xl.tolist()
        # list_lumen_y = array_lumen_yl.tolist()

    #### Saving to JSON
    data_json['shapes'].clear()  # remove all elements

    json_data_catheter = {
        'label': 'catheter',
        'points': np.column_stack((np.array(list_catheter_x), np.array(list_catheter_y))).tolist(),
        'group_id': None,
        'shape_type': 'linestrip',
        'flags': {}
    }
    json_data_lumen = {
        'label': 'lumen',
        'points': np.column_stack((np.array(list_lumen_x), np.array(list_lumen_y))).tolist(),
        'group_id': None,
        'shape_type': 'linestrip',
        'flags': {}
    }
    data_json['shapes'].append(json_data_catheter)
    data_json['shapes'].append(json_data_lumen)

    data_json['imageHeight'] = h_dst
    data_json['imageWidth'] = w_dst
    data_json["imageData"] = encodeImageForJson(img_color)

    with open(os.path.join(output_dir, file + '.json'), "w") as jsonFile:
        JSON.dump(data_json, jsonFile)
