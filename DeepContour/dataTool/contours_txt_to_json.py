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
import matplotlib.pyplot as plt


def encodeImageForJson(image):
    img_pil = PIL.Image.fromarray(image, mode='RGB')
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    enc_data = codecs.encode(data, 'base64').decode()
    enc_data = enc_data.replace('\n', '')

    return enc_data


def _cart2polar(x, y, kx, ky):
    r = [None] * len(x)
    theta = [None] * len(x)

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

    r = np.asarray(r)
    theta = np.asarray(theta)

    mag = kx * r
    angle = ky * theta

    return mag, angle


#### Settings
# root_dir = 'C:/Users/u0132260/OneDrive - KU Leuven/Data/ATLAS collaboration ESR5 ESR7/PVA phantom/branch2_continuous_1V'
root_dir = 'C:/Users/u0132260/Documents/Data/CRAS 2022/PVA phantom/branch2_continuous_2.5V'
contours_file = os.path.join(root_dir, 'contours_xy.csv')  # include full path
images_dir = os.path.join(root_dir, 'branch2_continuous_2.5V_images')  # include full path
output_dir = os.path.join(root_dir, 'labels_json')  # include full path

example_json = "../../config/example.json"  # Do not change
_px_to_mm = 0.0765306  # 12 x 5 mm / 784 px (IVUS)

_color_hex_catheter = '#%02x%02x%02x' % (230, 25, 75)  # BGR to HEX
_color_hex_lumen = '#%02x%02x%02x' % (60, 180, 75)  # BGR to HEX

#### Load lumen contours from txt file
with open(contours_file, "r") as file:
    data_contours = pd.read_csv(contours_file)

#### Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#### Load example of a json file to dump correctly
with open(example_json) as f:
    data_json = JSON.load(f)

contour_count = 0  # to go through the contours data (per columns)
fig, ax = plt.subplots(1, 1, constrained_layout=True)

#### For each image, save the catheter and lumen contour points in json file + image data
for img_id in os.listdir(images_dir):  # every label/image in the dir
    file, ext = os.path.splitext(img_id)

    img_color = cv2.imread(os.path.join(images_dir, img_id))
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    center = [h // 2, w // 2]

    contour_x = data_contours.values[:, contour_count] // _px_to_mm + center[0]  # in px
    contour_y = data_contours.values[:, contour_count + 1] // _px_to_mm + center[1]  # in px
    contour_x = contour_x[~np.isnan(contour_x)]
    contour_y = contour_y[~np.isnan(contour_y)]
    contour_count += 2

    ### Catheter label
    num_points = 35  # user-defined number of points on the circumference for the coordinates
    angle = np.linspace(0, 358, num=num_points)  # start in 1 to avoid having the 0 and 360 duplicates
    radius = 35  # px - user-defined
    catheter_coordinates = radius * np.cos(angle * np.pi / 180) + center[0]
    catheter_coordinates = np.c_[catheter_coordinates, radius * np.sin(angle * np.pi / 180) + center[1]]

    #### Saving to JSON
    data_json['shapes'].clear()  # remove all elements

    json_data_catheter = {
        'label': 'catheter',
        'points': catheter_coordinates.tolist(),
        'group_id': None,
        'shape_type': 'linestrip',
        'flags': {}
    }
    json_data_lumen = {
        'label': 'lumen',
        'points': np.column_stack((contour_x, contour_y)).tolist(),
        'group_id': None,
        'shape_type': 'linestrip',
        'flags': {}
    }
    data_json['shapes'].append(json_data_catheter)
    data_json['shapes'].append(json_data_lumen)

    data_json['imageHeight'] = h
    data_json['imageWidth'] = w
    data_json["imageData"] = encodeImageForJson(img_color)

    with open(os.path.join(output_dir, file + '.json'), "w") as jsonFile:
        JSON.dump(data_json, jsonFile)

    print('File {0} exported.'.format(file + '.json'))

    #### Visualization
    ax.imshow(img_color)
    ax.axis('off')

    ax.plot(contour_x, contour_y, color=_color_hex_lumen, marker='o', linewidth=1, markersize=5, label='lumen')

    ax.plot(catheter_coordinates[:, 0], catheter_coordinates[:, 1], color=_color_hex_catheter, marker='o', linewidth=1,
            markersize=5, label='catheter')

    plt.gcf().canvas.draw()
    plt.pause(0.1)
    ax.cla()
