import os
import shutil

json_dir = 'C:/Users/u0132260/Documents/Data/ATLAS-0001/2021_02_02 case 2/Labels_PD8P8EQH_rect_lumen_catheter/'
img_dir = 'C:/Users/u0132260/Documents/Data/ATLAS-0001/2021_02_02 case 2/TIFF_PD8P8EQH_rect/'

# create temporary folder to save selected images if these don't exist already. Otherwise, clean the folder
selected_img_save_dir = os.path.join(img_dir, '..', 'temporary')
if not os.path.exists(selected_img_save_dir):
    os.makedirs(selected_img_save_dir)
else:  # delete all the files if they exist in the tmp directory
    for file in os.listdir(selected_img_save_dir):
        os.remove(os.path.join(selected_img_save_dir, file))

for label_id in os.listdir(json_dir):
    # separate the name of json
    file, ext = os.path.splitext(label_id)

    if ext == '.json':
        selected_img_input_path = os.path.join(img_dir, file + '.tif')
        selected_img_save_path = os.path.join(selected_img_save_dir, file + '.tif')
        shutil.copy(selected_img_input_path, selected_img_save_path)



