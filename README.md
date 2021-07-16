# DeepContour

Coordinates encoding for the lumen segmentation of OCT and IVUS


In data toolfolder (most of these scripts are merged from the segmentation project : https://github.com/liaoguiqiu/OCT_segmentation)


read_json.py: transfer one folder of separeted json file into one compact pkl


generator_contour_sheath.py : generated ramdom image with folder of image and the corresponding pkl; it also embeds some scripts to check the stastics of the distribution of the labeled contour 


tSNE.py: which is used to see the data distribution with the reduced dimention (current this file does not consider about the label)



