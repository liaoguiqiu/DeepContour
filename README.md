# DeepContour

Coordinates encoding for the lumen segmentation of OCT and IVUS


In data toolfolder:


read_json.py: transfer one folder of separeted json file into one compact pkl


generator_contour_sheath.py : generated ramdom image with folder of image and the corresponding pkl; it also embeds some scripts to check the stastics of the distribution of the labeled contour 


tSNE.py: which is used to see the data distribution with the reduced dimention (current file does not consider about the label)




In deploy folder (this is normally used after the nets are trained): 

most of these scripts are used to predict contour, specifically:
 
DeepAutoJson.py is used to generate separated json files from the prediction result of the network.

And it also has a function for downsampling the dataset for trianing and testing.
