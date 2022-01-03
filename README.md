# DeepContour
This package provides an implementation of the training/inference/autoannotation pipeline of AlphaFold
v2.0. This is a model that can be either deploiyed with or without other feature extraction 
backbones (i.e. resent/vgg/unet/segnet..). 
. For simplicity the bottom model after any backbone(top model) as Coordinates Encoding Networks(CEnet)
## First time setup

The following steps are required in order to run CEnet:

1.  Install [Docker](https://www.docker.com/).
    *   Install
        [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
        for GPU support.
    *   Setup running
        [Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).
1.  
1.  Download model parameters (see below).
1.  Check that AlphaFold will be able to use a GPU by running:

Coordinates encoding for the lumen segmentation of OCT and IVUS


In data toolfolder:


read_json.py: transfer one folder of separeted json file into one compact pkl


generator_contour_sheath.py : generated ramdom image with folder of image and the corresponding pkl; it also embeds some scripts to check the stastics of the distribution of the labeled contour 


tSNE.py: which is used to see the data distribution with the reduced dimention (current file does not consider about the label)




In deploy folder (this is normally used after the nets are trained): 

most of these scripts are used to predict contour, specifically:
 
DeepAutoJson.py is used to generate separated json files from the prediction result of the network.

And it also has a function for downsampling the dataset for trianing and testing.
