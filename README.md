# DeepContour
This package provides an implementation of the training/inference/autoannotation pipeline of CEnet
v2.0. This is a model that can be either deployed with or without other feature extraction 
backbones (i.e. resnet/vgg/unet/segnet..). 
. For simplicity the bottom model after any backbone(top model) as Coordinates Encoding Networks(CEnet)
## First time setup

The following steps are required in order to run CEnet:

1.  Install deep learning python packages(either linux or windows) .
    *   Recomended: Install anaconda
        [anaconda](https://www.anaconda.com/).
    *   install pytorch
        [pytorch](https://pytorch.org/).
    *   install mission packages (i.e. cv2, seaborn..)
        [pip install packages](https://packaging.python.org/en/latest/tutorials/installing-packages/)
        or
        [conda install packages](https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/)
1.  Organize all the dataset from raw set following the [data set chapter of this instruction](https://docs.google.com/document/d/1mBG2aeF13Qqxt48tZfYnptq_DKhZpqHj/edit?usp=sharing&ouid=104923533845283983955&rtpof=true&sd=true)
	
	folder structure should be like this:

	-Root(my is "E:deeplearning")/

		--Root/dataset/

			---Root/dataset/label data

			---Root/dataset/for IVUS

				----Root/dataset/for IVUS/train

				----Root/dataset/for IVUS/train OLG

				----Root/dataset/for IVUS/test

		--Root/out
			---Root/out/CEnet_trained
				----Root/out/CEnet_trained/telecom




1.  Change de defination of working dir in [working_dir.py](https://gitlab.kuleuven.be/u0132260/atlas_collab_ivus/-/blob/main/DeepContour/working_dir_root.py)


1.  Download [pre-trained model parameters] (https://seafile.unistra.fr/d/0160d5182a1941c68e5a/)

 should be pasted in the folder:

		--Root/out/CEnet_trained

		OR

		--Root/out/unet_trained


1. For cloud based federated learning, a interation file containing the key to access your local folder is  uploaded to here, download [local to loud corresponding comunication json file here ] (https://drive.google.com/drive/folders/1IhIrjm-shSpv_YH1MVgQeU-bBMJ9YLXF)
If the name is  "/media/icube/DATA1/deeplearning/out/CEnet_trained/telecom/local_training_status.json", rename it as "local_training_status.json"

 should be pasted in the folder:

		----Root/out/CEnet_trained/telecom
Train CE-net as [" chapter6-fed learning" of this instruction](https://docs.google.com/document/d/1mBG2aeF13Qqxt48tZfYnptq_DKhZpqHj/edit?usp=sharing&ouid=104923533845283983955&rtpof=true&sd=true)	 
 
## Key useful tools/python scripts
To run any runable scripts in this project,
if using the visual studio (not vs code) or pycharm, it can be run directly. 
If using vs code, please run scripts as module.
If run in terminal of linux or anaconda prompt, run as module, using following command: python3 -m subfoler.scripts(i.e. python3 -m train_multi_obj.CEnet_EXS_p)

1.  In data toolfolder:


	*   read_json.py: transfer one folder of separeted json file into one compact pkl


	*   generator_contour_sheath.py : generated ramdom image with folder of image and the corresponding pkl; it also embeds some scripts to check the stastics of the distribution of the labeled contour 


	*   tSNE.py: which is used to see the data distribution with the reduced dimention (current file does not consider about the label)




1.  In deploy folder (this is normally used after the nets are trained): 

    most of these scripts are used to predict contour, specifically:
 
	*   DeepAutoJson.py is used to generate separated json files from the prediction result of the network.

    And it also has a function for downsampling the dataset for trianing and testing.
1.  train_multi_obj folder 
	*   CEnet_EXS_p.py: train the CEnet 
1.  train_lumen_sheath_segmentation folder, other trainer scripts that used to train other models
1.  other parametersettings:
	*   train_options_CEnets.py set up/modify the taininig options of the proposed CEnet
	*   dataset_ivus.py modify of the training data loader
