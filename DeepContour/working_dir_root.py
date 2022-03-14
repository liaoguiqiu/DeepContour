#the root dir for switching computer, this will be imported to script that interact with disk 
#PS: better keep the structure within the root the same for every computer


########################################################################
# The data root formate should be (double check the name of the folder):

#-Root(my is "E:deeplearning")/
#    --Root/dataset/
#        ---Root/dataset/label data
#        ---Root/dataset/for IVUS
#            ---Root/dataset/for IVUS/train
#            ---Root/dataset/for IVUS/train OLG
#            ---Root/dataset/for IVUS/test
#    --Root/out


########################################################################


#this is for Guiqiu Windows
# working_root = "E:/Deep learning/"

#this is for marcopolo linux

# working_root = "/media/icube/DATA1/deeplearning/"

# For Tania's computer
working_root="/media/icube/DATA1/Working dir CNN/"

#this is for Beatriz's computer
# working_root = "C:/Workdir/Develop/atlas_collab_ivus/"

#this is for Sanat's computer
# working_root = "/home/beatriz/Documents/Projects/atlas_collab_ivus/"

Dataset_root =  working_root + "dataset/"
config_root =   working_root + "config/"
Output_root =   working_root + "out/"