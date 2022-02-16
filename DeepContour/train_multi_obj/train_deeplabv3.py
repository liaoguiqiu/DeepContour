# the run py script for sheal and contour detection, 5th October 2020 update
# this uses the encodinhg tranform to use discriminator
# update on 26th July
# Setup the training
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from model import C_Deeplab  # the mmodel
from torchvision import models
from train_display import *
# the model
import arg_parse
import cv2
import numpy
import rendering
# from dataTool import generator_contour
from dataTool import generator_contour_ivus

from dataTool.generator_contour import Generator_Contour, Save_Contour_pkl, Communicate
from dataTool.generator_contour_ivus import Generator_Contour_sheath
from dataTool.generator_contour_ivus_multi import Generator_Contour_sheath
from dataset_ivus import myDataloader, Batch_size, Resample_size, Path_length, max_presence
from FedLearning.Cloud_API import Cloud_API
from train_multi_obj.local2cloud import Local2Cloud
import os
# from dataset_sheath import myDataloader,Batch_size,Resample_size, Path_length
# switch to another data loader for the IVUS, whih will have both the position and existence vector
from working_dir_root import Dataset_root, Output_root
from deploy.basic_trans import Basic_oper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Switch control for the Visdom or Not
Visdom_flag = False  # the flag of using the visdom or not
OLG_flag = False  # flag of training with on line generating or not
Hybrid_OLG = False  # whether  mix with online generated images and real images for training
validation_flag = False  # flag to stop the gradient, and, testing mode which will calculate matrics for validation
Display_fig_flag = True  # display and save result or not
Save_img_flag = False  # this flag determine if the reuslt will be save  in to a foler
Continue_flag = False  # if not true, it start from scratch again
Federated_learning_flag = False  # true to enable the federated learning to interact with cloud, otherwise use the conventional solo learning
Using_fed_model_flag = False  # True: Fed model, false: local model
loadmodel_index = '_2.pth'
Model_key = 'Deeplab'
infinite_save_id = 0  # use this method so that the index of the image will not start from 0 again when switch the folder

if Visdom_flag == True:
    from analy_visdom import VisdomLinePlotter

    plotter = VisdomLinePlotter(env_name='path finding training Plots')

# pth_save_dir = "C:/Workdir/Develop/atlas_collab/out/sheathCGAN_coordinates3/"
pth_save_dir = Output_root + "DeeplabV3_trained/"
if Federated_learning_flag == True:
    cloud_interaction = Local2Cloud(pth_save_dir)

# pth_save_dir = "../out/deep_layers/"

if not os.path.exists(pth_save_dir):
    os.makedirs(pth_save_dir)
from scipy import signal


# weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    pass


# Matrix_dir =  "../dataset/CostMatrix/1/"
# Save_pic_dir = '../DeepPathFinding/out/'
opt = arg_parse.opt
opt.cuda = True
# check the cuda device
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
# dataroot = "../dataset/CostMatrix/"
torch.set_num_threads(2)

# Guiqui 8 layers version
# netD = gan_body._netD_8()

# Guiqiu Resnet version
# netD = layer_body_sheath._netD_8_multiscal_fusion300_layer()

Model_creator = C_Deeplab.Deeplab_creater()  # the  CEnet trainer with CGAN
#   Use the same arch to create two nets
MODEL = Model_creator.creat_deeplab()  # one is for the contour cordinates
# Ex_Nets= Model_creator.creat_nets()   # one is for the contour existence

# netD = gan_body._netD_Resnet()

# netD.apply(weights_init)
MODEL.netD.apply(weights_init)
# MODEL.netG.apply(weights_init)
# MODEL.netE.apply(weights_init)

if Continue_flag == True:
    # netD.load_state_dict(torch.load(opt.netD))
    # ensure loaded module exist
    pretrained_dict = torch.load(pth_save_dir + Model_key + 'G' + loadmodel_index)
    model_dict = MODEL.netG.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict_trim = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict_trim)
    # 3. load the new state dict
    MODEL.netG.load_state_dict(model_dict)
    # MODEL.netG.load_state_dict(torch.load(pth_save_dir+'cGANG_epoch'+loadmodel_index))

    #D
    # MODEL.netD.load_state_dict(torch.load(pth_save_dir+'cGAND_epoch'+loadmodel_index))
    pretrained_dict = torch.load(pth_save_dir + Model_key + 'D' + loadmodel_index)
    model_dict = MODEL.netD.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict_trim = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict_trim)
    # 3. load the new state dict
    MODEL.netD.load_state_dict(model_dict)


    # MODEL.netG.side_branch1. load_state_dict(torch.load(pth_save_dir+'cGANG_branch1_epoch_1.pth'))

if Using_fed_model_flag == True:  # reload
    MODEL.netG = cloud_interaction.reset_model_para(MODEL.netG, name='cGANG')
    MODEL.netD = cloud_interaction.reset_model_para(MODEL.netD, name='cGAND')
if validation_flag == True:
    Federated_learning_flag = False

if validation_flag == True:
    MODEL.netG.Unet_back.eval()
print(MODEL.netD)
print(MODEL.netG)
# print(MODEL.netE)

# no longer use the mine nets

# real_label = 1
# fake_label = 0

if opt.cuda:
    print("CUDA TRUE")
    MODEL.netD.cuda()
    MODEL.netG.cuda()
    # MODEL.netE.cuda()

read_id = 0

epoch = 0
# transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
# transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num = 0
# the first data loader  OLG=OLG_flag depends on this lag for the onlien e generating
mydata_loader1 = myDataloader(Batch_size, Resample_size, Path_length, validation=validation_flag, OLG=OLG_flag)
# the second one will be a offline one for sure
mydata_loader2 = myDataloader(Batch_size, Resample_size, Path_length, validation=validation_flag, OLG=False)
switcher = 0  # this determines to use only one data loader or not (if not, synthetic will be mixed with original)

while (1):  # main infinite loop
    epoch += 1
    if mydata_loader1.read_all_flag2 == 1 and validation_flag == True:
        break

    while (1):  # loop for going through data set

        # -------------- load data and convert to GPU tensor format------------------#
        iteration_num += 1
        read_id += 1
        if (mydata_loader1.read_all_flag == 1):
            read_id = 0
            mydata_loader1.read_all_flag = 0
            break

            # ----switch between synthetic data  and  original data
        if switcher == 0:
            mydata_loader1.read_a_batch()
            mydata_loader = mydata_loader1
            if validation_flag == False and Hybrid_OLG == True:
                switcher = 1
        else:
            switcher = 0
            mydata_loader = mydata_loader2.read_a_batch()
            mydata_loader = mydata_loader2

            # change to 3 chanels
        ini_input = mydata_loader.input_image
        real = torch.from_numpy(numpy.float32(ini_input))
        real = real.to(device)
        np_input = numpy.append(ini_input, ini_input, axis=1)
        np_input = numpy.append(np_input, ini_input, axis=1)

        input = torch.from_numpy(numpy.float32(np_input))
        # input = input.to(device)
        # input = torch.from_numpy(numpy.float32(mydata_loader.input_image[0,:,:,:]))
        input = input.to(device)

        patht = torch.from_numpy(numpy.float32(
            mydata_loader.input_path) / Resample_size)  # the coordinates should be uniformed by the image height
        ex_t = torch.from_numpy(numpy.float32(mydata_loader.exis_vec))
        patht = patht.to(device)  # use the GPU
        ex_t = ex_t.to(device)
        # patht=patht.to(device)
        # patht= torch.from_numpy(numpy.float32(mydata_loader.input_path[0,:])/71.0 )

        # inputv = Variable(input)
        # labelv = patht
        inputv = Variable(input)
        # inputv = Variable(input.unsqueeze(0))
        # patht =patht.view(-1, 1).squeeze(1)
        labelv = Variable(patht)
        ex_v = Variable(ex_t)

        # -------------- load data and convert to GPU tensor format -  end------------------#

        # --------------input, Forward network,  and compare output with the label------------------#
        realA = real
        real_pathes = labelv
        real_exv = ex_v
        MODEL.update_learning_rate()  # update learning rates in the beginning of every epoch.
        MODEL.set_input(realA,real_pathes,real_exv,inputv)     # unpack data from dataset and apply preprocessing

        if validation_flag == True:
            MODEL.forward(validation_flag)
            MODEL.error_calculation()
        else:
            MODEL.optimize_parameters()  # calculate loss functions, get gradients, update network weights
        # --------------input, Forward network,  and compare output with the label - end------------------#

        # -------------- A variety of visualization  ------------------#

        if validation_flag == False:
            G_x = MODEL.displayloss1
            G_x_L12 = MODEL.displayloss2
            # D_x = MODEL.loss_D.data.mean()
            D_x = G_x
            # G_x = MODEL.loss_G . data.mean()
            # G_x_L12= MODEL.loss_G_L1_2 . data.mean()

        # optimizerG.step()

        # save_out  = Fake
        # train with fake
        # if cv2.waitKey(12) & 0xFF == ord('q'):
        #       break

        if validation_flag == False:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, 0, read_id, 0,
                     G_x, D_x, 0, 0, 0))

        if read_id % 1 == 0 and Visdom_flag == True and validation_flag == False:
            plotter.plot('l0', 'l0', 'l0', iteration_num, MODEL.displayloss0.cpu().detach().numpy())
            plotter.plot('l1', 'l1', 'l1', iteration_num, MODEL.displayloss1.cpu().detach().numpy())
            plotter.plot('l2', 'l2', 'l2', iteration_num, MODEL.displayloss2.cpu().detach().numpy())
            # plotter.plot( 'l3', 'l3', 'l3', iteration_num, MODEL.displayloss3.cpu().detach().numpy())
            plotter.plot('lE3', 'le3', 'le3', iteration_num, MODEL.displaylossE0.cpu().detach().numpy())

        if read_id % 1 == 0 and Display_fig_flag == True:
            # vutils.save_image(real_cpu,
            #        '%s/real_samples.png' % opt.outf,
            #        normalize=True)

            train_display(MODEL, realA, mydata_loader, Save_img_flag, read_id, infinite_save_id)

            infinite_save_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # -------------- A variety of visualization - end (this part is a little messy..)  ------------------#
        if read_id % 100 == 0:
            if Federated_learning_flag == True:
                MODEL = cloud_interaction.check_load_global_cloud(MODEL)
    # do checkpointing

    # --------------  save the current trained model after going through a folder  ------------------#
    torch.save(MODEL.netG.state_dict(), pth_save_dir +Model_key +"G_" + str(epoch) + ".pth")
    # torch.save(MODEL.netE.state_dict(), pth_save_dir +Model_key+ "E_" + str(epoch) + ".pth")
    torch.save(MODEL.netD.state_dict(), pth_save_dir +Model_key +"D_" + str(epoch) + ".pth")
    # check to upload
    if Federated_learning_flag == True:
        cloud_interaction.save_local_cloud(MODEL)
    # check to update

    # torch.save(MODEL.netG.side_branch1.  state_dict(), pth_save_dir+ "cGANG_branch1_epoch_"+str(epoch)+".pth")

    if epoch >= 5:  # just save 5 newest historical models
        epoch = 0


