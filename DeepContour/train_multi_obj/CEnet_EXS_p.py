# the run py script for sheal and contour detection, 5th October 2020 update
# this uses the encodinhg tranform to use discriminator
# update on 26th July 
#Setup the training
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from model import CE_build3 # the mmodel

 
# the model
import arg_parse
import cv2
import numpy
import rendering
from dataTool import generator_contour 
from dataTool.generator_contour import Generator_Contour,Save_Contour_pkl,Communicate
from dataTool.generator_contour_ivus import Generator_Contour_sheath
from dataset_ivus  import myDataloader,Batch_size,Resample_size, Path_length

import os

#from dataset_sheath import myDataloader,Batch_size,Resample_size, Path_length
#switch to another data loader for the IVUS, whih will have both the position and existence vector

from deploy.basic_trans import Basic_oper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Switch control for the Visdom or Not
Visdom_flag  = False  # the flag of using the visdom or not
OLG_flag = False    # flag of training with on line generating or not
Hybrid_OLG = False  # whether  mix with online generated images and real images for training
validation_flag = True  # flag to stop the gradient, and, testing mode which will calculate matrics for validation
Display_fig_flag = True  #  display and save result or not 
Save_img_flag  = False # this flag determine if the reuslt will be save  in to a foler 
Continue_flag = True  # if not true, it start from scratch again

infinite_save_id =0 # use this method so that the index of the image will not start from 0 again when switch the folder    

if Visdom_flag == True:
    from analy_visdom import VisdomLinePlotter
    plotter = VisdomLinePlotter(env_name='path finding training Plots')

# pth_save_dir = "C:/Workdir/Develop/atlas_collab/out/sheathCGAN_coordinates3/"
pth_save_dir = "../../out/sheathCGAN_coordinates3/"
backup_dir = "../../out/backup/"

#pth_save_dir = "../out/deep_layers/"

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
# 3 functions to drae the results in real time
def draw_coordinates_color(img1,vy,color):
        
        if color ==0:
           painter  = [254,0,0]
        elif color ==1:
           painter  = [0,254,0]
        elif color ==2:
           painter  = [0,0,254]
        else :
           painter  = [0,0,0]
                    #path0  = signal.resample(path0, W)
        H,W,_ = img1.shape
        for j in range (W):
                #path0l[path0x[j]]
                dy = numpy.clip(vy[j],2,H-2)
            

                img1[int(dy)+1,j,:]=img1[int(dy),j,:]=painter
                img1[int(dy)-1,j,:]=img1[int(dy)-2,j,:]=painter

                #img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter


        return img1
def draw_coordinates_color_s(img1,vy0,vy1):
        
         
        H,W,_ = img1.shape
        for j in range (W):
                #path0l[path0x[j]]
                dy1 = numpy.clip(vy1[j],2,H-2)
                dy0 = numpy.clip(vy0[j],2,H-2)

                
                if (dy1 == H-2 ):

                    img1[int(dy1)+1,j,:]=img1[int(dy1),j,:]=[0,254,254]
                    img1[int(dy1)-1,j,:]=img1[int(dy1)-2,j,:]=[0,254,254]
                if (abs(dy0-dy1)<= 5 ):

                    img1[int(dy1)+1,j,:]=img1[int(dy1),j,:]=[254,0,254]
                    img1[int(dy1)-1,j,:]=img1[int(dy1)-2,j,:]=[254,0,254]
                #img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter


        return img1
def display_prediction_exis(read_id,mydata_loader,save_out): # display in coordinates form 
    gray2 =   (mydata_loader.input_image[0,0,:,:] *104)+104
    show1 = gray2.astype(float)
    path2 = mydata_loader.exis_vec[0,:] * Resample_size
    #path2  = signal.resample(path2, Resample_size)
    path2 = numpy.clip(path2,0,Resample_size-1)
    color1 = numpy.zeros((show1.shape[0],show1.shape[1],3))
    color1[:,:,0]  =color1[:,:,1] = color1[:,:,2] = show1 

    for i in range ( len(path2)):
        color1 = draw_coordinates_color(color1,path2[i],i)
                             
            
    show2 =  gray2.astype(float)
    save_out = save_out.cpu().detach().numpy()

    save_out  = save_out[0,:] *(Resample_size)
    #save_out  = signal.resample(save_out, Resample_size)
    save_out = numpy.clip(save_out,0,Resample_size-1)
    color  = numpy.zeros((show2.shape[0],show2.shape[1],3))
    color[:,:,0]  =color[:,:,1] = color[:,:,2] = show2  
     
     

    for i in range ( len(save_out)):
        this_coordinate = signal.resample(save_out[i], Resample_size)
        color = draw_coordinates_color(color,this_coordinate,i)
     
    #show3 = numpy.append(show1,show2,axis=1) # cascade
    show4 = numpy.append(color1,color,axis=1) # cascade
 
     


    cv2.imshow('Deeplearning exitence 2',show4.astype(numpy.uint8)) 
def display_prediction(read_id,mydata_loader,save_out,hot,hot_real): # display in coordinates form 
    gray2 =   (mydata_loader.input_image[0,0,:,:] *104)+104
    show1 = gray2.astype(float)
    path2 = mydata_loader.input_path[0,:] 
    #path2  = signal.resample(path2, Resample_size)
    path2 = numpy.clip(path2,0,Resample_size-1)
    color1 = numpy.zeros((show1.shape[0],show1.shape[1],3))
    color1[:,:,0]  =color1[:,:,1] = color1[:,:,2] = show1 

    for i in range ( len(path2)):
        color1 = draw_coordinates_color(color1,path2[i],i)
                             
            
    show2 =  gray2.astype(float)
    save_out = save_out.cpu().detach().numpy()

    save_out  = save_out[0,:] *(Resample_size)
    #save_out  = signal.resample(save_out, Resample_size)
    save_out = numpy.clip(save_out,0,Resample_size-1)
    color  = numpy.zeros((show2.shape[0],show2.shape[1],3))
    color[:,:,0]  =color[:,:,1] = color[:,:,2] = show2  
    colorhot_real = (color+50 )*hot_real 
    sheath_real  = signal.resample(path2[0], Resample_size)
    tissue_real  = signal.resample(path2[1], Resample_size)
    colorhot_real = draw_coordinates_color_s(colorhot_real,sheath_real,tissue_real)
    circular_color_real = Basic_oper.tranfer_frome_rec2cir2(colorhot_real) 
    cv2.imshow('color real cir',circular_color_real.astype(numpy.uint8)) 
    if  Save_img_flag == True:
        cv2.imwrite("D:/Deep learning/out/1out_img/ground_circ/"  +
                        str(mydata_loader.save_id) +".jpg",circular_color_real )  

    for i in range ( len(save_out)):
        this_coordinate = signal.resample(save_out[i], Resample_size)
        color = draw_coordinates_color(color,this_coordinate,i)
    colorhot = color *hot
   


             
    sheath = signal.resample(save_out[0], Resample_size)
    tissue = signal.resample(save_out[1], Resample_size)
   
    color = draw_coordinates_color_s(color,sheath,tissue)
    color2 = draw_coordinates_color_s(colorhot,sheath,tissue)

    color_real  =  draw_coordinates_color_s(colorhot_real,sheath,tissue)
            
    #show3 = numpy.append(show1,show2,axis=1) # cascade
    show4 = numpy.append(color1,color,axis=1) # cascade
    circular1 = Basic_oper.tranfer_frome_rec2cir2(color) 
    circular2 = Basic_oper.tranfer_frome_rec2cir2(color2)
    if  Save_img_flag == True:
        cv2.imwrite("D:/Deep learning/out/1out_img/Ori_seg_rec_line/"  +
                            str(infinite_save_id) +".jpg",show4 )  


    cv2.imshow('Deeplearning one 2',show4.astype(numpy.uint8)) 

    cv2.imshow('Deeplearning circ',circular1.astype(numpy.uint8)) 
    cv2.imshow('Deeplearning circ2',circular2.astype(numpy.uint8))
    if  Save_img_flag == True:
        cv2.imwrite("D:/Deep learning/out/1out_img/seg_circ/"  +
                        str(mydata_loader.save_id) +".jpg",circular2 )  
    cv2.imshow('Deeplearning color',color2.astype(numpy.uint8)) 
    cv2.imshow('  color real',color_real.astype(numpy.uint8)) 


#Matrix_dir =  "../dataset/CostMatrix/1/"
#Save_pic_dir = '../DeepPathFinding/out/'
opt = arg_parse.opt
opt.cuda = True
# check the cuda device 
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
#dataroot = "../dataset/CostMatrix/"
torch.set_num_threads(2)
 
  
#Guiqui 8 layers version
#netD = gan_body._netD_8()

#Guiqiu Resnet version
#netD = layer_body_sheath._netD_8_multiscal_fusion300_layer()

Model_creator = CE_build3.CE_creator() # the  CEnet trainer with CGAN
#   Use the same arch to create two nets 
CE_Nets= Model_creator.creat_nets()   # one is for the contour cordinates
#Ex_Nets= Model_creator.creat_nets()   # one is for the contour existence

#netD = gan_body._netD_Resnet()

#netD.apply(weights_init)
CE_Nets.netD.apply(weights_init)
CE_Nets.netG.apply(weights_init)
CE_Nets.netE.apply(weights_init)

if Continue_flag == True:
    #netD.load_state_dict(torch.load(opt.netD))
    CE_Nets.netG.load_state_dict(torch.load(pth_save_dir+'cGANG_epoch_1.pth'))
    CE_Nets.netD.load_state_dict(torch.load(pth_save_dir+'cGAND_epoch_1.pth'))
    CE_Nets.netE.load_state_dict(torch.load(pth_save_dir+'cGANE_epoch_1.pth'))
    #CE_Nets.netG.side_branch1. load_state_dict(torch.load(pth_save_dir+'cGANG_branch1_epoch_1.pth'))

print(CE_Nets.netD)
print(CE_Nets.netG)
print(CE_Nets.netE)

 # no longer use the mine nets 
  
#real_label = 1
#fake_label = 0

if opt.cuda:
    print("CUDA TRUE")
    CE_Nets.netD.cuda()
    CE_Nets.netG.cuda()
    CE_Nets.netE.cuda()

     

read_id =0

epoch=0
#transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
#transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num =0
# the first data loader  OLG=OLG_flag depends on this lag for the onlien e generating 
mydata_loader1 = myDataloader(Batch_size,Resample_size,Path_length,validation = validation_flag,OLG=OLG_flag)
# the second one will be a offline one for sure 
mydata_loader2 = myDataloader(Batch_size,Resample_size,Path_length,validation = validation_flag,OLG=False)
switcher =0 # this determines to use only one data loader or not (if not, synthetic will be mixed with original)


while(1): # main infinite loop 
    epoch+= 1
    if mydata_loader1.read_all_flag2 == 1 and validation_flag ==True:
        break

    while(1): # loop for going through data set 
        #-------------- load data and convert to GPU tensor format------------------#
        iteration_num +=1
        read_id+=1
        if (mydata_loader1.read_all_flag ==1):
            read_id =0
            mydata_loader1.read_all_flag =0
            break   

        #----switch between synthetic data  and  original data 
        if switcher==0:
           mydata_loader1 .read_a_batch()
           mydata_loader =mydata_loader1 
           if validation_flag == False and Hybrid_OLG == True   :
                switcher=1
        else:
           switcher =0
           mydata_loader =mydata_loader2 .read_a_batch()
           mydata_loader =mydata_loader2  
        
        #change to 3 chanels
        ini_input = mydata_loader.input_image
        real =  torch.from_numpy(numpy.float32(ini_input)) 
        real = real.to(device)                
        np_input = numpy.append(ini_input,ini_input,axis=1)
        np_input = numpy.append(np_input,ini_input,axis=1)

        input = torch.from_numpy(numpy.float32(np_input)) 
        #input = input.to(device) 
        #input = torch.from_numpy(numpy.float32(mydata_loader.input_image[0,:,:,:])) 
        input = input.to(device)                
   
        patht= torch.from_numpy(numpy.float32(mydata_loader.input_path)/Resample_size) # the coordinates should be uniformed by the image height
        ex_t = torch.from_numpy(numpy.float32(mydata_loader.exis_vec))
        patht=patht.to(device) # use the GPU 
        ex_t = ex_t.to(device)
        #patht=patht.to(device)            
        #patht= torch.from_numpy(numpy.float32(mydata_loader.input_path[0,:])/71.0 )
        


        #inputv = Variable(input)
        #labelv = patht
        inputv = Variable(input )
        #inputv = Variable(input.unsqueeze(0))
        #patht =patht.view(-1, 1).squeeze(1)
        labelv = Variable(patht)
        ex_v = Variable(ex_t)

        #-------------- load data and convert to GPU tensor format -  end------------------#

         

        #--------------input, Forward network,  and compare output with the label------------------#
        realA =  real
        real_pathes = labelv
        real_exv = ex_v
        CE_Nets.update_learning_rate()    # update learning rates in the beginning of every epoch.
        CE_Nets.set_input(realA,real_pathes,real_exv,inputv)         # unpack data from dataset and apply preprocessing

        if validation_flag ==True:
            CE_Nets.forward()
            CE_Nets.error_calculation()
        else:
            CE_Nets.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        #--------------input, Forward network,  and compare output with the label - end------------------#

         
      
        #-------------- A variety of visualization  ------------------#
        
        if validation_flag ==False:
            G_x = CE_Nets.displayloss1 
            G_x_L12= CE_Nets.displayloss2
            #D_x = CE_Nets.loss_D.data.mean()
            D_x  = G_x
            #G_x = CE_Nets.loss_G . data.mean() 
            #G_x_L12= CE_Nets.loss_G_L1_2 . data.mean()  
             
        #optimizerG.step()

        #save_out  = Fake
        # train with fake
        # if cv2.waitKey(12) & 0xFF == ord('q'):
        #       break 

        if validation_flag==False:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, 0, read_id, 0,
                     G_x, D_x, 0, 0, 0))

        if read_id % 2 == 0 and Visdom_flag == True and validation_flag==False:
                plotter.plot( 'l0', 'l0', 'l0', iteration_num, CE_Nets.displayloss0.cpu().detach().numpy())
                plotter.plot( 'l1', 'l1', 'l1', iteration_num, CE_Nets.displayloss1.cpu().detach().numpy())
                plotter.plot( 'l2', 'l2', 'l2', iteration_num, CE_Nets.displayloss2.cpu().detach().numpy())
                #plotter.plot( 'l3', 'l3', 'l3', iteration_num, CE_Nets.displayloss3.cpu().detach().numpy())
                plotter.plot( 'lE3', 'le3', 'le3', iteration_num, CE_Nets.displaylossE0 .cpu().detach().numpy())

                 
        if read_id % 1 == 0 and Display_fig_flag== True:
            #vutils.save_image(real_cpu,
            #        '%s/real_samples.png' % opt.outf,
            #        normalize=True)

            gray2  =   realA[0,0,:,:].cpu().detach().numpy()*104+104
            show1 = gray2.astype(float)
            #path2 = mydata_loader.input_path[0,:] 
            ##path2  = signal.resample(path2, Resample_size)
            #path2 = numpy.clip(path2,0,Resample_size-1)
            color1  = numpy.zeros((show1.shape[0],show1.shape[1],3))
            color1[:,:,0]  =color1[:,:,1] = color1[:,:,2] = show1 [:,:]

            oneHot =  CE_Nets.fake_B_1_hot[0,:,:,:].cpu().detach().numpy() 

            hot  = numpy.zeros((oneHot.shape[1],oneHot.shape[2],3))
            hot[:,:,0]  =  oneHot [0,:,:]
            hot[:,:,1]  =  oneHot [1,:,:]
            hot[:,:,2]  =  oneHot [2,:,:]

            oneHot_real =  CE_Nets.real_B_one_hot[0,:,:,:].cpu().detach().numpy() 
          
            hot_real  = numpy.zeros((oneHot.shape[1],oneHot.shape[2],3))
            hot_real[:,:,0]  =  oneHot_real [0,:,:]
            hot_real[:,:,1]  =  oneHot_real [1,:,:]
            hot_real[:,:,2]  =  oneHot_real [2,:,:]
          
         
            saveout  = CE_Nets.fake_B
            show2 =  saveout[0,:,:,:].cpu().detach().numpy()*255 

            
         
            color  = numpy.zeros((show2.shape[1],show2.shape[2],3))
            color[:,:,0]  =color[:,:,1] = color[:,:,2] = show2[0,:,:] 
         
            #for i in range ( len(path2)):
            #    color = draw_coordinates_color(color,path2[i],i)
                
          


            
            #show3 = numpy.append(show1,show2,axis=1) # cascade
            show4 = numpy.append(color1,color,axis=1) # cascade
            # the circular of the original image 
            circ_original = Basic_oper.tranfer_frome_rec2cir2(color1) 

            cv2.imshow('Original circular',circ_original.astype(numpy.uint8)) 
            if  Save_img_flag == True:
                cv2.imwrite("D:/Deep learning/out/1out_img/original_circ/"  +
                            str(mydata_loader.save_id) +".jpg",circ_original )
            #infinite_save_id
            


            cv2.imshow('Deeplearning one',show4.astype(numpy.uint8)) 
            if  Save_img_flag == True:
                cv2.imwrite("D:/Deep learning/out/1out_img/Ori_seg_rec/"  +
                            str(infinite_save_id) +".jpg",show4 )
            real_label = CE_Nets.real_B
            show5 =  real_label[0,0,:,:].cpu().detach().numpy()*255 
            cv2.imshow('real',show5.astype(numpy.uint8)) 
            if  Save_img_flag == True:
                cv2.imwrite("D:/Deep learning/out/1out_img/ground_rec/"  +
                            str(infinite_save_id) +".jpg",show5 )

            #display_prediction(mydata_loader,  CE_Nets.out_pathes[0],hot)
            #display_prediction(mydata_loader,  CE_Nets.path_long3,hot)
            #display_prediction(mydata_loader,  CE_Nets.out_pathes3,hot)
            #display_prediction(read_id,mydata_loader,  CE_Nets.out_pathes0,hot,hot_real)
            display_prediction(read_id,mydata_loader,  CE_Nets.out_pathes0,hot,hot_real)
            display_prediction_exis(read_id,mydata_loader,  CE_Nets.out_exis_v0 )

            infinite_save_id += 1 
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break
            #-------------- A variety of visualization - end (this part is a little messy..)  ------------------#
            
    # do checkpointing

    #--------------  save the current trained model after going through a folder  ------------------#
    torch.save(CE_Nets.netG.state_dict(), pth_save_dir+ "cGANG_epoch_"+str(epoch)+".pth")
    torch.save(CE_Nets.netE.state_dict(), pth_save_dir+ "cGANE_epoch_"+str(epoch)+".pth")

    torch.save(CE_Nets.netD.state_dict(), pth_save_dir+ "cGAND_epoch_"+str(epoch)+".pth")
    torch.save(CE_Nets.netG.side_branch1.  state_dict(), pth_save_dir+ "cGANG_branch1_epoch_"+str(epoch)+".pth")
     
    if epoch >=5: # just save 5 newest historical models  
        epoch =0


