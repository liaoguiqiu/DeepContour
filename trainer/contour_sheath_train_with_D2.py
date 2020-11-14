# the run py script for sheal and contour detection, 5th October 2020 update
# this uses the encodinhg tranform to use discriminator
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from model import cGAN_build2 # the mmodel

 
# the model
import arg_parse
import cv2
import numpy
import rendering
from generator_contour import Generator_Contour,Save_Contour_pkl,Communicate,Generator_Contour_layers,Generator_Contour_sheath

import os
from dataset_sheath import myDataloader,Batch_size,Resample_size, Path_length
from deploy.basic_trans import Basic_oper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Switch control for the Visdom or Not
Visdom_flag  = True 
OLG_flag = True
validation_flag = False

Display_fig_flag = True
Continue_flag = False
if Visdom_flag == True:
    from analy_visdom import VisdomLinePlotter
    plotter = VisdomLinePlotter(env_name='path finding training Plots')


pth_save_dir = "../out/sheathCGAN_coordinates3/"
#pth_save_dir = "../out/deep_layers/"

if not os.path.exists(pth_save_dir):
    os.makedirs(pth_save_dir)
from scipy import signal 
Matrix_dir =  "../dataset/CostMatrix/1/"
Save_pic_dir = '../DeepPathFinding/out/'
opt = arg_parse.opt
opt.cuda = True
# check the cuda device 
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
dataroot = "../dataset/CostMatrix/"

torch.set_num_threads(2)
######################################################################
# Data
# ----
# 
# In this tutorial we will use the `Celeb-A Faces
# dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__ which can
# be downloaded at the linked site, or in `Google
# Drive <https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg>`__.
# The dataset will download as a file named *img_align_celeba.zip*. Once
# downloaded, create a directory named *celeba* and extract the zip file
# into that directory. Then, set the *dataroot* input for this notebook to
# the *celeba* directory you just created. The resulting directory
# structure should be:
# 
# ::
# 
#    /path/to/celeba
#        -> img_align_celeba  
#            -> 188242.jpg
#            -> 173822.jpg
#            -> 284702.jpg
#            -> 537394.jpg
#               ...
# 
# This is an important step because we will be using the ImageFolder
# dataset class, which requires there to be subdirectories in the
# dataset’s root folder. Now, we can create the dataset, create the
# dataloader, set the device to run on, and finally visualize some of the
# training data.
# 

# We can use an image folder dataset the way we have it setup.
# Create the dataset
 
nz = int(arg_parse.opt.nz) # number of latent variables
ngf = int(arg_parse.opt.ngf) # inside generator
ndf = int(arg_parse.opt.ndf) # inside discriminator
nc = 3 # channels

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#netG = gan_body._netG()
#netG.apply(weights_init)
#if opt.netG != '':
#    netG.load_state_dict(torch.load(opt.netG))
#print(netG)


#netD = gan_body._netD()
#Guiqui 8 layers version
#netD = gan_body._netD_8()

#Guiqiu Resnet version
#netD = layer_body_sheath._netD_8_multiscal_fusion300_layer()
gancreator = cGAN_build2.CGAN_creator() # the Cgan for the segmentation 
GANmodel= gancreator.creat_cgan()  #  G and D are created here 
#netD = gan_body._netD_Resnet()




#netD.apply(weights_init)
GANmodel.netD.apply(weights_init)
GANmodel.netG.apply(weights_init)
if Continue_flag == True:
    #netD.load_state_dict(torch.load(opt.netD))
    GANmodel.netG.load_state_dict(torch.load(pth_save_dir+'cGANG_epoch_4.pth'))
    GANmodel.netD.load_state_dict(torch.load(pth_save_dir+'cGAND_epoch_4.pth'))
    #GANmodel.netG.side_branch1. load_state_dict(torch.load(pth_save_dir+'cGANG_branch1_epoch_1.pth'))

    #torch.save(GANmodel.netG.side_branch1.  state_dict(), pth_save_dir+ "cGANG_branch1_epoch_"+str(epoch)+".pth")


print(GANmodel.netD)
print(GANmodel.netG)
 # no longer use the mine nets 


criterion = nn.L1Loss()
criterion2  = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    print("CUDA TRUE")
    GANmodel.netD.cuda()
    GANmodel.netG.cuda()

    #netD.cuda()
    #netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()


fixed_noise = Variable(fixed_noise)

# setup optimizer
#optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay =2e-4 )
#optimizerD = optim.SGD(netD.parameters(), lr=opt.lr,momentum= 0.9, weight_decay =2e-4 )


#saved_stastics = MY_ANALYSIS()
#saved_stastics=saved_stastics.read_my_signal_results()
#saved_stastics.display()

read_id =0

epoch=0
#transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
#transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num =0
mydata_loader1 = myDataloader (Batch_size,Resample_size,Path_length,validation = validation_flag,OLG=OLG_flag)
mydata_loader2 = myDataloader (Batch_size,Resample_size,Path_length,validation = validation_flag,OLG=False)
switcher =0
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
def display_prediction(mydata_loader,save_out,hot): # display in coordinates form 
    gray2  =   (mydata_loader.input_image[0,0,:,:] *104)+104
    show1 = gray2.astype(float)
    path2 = mydata_loader.input_path[0,:] 
    #path2  = signal.resample(path2, Resample_size)
    path2 = numpy.clip(path2,0,Resample_size-1)
    color1  = numpy.zeros((show1.shape[0],show1.shape[1],3))
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
    colorhot = color *hot
             
    sheath = signal.resample(save_out[0], Resample_size)
    tissue = signal.resample(save_out[1], Resample_size)

    color = draw_coordinates_color_s(color,sheath,tissue)
    color2 = draw_coordinates_color_s(colorhot,sheath,tissue)

            
    #show3 = numpy.append(show1,show2,axis=1) # cascade
    show4 = numpy.append(color1,color,axis=1) # cascade
    circular1 = Basic_oper.tranfer_frome_rec2cir2(color) 
    circular2 = Basic_oper.tranfer_frome_rec2cir2(color2) 


    cv2.imshow('Deeplearning one 2',show4.astype(numpy.uint8)) 
    cv2.imshow('Deeplearning circ',circular1.astype(numpy.uint8)) 
    cv2.imshow('Deeplearning circ2',circular2.astype(numpy.uint8)) 
    cv2.imshow('Deeplearning color',color2.astype(numpy.uint8)) 




while(1):
    epoch+= 1
    #almost 900 pictures
    while(1):
        iteration_num +=1
        read_id+=1
        if (mydata_loader1.read_all_flag ==1):
            read_id =0
            mydata_loader1.read_all_flag =0
            break

        #mydata_loader1 .read_a_batch()
        #mydata_loader =mydata_loader1 

         ##switch btween fake and real
        if switcher==0:
           mydata_loader1 .read_a_batch()
           mydata_loader =mydata_loader1 
           if validation_flag == False    :
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
   
        patht= torch.from_numpy(numpy.float32(mydata_loader.input_path)/Resample_size )
        #patht=patht.to(device)
                
        #patht= torch.from_numpy(numpy.float32(mydata_loader.input_path[0,:])/71.0 )
        patht=patht.to(device)
        #inputv = Variable(input)
        # using unsqueeze is import  for with out bactch situation
        #inputv = Variable(input.unsqueeze(0))

        #labelv = Variable(patht)
        #inputv = input

        #labelv = patht
        inputv = Variable(input )
        #inputv = Variable(input.unsqueeze(0))
        #patht =patht.view(-1, 1).squeeze(1)

        labelv = Variable(patht)
        # just test the first boundary effect 
        #labelv  = labelv[:,0,:]

        ###
        ###chnage the imout domain can use the same label to ralize generating or line segementation 
        #realA = rendering.layers_visualized(labelv,Resample_size)
        #realB = real
        realA =  real
        #realB =  rendering.layers_visualized_integer_encodeing(labelv,Resample_size)
        #realB =  rendering.layers_visualized (labelv,Resample_size)

        real_pathes = labelv
          
        GANmodel.update_learning_rate()    # update learning rates in the beginning of every epoch.
        GANmodel.set_input(realA,real_pathes,inputv)         # unpack data from dataset and apply preprocessing

        if validation_flag ==True:
            GANmodel.forward()
            GANmodel.error_calculation()
        else:
            GANmodel.optimize_parameters()   # calculate loss functions, get gradients, update network weights

         
        #outputall = netD(inputv)
        #outputall =  outputall[:,:,0,:]
        #output = outputall[0].view(Batch_size,netD.layer_num,Path_length).squeeze(1)
        #output1 = outputall[1].view(Batch_size,netD.layer_num,Path_length).squeeze(1)
        #output2 = outputall[2].view(Batch_size,netD.layer_num,Path_length).squeeze(1)
        ##output = outputall[0]  
        ##output1 = outputall[1] 
        ##output2 = outputall[2]  
        ##output


        #netG.zero_grad()
        #errD_real = criterion(Fake, real)
        ##errD_real1 = criterion(output1, labelv)
        ##errD_real2 = criterion(output2, labelv)
        ##errD_real_fuse = 1.0*(errD_real+  0.1*errD_real1 +  0.1*errD_real2)

        ##errD_real1.backward()errD_real = criterion(output, labelv)
        ##errD_real1 = criterion(output1, labelv)
        ##errD_real2 = criterion(output22, labelv)
        ##errD_real_fuse =   0.1*errD_real1 +  0.1*errD_real2

        #errD_real.backward()

        
        #errD_real.backward()
 
        if validation_flag ==False:
            D_x = GANmodel.loss_D.data.mean()
            G_x = GANmodel.displayloss1 
            G_x_L12= GANmodel.displayloss2
            #G_x = GANmodel.loss_G . data.mean() 
            #G_x_L12= GANmodel.loss_G_L1_2 . data.mean()  

        #D_x1 = errD_real1.data.mean()
        #D_x2 = errD_real2.data.mean()
        #D_xf = errD_real_fuse.data.mean()


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
                plotter.plot( 'l0', 'l0', 'l0', iteration_num, GANmodel.displayloss0.cpu().detach().numpy())

                plotter.plot( 'l1', 'l1', 'l1', iteration_num, GANmodel.displayloss1.cpu().detach().numpy())
                plotter.plot( 'l2', 'l2', 'l2', iteration_num, GANmodel.displayloss2.cpu().detach().numpy())
                plotter.plot( 'l3', 'l3', 'l3', iteration_num, GANmodel.displayloss3.cpu().detach().numpy())

                #plotter.plot( 'cLOSS1', 'cLOSS1', 'cLOSS1', iteration_num, D_x1.cpu().detach().numpy())
                #plotter.plot( 'cLOSS12', 'cLOSS2', 'cLOSS2', iteration_num, D_x2.cpu().detach().numpy())
                #plotter.plot( 'cLOSS_f', 'cLOSSf', 'cLOSSf', iteration_num, D_xf.cpu().detach().numpy())
        if read_id % 1 == 0 and Display_fig_flag== True:
            #vutils.save_image(real_cpu,
            #        '%s/real_samples.png' % opt.outf,
            #        normalize=True)
            #netG.eval()
            #fake = netG(fixed_noise)
            #cv2.imwrite(Save_pic_dir  + str(i) +".jpg", mat)
            #show the result


            gray2  =   realA[0,0,:,:].cpu().detach().numpy()*104+104
            show1 = gray2.astype(float)
            #path2 = mydata_loader.input_path[0,:] 
            ##path2  = signal.resample(path2, Resample_size)
            #path2 = numpy.clip(path2,0,Resample_size-1)
            color1  = numpy.zeros((show1.shape[0],show1.shape[1],3))
            color1[:,:,0]  =color1[:,:,1] = color1[:,:,2] = show1 [:,:]

            oneHot =  GANmodel.fake_B_1_hot[0,:,:,:].cpu().detach().numpy() 

            
            hot  = numpy.zeros((oneHot.shape[1],oneHot.shape[2],3))
            hot[:,:,0]  =  oneHot [0,:,:]
            hot[:,:,1]  =  oneHot [1,:,:]
            hot[:,:,2]  =  oneHot [2,:,:]

            #oneHot  = GANmodel.fake_B_1_hot[0,:,:,:]
            #oneHot = oneHot.view(Path_length,Path_length,3)
            #oneHot  =  oneHot.cpu().detach().numpy()
            #color1 = color1* hot
           

            #for i in range ( len(path2)):
            #    color1 = draw_coordinates_color(color1,path2[i],i)
                 
            saveout  = GANmodel.fake_B
            
            show2 =  saveout[0,:,:,:].cpu().detach().numpy()*255 

            
            #color  = numpy.zeros((show2.shape[1],show2.shape[2],3))
            #color[:,:,0]  =  show2 [0,:,:]
            #color[:,:,1]  =  show2 [1,:,:]
            #color[:,:,2]  =  show2 [2,:,:]
            color  = numpy.zeros((show2.shape[1],show2.shape[2],3))
            color[:,:,0]  =color[:,:,1] = color[:,:,2] = show2[0,:,:] 
         
            #for i in range ( len(path2)):
            #    color = draw_coordinates_color(color,path2[i],i)
                
          


            
            #show3 = numpy.append(show1,show2,axis=1) # cascade
            show4 = numpy.append(color1,color,axis=1) # cascade

            cv2.imshow('Deeplearning one',show4.astype(numpy.uint8)) 

            real_label = GANmodel.real_B
            show5 =  real_label[0,0,:,:].cpu().detach().numpy()*255 
            cv2.imshow('real',show5.astype(numpy.uint8)) 

            #display_prediction(mydata_loader,  GANmodel.out_pathes[0],hot)
            #display_prediction(mydata_loader,  GANmodel.out_pathes0,hot)
            display_prediction(mydata_loader,  GANmodel.path_long3,hot)
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch
    torch.save(GANmodel.netG.state_dict(), pth_save_dir+ "cGANG_epoch_"+str(epoch)+".pth")
    torch.save(GANmodel.netD.state_dict(), pth_save_dir+ "cGAND_epoch_"+str(epoch)+".pth")
    torch.save(GANmodel.netG.side_branch1.  state_dict(), pth_save_dir+ "cGANG_branch1_epoch_"+str(epoch)+".pth")


    #cv2.imwrite(Save_pic_dir  + str(epoch) +".jpg", show2)
    #cv2.imwrite(pth_save_dir  + str(epoch) +".jpg", show2)
    if epoch >=5:
        epoch =0


