# the run py script for sheal and contour detection, 5th October 2020 update
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
 
from model import cGAN_build # the mmodel
 
import layer_body_sheath # the model
import arg_parse
#import imagenet
from analy import MY_ANALYSIS
from analy import Save_signal_enum
import cv2
import numpy
from image_trans import BaseTransform  
from generator_contour import Generator_Contour,Save_Contour_pkl,Communicate 
from dataTool.generator_contour_ivus import Generator_Contour_sheath
import rendering


import os
#from dataset_sheath import myDataloader,Batch_size,Resample_size, Path_length
from dataset_ivus  import myDataloader,Batch_size,Resample_size, Path_length

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Switch control for the Visdom or Not
# Switch control for the Visdom or Not
Visdom_flag  = False  # the flag of using the visdom or not
OLG_flag = True    # flag of training with on line generating or not
Hybrid_OLG = False  # whether  mix with online generated images and real images for training
validation_flag = True  # flag to stop the gradient, and, testing mode which will calculate matrics for validation
Display_fig_flag = True  #  display and save result or not 
Save_img_flag  = False # this flag determine if the reuslt will be save  in to a foler 
Continue_flag = True  # if not true, it start from scratch again
if Visdom_flag == True:
    from analy_visdom import VisdomLinePlotter
    plotter = VisdomLinePlotter(env_name='path finding training Plots')
#infinite saving term
infinite_save_id =0

pth_save_dir = "../out/sheathCGAN/"
pth_save_dir = "../out/deep_layers/"

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
netD = layer_body_sheath._netD_8_multiscal_fusion300_layer()
gancreator = cGAN_build.CGAN_creator() # the Cgan for the segmentation 
GANmodel= gancreator.creat_cgan()  #  G and D are created here 
#netD = gan_body._netD_Resnet()




#netD.apply(weights_init)
GANmodel.netD.apply(weights_init)
GANmodel.netG.apply(weights_init)
if Continue_flag == True:
    #netD.load_state_dict(torch.load(opt.netD))
    GANmodel.netG.load_state_dict(torch.load('../out/deep_layers/cGANG_epoch_1.pth'))
    GANmodel.netD.load_state_dict(torch.load('../out/deep_layers/cGAND_epoch_1.pth'))

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
    netD.cuda()
    #netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()


fixed_noise = Variable(fixed_noise)

# setup optimizer
#optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay =2e-4 )
#optimizerD = optim.SGD(netD.parameters(), lr=opt.lr,momentum= 0.9, weight_decay =2e-4 )


#saved_stastics = MY_ANALYSIS()
#saved_stastics=saved_stastics.read_my_signal_results()
#saved_stastics.display()

read_id =0

epoch=0
#transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
#transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num =0
# the first data loader  OLG=OLG_flag depends on this lag for the onlien e generating 
mydata_loader1 = myDataloader(Batch_size,Resample_size,Path_length,validation = validation_flag,OLG=OLG_flag)
# the second one will be a offline one for sure 
mydata_loader2 = myDataloader(Batch_size,Resample_size,Path_length,validation = validation_flag,OLG=False)
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
                #img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter


        return img1


switcher = 0
while(1):
    epoch+= 1
    if mydata_loader1.read_all_flag2 == 1 and validation_flag ==True:
        break
    #almost 900 pictures
    while(1):
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
        realB =  rendering.layers_visualized_integer_encodeing(labelv,Resample_size)

        GANmodel.update_learning_rate()    # update learning rates in the beginning of every epoch.
        GANmodel.set_input(realA,realB,labelv)         # unpack data from dataset and apply preprocessing

        if validation_flag == True :
            GANmodel.forward()
            GANmodel.error_calculation()
            D_x =  0
            G_x =  0
        else:
            GANmodel.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            D_x = GANmodel.loss_D.data.mean()
            G_x = GANmodel.loss_G.data.mean()
         
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
        

        #D_x1 = errD_real1.data.mean()
        #D_x2 = errD_real2.data.mean()
        #D_xf = errD_real_fuse.data.mean()


        #optimizerG.step()

        #save_out  = Fake
        # train with fake
        # if cv2.waitKey(12) & 0xFF == ord('q'):
        #       break 
         
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, 0, read_id, 0,
                 G_x, D_x, 0, 0, 0))
        if read_id % 2 == 0 and Visdom_flag == True:
                plotter.plot( 'cLOSS', 'cLOSS', 'cLOSS', iteration_num, D_x.cpu().detach().numpy())
                plotter.plot( 'cLOSS', 'cLOSS', 'cLOSS', iteration_num, G_x.cpu().detach().numpy())

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
         
           

            #for i in range ( len(path2)):
            #    color1 = draw_coordinates_color(color1,path2[i],i)
                 
            saveout  = GANmodel.fake_B
            
            show2 =  saveout[0,0,:,:].cpu().detach().numpy()*255 

            
            color  = numpy.zeros((show2.shape[0],show2.shape[1],3))
            color[:,:,0]  =color[:,:,1] = color[:,:,2] = show2  
         
           

            #for i in range ( len(path2)):
            #    color = draw_coordinates_color(color,path2[i],i)
                
          


            
            #show3 = numpy.append(show1,show2,axis=1) # cascade
            show4 = numpy.append(color1,color,axis=1) # cascade

            cv2.imshow('Deeplearning one',show4.astype(numpy.uint8)) 
            cv2.imwrite("D:/Deep learning/out/1out_img/Ori_seg_rec_Unet/"  +
                        str(infinite_save_id) +".jpg",show4 )
            real_label = GANmodel.real_B
            show5 =  real_label[0,0,:,:].cpu().detach().numpy()*255 
            cv2.imshow('real',show5.astype(numpy.uint8)) 

            infinite_save_id += 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
              break
    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch
    torch.save(GANmodel.netG.state_dict(), pth_save_dir+ "cGANG_epoch_"+str(epoch)+".pth")
    torch.save(GANmodel.netD.state_dict(), pth_save_dir+ "cGAND_epoch_"+str(epoch)+".pth")

    #cv2.imwrite(Save_pic_dir  + str(epoch) +".jpg", show2)
    #cv2.imwrite(pth_save_dir  + str(epoch) +".jpg", show2)
    if epoch >=5:
        epoch =0


