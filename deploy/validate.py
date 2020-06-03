import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import gan_body
import arg_parse
import imagenet
from analy import MY_ANALYSIS
from analy import Save_signal_enum
import cv2
import numpy
from image_trans import BaseTransform  
from generator_contour import Generator_Contour,Save_Contour_pkl
import matplotlib.pyplot as plt
Display_sig_flag = True
from scipy import signal 
from scipy.signal import find_peaks


import os
from dataset import myDataloader,Batch_size,Resample_size, Path_length
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Switch control for the Visdom or Not
Visdom_flag  = True 
Display_fig_flag = True
if Visdom_flag == True:
    from analy_visdom import VisdomLinePlotter
    plotter = VisdomLinePlotter(env_name='path finding training Plots')

validate_dir = "D:/PhD/trying/tradition_method/OCT/contour_vali/"
pth_save_dir = "../out/deep_contour/"
 
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
netD = gan_body._netD_8_multiscal_fusion()
#netD = gan_body._netD_Resnet()




netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)
 

criterion = nn.L1Loss()

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
#optimizerD = optim.SGD(netD.parameters(), lr=opt.lr,momentum= 0.5, weight_decay =2e-4 )


#saved_stastics = MY_ANALYSIS()
#saved_stastics=saved_stastics.read_my_signal_results()
#saved_stastics.display()

read_id =0

epoch=0
#transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
#transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num =0
mydata_loader = myDataloader (Batch_size,Resample_size,Path_length)
while(1):
    epoch+= 1
    #almost 900 pictures
    for i in os.listdir(validate_dir):
        a, b = os.path.splitext(i)
            # 如果后缀名是“.xml”就旋转related的图像
        if b == ".jpg"or b== ".JPG":
            img_path = validate_dir + a + ".jpg"
            img1 = cv2.imread(img_path)
            img_piece  =   cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            ini_H,ini_W  = img_piece.shape
            img_piece = img_piece[0:ini_H,:]
            #img_piece = cv2.medianBlur(img_piece,5)
            #long =img_piece
            long = numpy.append(img_piece,img_piece,axis=1)
            long = numpy.append(long,img_piece,axis=1)
            B_line = numpy.sum(long, axis=0)/long.shape[0]
            B_line=  signal.medfilt(B_line,7)
            B_line= numpy.convolve(B_line, numpy.ones((200,))/200, mode='valid')
            B_line  = B_line/max(B_line)
            len_b  = len(B_line)
            border1 = numpy.zeros(len_b) +1- 0.5*(1 - min(B_line))
            border2= numpy.ones(len_b)*1.1
            peaks, _ = find_peaks(B_line, height=(border1, border2),distance=30,prominence=0.1)
            start = peaks[1]
            # to search border 
            for k  in range(ini_W):
                this = B_line[start+k]
                if this <=border1[start+k]:
                    right  = start+k+100
                    break
            for k  in range(ini_W):
                this = B_line[start-k]
                if this <=border1[start-k]:
                    left  = start-k+100
                    break


            if Display_sig_flag == True:
                fig = plt.figure()
                ax = plt.axes()
                ax.plot(B_line)

                plt.plot(B_line)
                plt.plot(border1, "--", color="gray")
                plt.plot(border2, ":", color="gray")
                plt.plot(peaks, B_line[peaks], "x")
                plt.show()
            img_piece  = long[:,left:right]
            img_piece = cv2.resize(img_piece, (Resample_size,Resample_size), interpolation=cv2.INTER_AREA)
            


        #mydata_loader .read_a_batch()
        #change to 3 chanels
        np_input = numpy.zeros((1,3,Resample_size,Resample_size)) # a batch with piece num
        np_input[0,0,:,:] = img_piece - 104.0
        np_input[0,1,:,:] = img_piece - 104.0
        np_input[0,2,:,:] = img_piece - 104.0
        
  

        input = torch.from_numpy(numpy.float32(np_input)) 
        #input = input.to(device) 
        #input = torch.from_numpy(numpy.float32(mydata_loader.input_image[0,:,:,:])) 
        input = input.to(device)                
   
        #patht= torch.from_numpy(numpy.float32(mydata_loader.input_path)/71.0 )
        #patht=patht.to(device)
                
        #patht= torch.from_numpy(numpy.float32(mydata_loader.input_path[0,:])/71.0 )
        #patht=patht.to(device)
        #inputv = Variable(input)
        # using unsqueeze is import  for with out bactch situation
        #inputv = Variable(input.unsqueeze(0))

        #labelv = Variable(patht)
        #inputv = input

        #labelv = patht
        inputv = Variable(input )
        #inputv = Variable(input.unsqueeze(0))
        #patht =patht.view(-1, 1).squeeze(1)

        #labelv = Variable(patht)
        output = netD(inputv)
        output = output[0,:]
        save_out  = output
        #netD.zero_grad()
        #errD_real = criterion(output, labelv)
        #errD_real.backward()
        #D_x = errD_real.data.mean()
        #optimizerD.step()
        # train with fake
        # if cv2.waitKey(12) & 0xFF == ord('q'):
        #       break 
         
         
            #vutils.save_image(real_cpu,
            #        '%s/real_samples.png' % opt.outf,
            #        normalize=True)
            #netG.eval()
            #fake = netG(fixed_noise)
            #cv2.imwrite(Save_pic_dir  + str(i) +".jpg", mat)
            #show the result


        gray2  =   img_piece
        show1 = gray2.astype(float)
        #path2 = mydata_loader.input_path[0,:]/71*(Resample_size-2)
        #path2  = signal.resample(path2, Resample_size)

        #for i in range ( len(path2)):
        #    path2[i]= min(path2[i],Resample_size-1)
        #    path2[i]= max(path2[i],0)  
        #    show1[int(path2[i]),i]=254
             
        show2 =  gray2.astype(float)
        save_out = save_out.cpu().detach().numpy()

        save_out  = save_out  *(Resample_size)
        save_out  = signal.resample(save_out, Resample_size)

        for i in range ( len(save_out)):
            save_out[i]= min(save_out[i],Resample_size-1)
            save_out[i]= max(save_out[i],0)    
            show2[int(save_out[i]),i]=254
            #show2[int(path2[i]),i]=254


            
        #show3 = numpy.append(show1,show2,axis=1) # cascade
        cv2.imshow('Deeplearning one',show2.astype(numpy.uint8)) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    