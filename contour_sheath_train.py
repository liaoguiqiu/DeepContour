# the run py script for sheal and contour detection, 5th October 2020 update
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import layer_body_sheath
import arg_parse
import cv2
import numpy
from generator_contour import Generator_Contour,Save_Contour_pkl,Communicate,Generator_Contour_layers,Generator_Contour_sheath


validation_flag =False
import os
from dataset_sheath import myDataloader,Batch_size,Resample_size, Path_length
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Switch control for the Visdom or Not
Visdom_flag  = False 
Display_fig_flag = True
Continue_flag = True

if Visdom_flag == True:
    from analy_visdom import VisdomLinePlotter
    plotter = VisdomLinePlotter(env_name='path finding training Plots')


pth_save_dir = "../out/deep_sheath/"
 
if not os.path.exists(pth_save_dir):
    os.makedirs(pth_save_dir)
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
#netD = gan_body._netD_Resnet()




netD.apply(weights_init)
if Continue_flag == True:
    netD.load_state_dict(torch.load('../out/deep_sheath/netD_epoch_3.pth'))
print(netD)
 

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
mydata_loader = myDataloader (Batch_size,Resample_size,Path_length)
test_data_loader =  myDataloader (Batch_size,Resample_size,Path_length,validation=True)

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
def read_transform(data_loader):
    #change to 3 chanels
    ini_input = data_loader.input_image
    np_input = numpy.append(ini_input,ini_input,axis=1)
    np_input = numpy.append(np_input,ini_input,axis=1)

    input = torch.from_numpy(numpy.float32(np_input)) 
    #input = input.to(device) 
    #input = torch.from_numpy(numpy.float32(mydata_loader.input_image[0,:,:,:])) 
    input = input.to(device)                
   
    patht= torch.from_numpy(numpy.float32(data_loader.input_path)/Resample_size )
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
    return inputv,labelv
def display_prediction(mydata_loader,save_out):
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
        color = draw_coordinates_color(color,save_out[i],i)
                
          


            
    #show3 = numpy.append(show1,show2,axis=1) # cascade
    show4 = numpy.append(color1,color,axis=1) # cascade

    cv2.imshow('Deeplearning one',show4.astype(numpy.uint8)) 

while(1):
    epoch+= 1
    #almost 900 pictures
    while(1):
        iteration_num +=1
        read_id+=1
        if (mydata_loader.read_all_flag ==1):
            read_id =0
            mydata_loader.read_all_flag =0
            break
        if (test_data_loader.read_all_flag ==1):
           
            test_data_loader.read_all_flag =0
            

        mydata_loader .read_a_batch()
        test_data_loader.read_a_batch()
        
        # just test the first boundary effect 
        #labelv  = labelv[:,0,:]

        inputv,labelv = read_transform(mydata_loader)

        outputall = netD(inputv)
        #outputall =  outputall[:,:,0,:]
        output = outputall[0].view(Batch_size,netD.layer_num,Path_length).squeeze(1)
        output1 = outputall[1].view(Batch_size,netD.layer_num,Path_length).squeeze(1)
        output2 = outputall[2].view(Batch_size,netD.layer_num,Path_length).squeeze(1)
        #output = outputall[0]  
        #output1 = outputall[1] 
        #output2 = outputall[2]  
        #output


        netD.zero_grad()
        errD_real = criterion(output, labelv)
        errD_real1 = criterion(output1, labelv)
        errD_real2 = criterion(output2, labelv)
        #errD_real_fuse = 1.0*(errD_real+  0.1*errD_real1 +  0.1*errD_real2)
        errD_real_fuse = 1.0*(errD_real )


        #errD_real1.backward()errD_real = criterion(output, labelv)
        #errD_real1 = criterion(output1, labelv)
        #errD_real2 = criterion(output22, labelv)
        #errD_real_fuse =   0.1*errD_real1 +  0.1*errD_real2

        errD_real_fuse.backward()

        
        #errD_real.backward()
        D_x = errD_real.data.mean()
        D_x1 = errD_real1.data.mean()
        D_x2 = errD_real2.data.mean()
        D_xf = errD_real_fuse.data.mean()


        optimizerD.step()

        save_out  = output
        if validation_flag==True:
            inputvt,labelvt = read_transform(test_data_loader)

            outputallt = netD(inputvt)
            #outputall =  outputall[:,:,0,:]
            outputt = outputallt[0].view(Batch_size,netD.layer_num,Path_length).squeeze(1)
            err_validation= criterion(outputt, labelvt)
            save_test  = outputt


        # train with fake
        # if cv2.waitKey(12) & 0xFF == ord('q'):
        #       break 
         
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, 0, read_id, 0,
                 errD_real.data, D_x, 0, 0, 0))
        if read_id % 2 == 0 and Visdom_flag == True:

            plotter.plot( 'cLOSS', 'cLOSS', 'cLOSS', iteration_num, D_x.cpu().detach().numpy())
            plotter.plot( 'cLOSS1', 'cLOSS1', 'cLOSS1', iteration_num, D_x1.cpu().detach().numpy())
            plotter.plot( 'cLOSS12', 'cLOSS2', 'cLOSS2', iteration_num, D_x2.cpu().detach().numpy())
            plotter.plot( 'cLOSS_f', 'cLOSSf', 'cLOSSf', iteration_num, D_xf.cpu().detach().numpy())
        if read_id % 1 == 0 and Display_fig_flag== True:
            #vutils.save_image(real_cpu,
            #        '%s/real_samples.png' % opt.outf,
            #        normalize=True)
            #netG.eval()
            #fake = netG(fixed_noise)
            #cv2.imwrite(Save_pic_dir  + str(i) +".jpg", mat)
            #show the result


            
            display_prediction(mydata_loader,save_out)
            #display_prediction(test_data_loader,save_test)

            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch
    torch.save(netD.state_dict(), pth_save_dir+ "netD_epoch_"+str(epoch)+".pth")
    #cv2.imwrite(Save_pic_dir  + str(epoch) +".jpg", show2)
    #cv2.imwrite(pth_save_dir  + str(epoch) +".jpg", show2)
    if epoch >=5:
        epoch =0

