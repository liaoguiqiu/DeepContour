import torch
from model.base_model import BaseModel
import model.networks as  networks
from test_model import layer_body_sheath_res2
from test_model import fusion_nets_ivus
from test_model.loss_MTL import MTL_loss
import rendering
from dataset_sheath import myDataloader,Batch_size,Resample_size, Path_length
from time import time
import torch.nn as nn
from torch.autograd import Variable
from databufferExcel import EXCEL_saver
#torch.autograd.set_detect_anomaly(True) # Fix for problem: RuntimeError: one of the variables needed for gradient
# computation has been modified by an inplace operation: [torch.cuda.FloatTensor [1024]] is at version 3;
# expected version 2 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient,
# with torch.autograd.set_detect_anomaly(True).


class Pix2LineModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        #LGQ
        #parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            #LGQ
            #parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)

        # LGQ here I change the generator to my line encoding
        #self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG  = fusion_nets_ivus._2layerFusionNets_() # generate the coodinates
        self.netE  = fusion_nets_ivus._2layerFusionNets_(classfy = False) # generate the exsitence lalel

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # LGQ add another loss for G
            self.criterionMTL= MTL_loss(Loss ="L1") # default loss =  L1", that is used  for the Coordinates position
            # LGQ add another loss for G
            self.criterionMTL_BCE= MTL_loss(Loss ="L1") # multi_scale_cross entrofy for the existence 

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_f = torch.optim.Adam(self.netG.fusion_layer.  parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_1 = torch.optim.Adam(self.netG.side_branch1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_2 = torch.optim.Adam(self.netG.side_branch2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_3 = torch.optim.Adam(self.netG.side_branch3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            # The same optimization for the 
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_E_f = torch.optim.Adam(self.netE.fusion_layer.  parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_E_1 = torch.optim.Adam(self.netE.side_branch1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_E_2 = torch.optim.Adam(self.netE.side_branch2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_E_3 = torch.optim.Adam(self.netE.side_branch3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))



            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_E)
            self.optimizers.append(self.optimizer_D)

        self.validation_init()
        self.bw_cnt =0
        self.displayloss1=0
        self.displayloss2=0
        self.displayloss3=0
        self.loss_G_L1 =torch.tensor(0,dtype=torch.float)
        self.loss_G_L2 =torch.tensor(0,dtype=torch.float)
        self.loss_G_L3 =torch.tensor(0,dtype=torch.float)
        self.metrics_saver = EXCEL_saver(8) # 8 values
         

    def validation_init(self):
        self.L1 = 0
        self.L2 = 0
        self.J1 = 0
        self.J2 = 0
        self.J3 = 0

        self.D1 = 0
        self.D2 = 0
        self.D3 = 0
        self.validation_cnt =0
    def error_calculation(self): 
        #average Jaccard index J
        def cal_J(true,predict):
            AnB = true*predict # assume that the lable are all binary
            AuB = true+predict
            AuB=torch.clamp(AuB, 0, 1)
            s  = 0.0001
            this_j = (torch.sum(AnB)+s)/(torch.sum(AuB)+s)
            return this_j
        # dice cofefficient
        def cal_D(true,predict): 
            AB = true*predict # assume that the lable are all binary
            #AuB = true+predict
            #AuB=torch.clamp(AuB, 0, 1)
            s  = 0.0001

            this_d = (2 * torch.sum(AB)+s)/(torch.sum(true) + torch.sum(predict)+s)
            return this_d
        def cal_L (true,predct):  # the L1 distance of one contour
        # calculate error
            true=torch.clamp(true, 0, 1)
            predct=torch.clamp(predct, 0, 1)


            error = torch.abs(true-predct)
            l = error.size()
            x= torch.sum(error)/l[0]
            return x
        self.set_requires_grad(self.netG, False)  # D requires no gradients when optimizing G

        self.validation_cnt += 1

        #loss = self.criterionMTL.multi_loss(self.out_pathes,self.real_pathes)
        #self.error = 1.0*loss[0] 
        #out_pathes[fusion_predcition][batch 0, contour index,:]
        cutedge = 30
        self.L1 = cal_L(self.out_pathes[0][0,0,cutedge:Resample_size-cutedge],self.real_pathes[0,0,cutedge:Resample_size-cutedge]) * Resample_size
        self.L2 = cal_L(self.out_pathes[0][0,1,cutedge:Resample_size-cutedge],self.real_pathes[0,1,cutedge:Resample_size-cutedge]) * Resample_size

        print (" L1 =  "  + str(self.L1))
        print (" L2 =  "  + str(self.L2))

        # calculate J (IOU insetion portion)
        real_b_hot = rendering.layers_visualized_OneHot_encodeing  (self.real_pathes,Resample_size) 
        fake_b_hot = self.fake_B_1_hot 
        # this is the format of hot map
        #out  = torch.zeros([bz,3, H,W], dtype=torch.float)
        self.J1 = cal_J(real_b_hot[0,0,:,cutedge:Resample_size-cutedge],fake_b_hot[0,0,:,cutedge:Resample_size-cutedge])
        self.J2 = cal_J(real_b_hot[0,1,:,cutedge:Resample_size-cutedge],fake_b_hot[0,1,:,cutedge:Resample_size-cutedge])
        self.J3 = cal_J(real_b_hot[0,2,:,cutedge:Resample_size-cutedge],fake_b_hot[0,2,:,cutedge:Resample_size-cutedge])
        print (" J1 =  "  + str(self.J1 ))
        print (" J2 =  "  + str(self.J2 ))
        print (" J3 =  "  + str(self.J3 ))



        self.D1 = cal_D(real_b_hot[0,0,:,cutedge:Resample_size-cutedge],fake_b_hot[0,0,:,cutedge:Resample_size-cutedge])
        self.D2 = cal_D(real_b_hot[0,1,:,cutedge:Resample_size-cutedge],fake_b_hot[0,1,:,cutedge:Resample_size-cutedge])
        self.D3 = cal_D(real_b_hot[0,2,:,cutedge:Resample_size-cutedge],fake_b_hot[0,2,:,cutedge:Resample_size-cutedge])
        print (" D1 =  "  + str(self.D1 ))
        print (" D2 =  "  + str(self.D2 ))
        print (" D3 =  "  + str(self.D3 ))
        vector = [self.L1,self.L2,self.J1, self.J2,self.J3,self.D1,self.D2,self.D3]
        vector = torch.stack(vector)
        vector= vector.cpu().detach().numpy()
        save_dir = "D:/Deep learning/out/1Excel/"
        self.metrics_saver.append_save(vector,save_dir)

    def set_input(self, realA,pathes,exis_v,input_img):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """


        # LGQ modify it as one way 
        #AtoB = self.opt.direction == 'AtoB'
        #self.real_A = input['A' if AtoB else 'B'].to(self.device)
        #self.real_B = input['B' if AtoB else 'A'].to(self.device)
        #self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # LGQ modify it as one way 
        self.real_A = realA.to(self.device)

        #self.real_B = realB.to(self.device)
        self.real_B=rendering.layers_visualized_integer_encodeing(pathes,Resample_size)
        self.real_B_one_hot=rendering.layers_visualized_OneHot_encodeing(pathes,Resample_size)

        # LGQ add real path as creterioa for G
        self.real_pathes = pathes
        self.real_exv  =  exis_v 
        self.input_G  = input_img
        self.input_E  = input_img
    def set_GE_input(self,img):
        self.input_G  = img 
        self.input_E = img
 


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        start_time = time()
        #self.out_pathes = self.netG(self.input_G) # coordinates encoding
        f1,self.out_pathes1,self.path_long1 = self.netG.side_branch1 (self.input_G) # coordinates encoding
        
        f2,self.out_pathes2,self.path_long2 = self.netG.side_branch2 (self.input_G) # coordinates encoding
        f3,self.out_pathes3,self.path_long3 = self.netG.side_branch3 (self.input_G) # coordinates encoding
        self.out_pathes0 =self.netG.fuse_forward(f1,f2,f3)
        test_time_point = time()
        print (" all test point time is [%f] " % ( test_time_point - start_time))
        self.out_pathes = [self.out_pathes0,self.out_pathes1,self.out_pathes2,self.out_pathes3]

        # use the same fusion method to predict the 
        f_e1,self.out_exis_v1,self.exis_long1 = self.netE.side_branch1 (self.input_E) # coordinates encoding     
        f_e2,self.out_exis_v2,self.exis_long2 = self.netE.side_branch2 (self.input_E) # coordinates encoding
        f_e3,self.out_exis_v3,self.exis_long3 = self.netE.side_branch3 (self.input_E) # coordinates encoding
        self.out_exis_v0 =self.netE.fuse_forward(f_e1,f_e2,f_e3)
        self.out_exis_vs = [self.out_exis_v0,self.out_exis_v1,self.out_exis_v2,self.out_exis_v3]

        self.fake_B=  rendering.layers_visualized_integer_encodeing (self.out_pathes0,Resample_size)
        self.fake_B_1_hot = rendering.layers_visualized_OneHot_encodeing(self.out_pathes0,Resample_size)
        #self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        self.set_requires_grad(self.netG, False)       
        self.optimizer_D.zero_grad()        # set G's gradients to zero

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.optimizer_G.zero_grad()        # set G's gradients to zero

        #self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG, False)  # D requires no gradients when optimizing G
        # just remain the upsample fusion parameter to optimization 
        # self.set_requires_grad(self.netG.side_branch1, False)  # D requires no gradients when optimizing G
        # self.set_requires_grad(self.netG.side_branch2, False)  # D requires no gradients when optimizing G
        # self.set_requires_grad(self.netG.side_branch3, False)  # D requires no gradients when optimizing G
        # self.set_requires_grad(self.netG.side_branch1.fullout, True)  # D requires no gradients when optimizing G
        # self.set_requires_grad(self.netG.side_branch2.fullout, True)  # D requires no gradients when optimizing G
        # self.set_requires_grad(self.netG.side_branch3.fullout, True)  # D requires no gradients when optimizing G
        # self.set_requires_grad(self.netG.fusion_layer , True)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netE, False)  # enable backprop for D
        
        self.set_requires_grad(self.netG, True)


        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        #self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        #LGQ special fusion loss
        #self.loss=self.criterionMTL.multi_loss([self.out_pathes0],self.real_pathes)
        #self.loss=self.criterionMTL.multi_loss(self.out_pathes,self.real_pathes)
        # 3 imput , also rely on the existence vector
        self.loss=self.criterionMTL.multi_loss_contour_exist(self.out_pathes,self.real_pathes, self.out_exis_vs) # 

        

        #self.loss_G_L1 =( 1.0*loss[0]  + 0.5*loss[1] + 0.1*loss[2] + 0.2*loss[3])*self.opt.lambda_L1
        #self.loss_G_L1 =( 1.0*loss[0]  + 0.02*loss[1] + 0.02*loss[2]+ 0.02*loss[3]+ 0.02*loss[4]+ 0.02*loss[5])*self.opt.lambda_L1
        #self.loss_G_L1_2 = 0.5*loss[0] 
        #self.loss_G_L1 =( 1.0*loss[0]  +   0.01*loss[1] + 0.01*loss[2] +0.01*loss[3]  )*self.opt.lambda_L1
        # self.loss_G_L0 =( self.loss[0]    )*self.opt.lambda_L1
        #self.loss_G_L0 = (self.loss[0])
        # self.loss_G =0* self.loss_G_GAN + self.loss_G_L0
        self.loss_G =   ( 1.0*self.loss[0]  + 0.1*self.loss[1] + 0.01*self.loss[2] + 0.01*self.loss[3])

        self.loss_G.backward( )
        #self.optimizer_G.step()             # udpate G's weights
        self.optimizer_G.step()             # udpate G's weights
    def backward_E(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.optimizer_E.zero_grad()        # set G's gradients to zero
        #self.set_requires_grad(self.netG, False)       

        self.set_requires_grad(self.netE, True)       
        # use BEC for the existence vectors
        #self.loss=self.criterionMTL_BCE.multi_loss(self.out_exis_vs,self.real_exv)
        self.lossE=self.criterionMTL_BCE.multi_loss(self.out_exis_vs,self.real_exv)


        #self.loss_G_L1 =( 1.0*loss[0]  + 0.5*loss[1] + 0.1*loss[2] + 0.2*loss[3])*self.opt.lambda_L1
        #self.loss_G_L1 =( 1.0*loss[0]  + 0.02*loss[1] + 0.02*loss[2]+ 0.02*loss[3]+ 0.02*loss[4]+ 0.02*loss[5])*self.opt.lambda_L1
        #self.loss_G_L1_2 = 0.5*loss[0] 
        #self.loss_G_L1 =( 1.0*loss[0]  +   0.01*loss[1] + 0.01*loss[2] +0.01*loss[3]  )*self.opt.lambda_L1
        # self.loss_G_L0 =( self.loss[0]    )*self.opt.lambda_L1
        #self.loss_G_L0 = (self.loss[0])
        # self.loss_G =0* self.loss_G_GAN + self.loss_G_L0
        self.lossEa =   ( 1.0*self.lossE[0]  + 0.1*self.lossE[1] + 0.01*self.lossE[2] + 0.01*self.lossE[3])

        self.lossEa.backward( )
        #self.optimizer_G.step()             # udpate G's weights
        self.optimizer_E.step()             # udpate G's weights
      
    def backward_G_1(self):
        self.optimizer_G.zero_grad()        # set G's gradients to zero
 
        # First, G(A) should fake the discriminator
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)  # D requires no gradients when optimizing G

        #self.set_requires_grad(self.netG.side_branch1, True)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch2, False)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch3, False)  # D requires no gradients when optimizing G
        
        loss1 = self.criterionMTL.multi_loss([self.out_pathes1],self.real_pathes) 
        #self.loss_G_L1 = Variable( loss1[0],requires_grad=True)
        self.loss_G_L1 =   loss1[0] 

          
        self.loss_G_L1.backward(retain_graph=True)
        self.optimizer_G_1.step()             # udpate G's weights
    def backward_G_2(self):
        self.optimizer_G.zero_grad()        # set G's gradients to zero
 
        # First, G(A) should fake the discriminator
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)  # D requires no gradients when optimizing G

        #self.set_requires_grad(self.netG.side_branch1, False)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch2, True)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch3, False)  # D requires no gradients when optimizing G
         
        loss2 = self.criterionMTL.multi_loss([self.out_pathes2],self.real_pathes) 
        
        self.loss_G_L2 =   loss2[0] 

          
        self.loss_G_L2.backward(retain_graph=True)
        self.optimizer_G_2.step()             # udpate G's weights
    def backward_G_3(self):

        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.optimizer_G.zero_grad()        # set G's gradients to zero

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)  # D requires no gradients when optimizing G

        #self.set_requires_grad(self.netG.side_branch1, False)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch2, False)  # D requires no gradients when optimizing G
        #self.set_requires_grad(self.netG.side_branch3, True)  # D requires no gradients when optimizing G
         
        loss3 = self.criterionMTL.multi_loss([self.out_pathes3],self.real_pathes) 
        
        self.loss_G_L3 =  loss3[0] 
        #self.loss_G_L3.backward(retain_graph=True)
        self.loss_G_L3.backward( retain_graph=True)

        self.optimizer_G_3.step()             # udpate G's weights

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A) # seperatee the for
        # update D
        #self.set_requires_grad(self.netD, True)  # enable backprop for D
        #self.optimizer_D.zero_grad()     # set D's gradients to zero
        #self.backward_D()                # calculate gradients for D
        #self.optimizer_D.step()          # update D's weights
        # update G


        # self.backward_G_1()                   # calculate graidents for G
        #
        # self.backward_G_2()                   # calculate graidents for G
        #
        # self.backward_G_3()                   # calculate graidents for G

        self.backward_G()                   # calculate graidents for G
        #self.backward_E()                   # calculate graidents for E


        self.displayloss0 = self.loss_G. data.mean()
        self.displayloss1 = self.loss_G. data.mean()
        self.displayloss2 = self.loss_G. data.mean()
        self.displayloss3 = self.loss_G. data.mean()

        self.displaylossE0 = self.loss_G. data.mean()
      
        #if self.  bw_cnt %2 ==0:
        #   self.backward_G()                   # calculate graidents for G
        #else:  
        #    self.backward_G_1()                   # calculate graidents for G
        #    self.backward_G_2()                   # calculate graidents for G
        #    self.backward_G_3()                   # calculate graidents for G
        #    self.displayloss = self.loss_G_L2. data.mean()
        #self.   bw_cnt +=1               # calculate graidents for G
        #if self.   bw_cnt >100:
        #    self.   bw_cnt =0




