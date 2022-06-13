import torch
import torch.nn.functional as F
from model.base_model import BaseModel
import model.networks as  networks
from test_model import layer_body_sheath_res2
# from test_model import fusion_nets_ivus
import test_model.fusion_nets_multi as fusion_nets_ivus
from test_model.fusion_nets_multi import Without_Auxiliary,Without_ExP
from test_model.loss_MTL import MTL_loss,DiceLoss
import rendering
from dataset_ivus import myDataloader,Batch_size,Resample_size, Path_length,Reverse_existence,Sep_Up_Low
from time import time
import torch.nn as nn
from torch.autograd import Variable
from databufferExcel import EXCEL_saver
#torch.autograd.set_detect_anomaly(True) # Fix for problem: RuntimeError: one of the variables needed for gradient
# computation has been modified by an inplace operation: [torch.cuda.FloatTensor [1024]] is at version 3;
# expected version 2 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient,
# with torch.autograd.set_detect_anomaly(True).
from working_dir_root import Dataset_root,Output_root
import numpy as np

# Learning rate for backbone
Coord_lr = 0.000001
Pix_lr_lambda = 10.0
EXxtens_lr_lambda = 100.0

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
            # self.criterionL1 = torch.nn.L1Loss()
            # self.criterionL1 = torch.nn.BCELoss()
            # self.criterionL1 = torch.nn.CrossEntropyLoss()
            self.criterion_Dice = DiceLoss()
            self.criterionL1 = torch.nn.L1Loss()


            # LGQ add another loss for G
            self.criterionMTL= MTL_loss(Loss ="L1") # default loss =  L1", that is used  for the Coordinates position
            # LGQ add another loss for G
            self.criterionMTL_BCE= MTL_loss(Loss ="L1") # multi_scale_cross entrofy for the existence 
            self.customeBCE =  torch.nn.BCELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.

            # self.optimizer_G = torch.optim.Adam([
            #     {'params': self.netG.Unet_back.parameters()},
            #     {'params': self.netG.backbone.parameters()},
            #     {'params':self.netG.side_branch1.  parameters()},
            #     {'params': self.netG.side_branch2.  parameters()},
            #     {'params': self.netG.side_branch3.parameters()},
            #     {'params': self.netG.low_level_encoding.parameters()},
            #     {'params': self.netG.fusion_layer.parameters()},
            #
            # ], lr=opt.lr, betas=(opt.beta1, 0.999))
            # Optimizer of the CEnet after backbone
            self.optimizer_G = torch.optim.Adam([
                # {'params': self.netG.Unet_back.parameters()},
                {'params': self.netG.backbone.parameters()},
                {'params': self.netG.side_branch1.parameters()},
                {'params': self.netG.side_branch2.parameters()},
                {'params': self.netG.side_branch3.parameters()},
                {'params': self.netG.low_level_encoding.parameters()},
                {'params': self.netG.fusion_layer.parameters()},
            ], lr=Coord_lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.SGD([
            #     # {'params': self.netG.Unet_back.parameters()},
            #     {'params': self.netG.backbone.parameters()},
            #     {'params': self.netG.side_branch1.parameters()},
            #     {'params': self.netG.side_branch2.parameters()},
            #     {'params': self.netG.side_branch3.parameters()},
            #     {'params': self.netG.low_level_encoding.parameters()},
            #     {'params': self.netG.fusion_layer.parameters()},
            # ], lr=Coord_lr, momentum=0.9)
            # Optimizer of the Unet like backbone
            self.optimizer_G_unet = None
            if self.netG.UnetBack_flag == True:  # and self.swither_G<=5:

                self.optimizer_G_unet = torch.optim.Adam([
                    {'params': self.netG.Unet_back.parameters()},
                    {'params': self.netG.pixencoding.parameters()},
                ], lr=Coord_lr, betas=(opt.beta1, 0.999))

            self.optimizer_G_f = torch.optim.Adam(self.netG.fusion_layer.  parameters(), lr=Coord_lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_1 = torch.optim.Adam(self.netG.side_branch1.parameters(), lr=Coord_lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_2 = torch.optim.Adam(self.netG.side_branch2.parameters(), lr=Coord_lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_3 = torch.optim.Adam(self.netG.side_branch3.parameters(), lr=Coord_lr, betas=(opt.beta1, 0.999))

            # The same optimization for the 
            # self.optimizer_G_e = torch.optim.Adam([
            #     {'params': self.netG.fusion_layer_exist .parameters()}
            # ], lr=Coord_lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_e = torch.optim.Adam([
                # {'params': self.netG.Unet_back.parameters()},

                {'params': self.netG.backbone_e.parameters()},
                {'params': self.netG.side_branch1e.parameters()},
                {'params': self.netG.side_branch2e.parameters()},
                {'params': self.netG.side_branch3e.parameters()},
                {'params': self.netG.fusion_layer_exist.parameters()},

            ], lr=Coord_lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G_e= torch.optim.SGD([
            #     {'params': self.netG.backbone_e.parameters()},
            #     {'params': self.netG.side_branch1e.parameters()},
            #     {'params': self.netG.side_branch2e.parameters()},
            #     {'params': self.netG.side_branch3e.parameters()},
            #     {'params': self.netG.fusion_layer_exist.parameters()},
            # ], lr=Coord_lr, momentum=0.9)
            self.optimizer_E_f = torch.optim.Adam(self.netE.fusion_layer.  parameters(), lr=Coord_lr, betas=(opt.beta1, 0.999))
            self.optimizer_E_1 = torch.optim.Adam(self.netE.side_branch1.parameters(), lr=Coord_lr, betas=(opt.beta1, 0.999))
            self.optimizer_E_2 = torch.optim.Adam(self.netE.side_branch2.parameters(), lr=Coord_lr, betas=(opt.beta1, 0.999))
            self.optimizer_E_3 = torch.optim.Adam(self.netE.side_branch3.parameters(), lr=Coord_lr, betas=(opt.beta1, 0.999))



            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=Coord_lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_E)
            self.optimizers.append(self.optimizer_D)
        # self.validation_init()
        self.bw_cnt =0
        self.displayloss1=0
        self.displayloss2=0
        self.displayloss3=0
        self.loss_G_L1 =torch.tensor(0,dtype=torch.float)
        self.loss_G_L2 =torch.tensor(0,dtype=torch.float)
        self.loss_G_L3 =torch.tensor(0,dtype=torch.float)
        self.loss_pix = torch.tensor(0,dtype=torch.float)
        self.loss_G = torch.tensor(0,dtype=torch.float)
        self.lossEa = torch.tensor(0,dtype=torch.float)

        self.metrics_saver = EXCEL_saver(10) # 8 values
        self.switcher = 0  # used to switch gradient between different part of the nets
        self.swither_G = 0 # used to optimize the back bone or not



    def set_input(self, realA,pathes,exis_v,input_img):

        self.real_A = realA.to(self.device)

        #self.real_B = realB.to(self.device)
        # this for the old OCT that has no sepearation on the Upper and lower boundaries

        if Sep_Up_Low == False:
            self.real_B=rendering.layers_visualized_integer_encodeing(pathes,Resample_size) # this way render it as semantic map
        else:
            self.real_B=rendering.layers_visualized_integer_encodeing_full(pathes,exis_v,Resample_size,Reverse_existence) # this is a way to encode it as boundary (very spars)

        # self.real_B=rendering.boundary_visualized_integer_encodeing(pathes,Resample_size) # this is a way to encode it as boundary (very spars)
        # TODO: this  is the new way that separate the upper and lower, need to uniform all the label encoding

        # self.real_B=rendering.layers_visualized_integer_encodeing_full(pathes,exis_v,Resample_size,Reverse_existence) # this is a way to encode it as boundary (very spars)
        self.real_B_one_hot = rendering.integer2onehot(self.real_B)
        # self.real_B_one_hot=rendering.layers_visualized_OneHot_encodeing(pathes,Resample_size)

        # LGQ add real path as creterioa for G
        self.real_pathes = pathes
        self.real_exv  =  exis_v 
        self.input_G  = input_img
        self.input_E  = input_img
    def set_GE_input(self,img):
        self.input_G  = img 
        self.input_E = img
 
    def mask_with_exist(self):
        if Reverse_existence == True:
            exvP =    self.out_exis_v0 <0.7
            #exvT =    self.real_exv<0.7
        else:
            exvP = self.out_exis_v0 > 0.7
            #exvT = self.real_exv > 0.7
        self.out_pathes[0] = self.out_pathes[0] * exvP + (~exvP) # reverse the mask
        #self.real_pathes = self.real_pathes * exvT + (~exvT)

    def forward(self,validation_flag,one_hot_render = True):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        start_time = time()
        #self.out_pathes = self.netG(self.input_G) # coordinates encoding
        # this version all perdiction are in one net(shared front layer)
        self.out_pathes0, self.out_pathes1,self.out_pathes2,self.out_pathes3,self.pix_wise,self.out_exis_v0 = self.netG(self.input_G)
        self.out_pathes=[self.out_pathes0, self.out_pathes1,self.out_pathes2,self.out_pathes3]
        self.out_exis_vs = [self.out_exis_v0,self.out_exis_v0,self.out_exis_v0,self.out_exis_v0]

        test_time_point = time()
        print (" all test point time is [%f] " % ( test_time_point - start_time))

        # use the same fusion method to predict the 
        # self.out_exis_v0, self.out_exis_v1,self.out_exis_v2,self.out_exis_v3,_ = self.netE(self.input_E)
        if (validation_flag == True):
             self.mask_with_exist()
        if (one_hot_render == True):



            #TODO: this  is the old way that without separating the upper and lower
            if Sep_Up_Low == False:
                self.fake_B=  rendering.layers_visualized_integer_encodeing (self.out_pathes[0],Resample_size) # encode as semantic map
            # self.fake_B=  rendering.boundary_visualized_integer_encodeing(self.out_pathes[0],Resample_size) # encode as boundary
            else:
            # TODO: this  is the new way that separate the upper and lower, need to uniform all the label encoding
                self.fake_B=  rendering.layers_visualized_integer_encodeing_full(self.out_pathes[0], self. out_exis_vs[0],Resample_size,Reverse_existence) # encode as boundary



            # self.fake_B_1_hot = rendering.layers_visualized_OneHot_encodeing(self.out_pathes[0],Resample_size)
            self.fake_B_1_hot = rendering.integer2onehot(self.fake_B )
        if self.pix_wise is None:
            self.pix_wise = self.fake_B
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
        # self.swither_G = 9
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        # self.optimizer_G_unet.zero_grad()  # udpate G's weights

        if self.netG.UnetBack_flag == True :#and self.swither_G<=5:
            self.optimizer_G_unet.zero_grad()  # udpate G's weights

            self.swither_G+=1
            self.set_requires_grad(self.netG, True)
            # self.set_requires_grad(self.netG.side_branch1, False)
            # self.set_requires_grad(self.netG.side_branch2, False)
            # self.set_requires_grad(self.netG.side_branch3, False)
            # self.set_requires_grad(self.netG.fusion_layer, False)


            # pix_loss1 = self.criterion_Dice (self.pix_wise, self.real_B)

            background = (self.real_B_one_hot < 0.1)
            Nonebackground = (self.real_B_one_hot > 0.1)


            backgroud_beta = (torch.sum(Nonebackground,dim=[2,3]) + 0.0001) / (torch.sum(background,dim=[2,3]) + torch.sum(Nonebackground,dim=[2,3]) + 0.0001)
            backgroud_mask = background
            B,C,H,W  = backgroud_mask.size()

            # backgroud_mask[0:B,0:C,:,:] = backgroud_beta
            backgroud_beta  =  torch.unsqueeze(backgroud_beta, 2)
            backgroud_beta  =  torch.unsqueeze(backgroud_beta, 3)

            backgroud_mask = background[:, :, 0:H, 0:W] * backgroud_beta + Nonebackground[:, :, 0:H, 0:W] * (1.0-backgroud_beta)

            # backgroud_mask = backgroud_mask + Nonebackground*1.0


            # pix_loss2 = self.criterionL1 (self.pix_wise *backgroud_mask, self.real_B_one_hot*backgroud_mask)
            # pix_loss2 = self.criterionL1(self.pix_wise ,self.real_B)
            pix_loss2 = self.customeBCE(self.pix_wise ,self.real_B)

            # pix_loss = self.criterionL1 (self.pix_wise *(self.real_B>0.1+3)/4.0, self.real_B)
            # pix_loss = self.criterionL1 (self.pix_wise *(self.real_B>0.1+3)/4.0, self.real_B)
            self.loss_pix =  Pix_lr_lambda * pix_loss2

            # self.loss_pix = 1000*(0.5 * pix_loss1 + 0.5 * pix_loss2)
            self.loss_pix.backward(retain_graph=True)
            self.optimizer_G_unet.step()  # udpate G's weights

            self.out_pathes0, self.out_pathes1, self.out_pathes2, self.out_pathes3, self.pix_wise, self.out_exis_v0 = self.netG(
                self.input_G)
            self.out_pathes = [self.out_pathes0, self.out_pathes1, self.out_pathes2, self.out_pathes3]
            self.out_exis_vs = [self.out_exis_v0, self.out_exis_v0, self.out_exis_v0, self.out_exis_v0]

        # else:
        self.swither_G+=1
        self.set_requires_grad(self.netG, True)
        # self.set_requires_grad(self.netG.Unet_back, False)
        # self.set_requires_grad(self.netG.pixencoding, False)

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        #self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        #LGQ special fusion loss
        #self.loss=self.criterionMTL.multi_loss([self.out_pathes0],self.real_pathes)
        # self.loss=self.criterionMTL.multi_loss(self.out_pathes,self.real_pathes)
        # 3 imput , also rely on the existence vector


        # self.loss_G = ( 1.0*self.loss[0]  + 0.01*self.loss[1] + 0.001*self.loss[2] + 0.001*self.loss[3])
        # TODO: For last training, use line below and comment the one above
        # self.loss_G = self.loss[0]


        # TODO: Enable at the "end"/fine-tuning of training
        # self.loss=self.criterionMTL.multi_loss (self.out_pathes,self.real_pathes ) #
        if Without_ExP ==False:
            self.loss = self.criterionMTL.multi_loss_contour_exist(self.out_pathes, self.real_pathes, self.real_exv,
                                                               Reverse_existence)  #
        else:
            self.loss = self.criterionMTL.multi_loss(self.out_pathes,self.real_pathes)  #
        if Without_Auxiliary == False:
            self.loss_G = ( 1.0*self.loss[0]  + 0.001*self.loss[1] + 0.001*self.loss[2] + 0.001*self.loss[3])
        else:
            self.loss_G = self.loss[0]

        self.loss_G.backward(retain_graph=True)
        self.optimizer_G.step()  # udpate G's weights

        #self.optimizer_G.step()             # udpate G's weights
    def backward_E(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.out_pathes0, self.out_pathes1, self.out_pathes2, self.out_pathes3, self.pix_wise, self.out_exis_v0 = self.netG(
            self.input_G)
        self.out_pathes = [self.out_pathes0, self.out_pathes1, self.out_pathes2, self.out_pathes3]
        self.out_exis_vs = [self.out_exis_v0, self.out_exis_v0, self.out_exis_v0, self.out_exis_v0]
        self.optimizer_G_e .zero_grad()        # set G's gradients to zero
        #self.set_requires_grad(self.netG, False)       

        self.set_requires_grad(self.netG, True)
        # use BEC for the existence vectors
        #self.loss=self.criterionMTL_BCE.multi_loss(self.out_exis_vs,self.real_exv)
        # self.lossE=self.criterionMTL_BCE.multi_loss(self.out_exis_vs,self.real_exv)
        # self.lossE=self.criterion_Dice(self.out_exis_vs[0],self.real_exv[0])



        #self.loss_G_L1 =( 1.0*loss[0]  + 0.5*loss[1] + 0.1*loss[2] + 0.2*loss[3])*self.opt.lambda_L1
        #self.loss_G_L1 =( 1.0*loss[0]  + 0.02*loss[1] + 0.02*loss[2]+ 0.02*loss[3]+ 0.02*loss[4]+ 0.02*loss[5])*self.opt.lambda_L1
        #self.loss_G_L1_2 = 0.5*loss[0] 
        #self.loss_G_L1 =( 1.0*loss[0]  +   0.01*loss[1] + 0.01*loss[2] +0.01*loss[3]  )*self.opt.lambda_L1
        # self.loss_G_L0 =( self.loss[0]    )*self.opt.lambda_L1
        #self.loss_G_L0 = (self.loss[0])
        # self.loss_G =0* self.loss_G_GAN + self.loss_G_L0
        # TODO: Enable at the beginning of the training
        # self.lossEa =   ( 1.0*self.lossE[0]  + 0.01*self.lossE[1] + 0.01*self.lossE[2] + 0.01*self.lossE[3])
        # self.lossEa = self.lossE[0]

        # TODO: Enable at the "end" of training
        #  (sacrifice accuracy of higher resolution branch for overall better output)
        B, C, W = self.real_exv.size()
        # self.real_exv = self.real_exv . type(torch.LongTensor)
        self.lossEa = EXxtens_lr_lambda* self.customeBCE (self.out_exis_vs[0] ,self.real_exv )

        self.lossEa.backward(retain_graph=True )
        self.optimizer_G_e .step()             # udpate G's weights

        #self.optimizer_G.step()             # udpate G's weights


    def optimize_parameters(self,validation_flag):
        self.forward(validation_flag)                   # compute fake images: G(A) # seperatee the for
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
        self.backward_G()  # calculate graidents for G
        self.backward_E()  # calculate graidents for E


        # if (self.switcher<=5):
        #     self.backward_G()                   # calculate graidents for G
        # # else:
        #     self.backward_E()                   # calculate graidents for E

        self.switcher += 1
        if (self.switcher>=11):
            self.switcher = 0

        self.displayloss0 = self.loss_G. data.mean()
        self.displayloss1 = self.loss_pix. data.mean()
        self.displayloss2 = self.lossEa. data.mean()
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




