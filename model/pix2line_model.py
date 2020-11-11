import torch
from model.base_model import BaseModel
import model.networks as  networks
from test_model import layer_body_sheath_res2
from test_model import fusion_nets
from test_model.loss_MTL import MTL_loss
import rendering
from dataset_sheath import myDataloader,Batch_size,Resample_size, Path_length
from time import time


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
        self.netG  = fusion_nets._2layerFusionNets_()
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # LGQ add another loss for G
            self.criterionMTL= MTL_loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.validation_init()
    def validation_init(self):
        self.J1 = 0
        self.J2 = 0
        self.J3 = 0

        self.D1 = 0
        self.D2 = 0
        self.D3 = 0
        self.validation_cnt =0

    def set_input(self, realA,pathes,inputG):
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
        # LGQ add real path as creterioa for G
        self.real_pathes = pathes
        self.input_G  = inputG
    def set_G_input(self,input_G):
        self.input_G  = input_G 
 


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        start_time = time()

        self.out_pathes = self.netG(self.input_G) # coordinates encoding
        test_time_point = time()
        print (" all test point time is [%f] " % ( test_time_point - start_time))

        self.fake_B=  rendering.layers_visualized_integer_encodeing (self.out_pathes[0],Resample_size) 
        self.fake_B_1_hot = rendering.layers_visualized_OneHot_encodeing  (self.out_pathes[0],Resample_size) 
        #self.fake_B = self.netG(self.real_A)  # G(A)
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
        # calculate error
        loss = self.criterionMTL.multi_loss(self.out_pathes,self.real_pathes)
        self.error = 1.0*loss[0] 
        # calculate J (IOU insetion portion)
        real_b_hot = rendering.layers_visualized_OneHot_encodeing  (self.real_pathes,Resample_size) 
        fake_b_hot = self.fake_B_1_hot 
        # this is the format of hot map
        #out  = torch.zeros([bz,3, H,W], dtype=torch.float)
        self.validation_cnt += 1
        self.J1 += cal_J(real_b_hot[0,0,:,:],fake_b_hot[0,0,:,:])
        self.J2 += cal_J(real_b_hot[0,1,:,:],fake_b_hot[0,1,:,:])
        self.J3 += cal_J(real_b_hot[0,2,:,:],fake_b_hot[0,2,:,:])
        print (" J1 =  "  + str(self.J1/self.validation_cnt))
        print (" J2 =  "  + str(self.J2/self.validation_cnt))
        print (" J3 =  "  + str(self.J3/self.validation_cnt))



        self.D1 += cal_D(real_b_hot[0,0,:,:],fake_b_hot[0,0,:,:])
        self.D2 += cal_D(real_b_hot[0,1,:,:],fake_b_hot[0,1,:,:])
        self.D3 += cal_D(real_b_hot[0,2,:,:],fake_b_hot[0,2,:,:])
        print (" D1 =  "  + str(self.D1/self.validation_cnt))
        print (" D2 =  "  + str(self.D2/self.validation_cnt))
        print (" D3 =  "  + str(self.D3/self.validation_cnt))

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
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
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        #self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        #LGQ special fusion loss
        loss = self.criterionMTL.multi_loss(self.out_pathes,self.real_pathes)
        #self.loss_G_L1 =( 1.0*loss[0]  + 0.5*loss[1] + 0.1*loss[2] + 0.2*loss[3])*self.opt.lambda_L1
        #self.loss_G_L1 =( 1.0*loss[0]  + 0.02*loss[1] + 0.02*loss[2]+ 0.02*loss[3]+ 0.02*loss[4]+ 0.02*loss[5])*self.opt.lambda_L1
        #self.loss_G_L1_2 = 0.5*loss[0] 
        self.loss_G_L1 =( 1.0*loss[0]     )*self.opt.lambda_L1
        self.loss_G_L1_2 = 0.5*loss[5] 
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        #self.loss_G =   self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A) # seperatee the for
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
