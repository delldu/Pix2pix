import torch
from .base_model import BaseModel
from . import networks
import pdb


class Pix2PixModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # pdb.set_trace()
        # (Pdb) pp opt
        # Namespace(batch_size=16, beta1=0.5, checkpoints_dir='./checkpoints', continue_train=False,
        #     crop_size=256, dataroot='dataset', dataset_mode='colorization', direction='AtoB',
        #     display_env='main', display_freq=400, display_id=1, display_ncols=4, display_port=8097,
        #     display_server='http://localhost', display_winsize=256, epoch='1000', epoch_count=200,
        #     gan_mode='vanilla', gpu_ids=[0], init_gain=0.02, init_type='normal', input_nc=1, isTrain=True,
        #     lambda_L1=100.0, load_iter=0, load_size=286, lr=0.0002, lr_decay_iters=50, lr_policy='linear',
        #     max_dataset_size=inf, model='colorization', n_layers_D=3, name='experiment_name', ndf=64,
        #     netD='basic',
        #     netG='unet_256', ngf=64, niter=500, niter_decay=500, no_dropout=False, no_flip=False, no_html=True,
        #     norm='batch', num_threads=4, output_nc=2, phase='train', pool_size=0, preprocess='resize_and_crop',
        #     print_freq=100, save_by_iter=False, save_epoch_freq=5, save_latest_freq=5000, serial_batches=False,
        #     suffix='', update_html_freq=1000, verbose=False)

        # specify the training losses you want to print out
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display.
        # The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define a discriminator; conditional GANs need to take both input and output images;
            # Therefore, channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type,
                                          opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # (Pdb) pp self.netG
        # DataParallel(
        #   (module): UnetGenerator(
        #     (model): UnetSkipConnectionBlock(
        #       (model): Sequential(
        #         (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #         (1): UnetSkipConnectionBlock(
        #           (model): Sequential(
        #             (0): LeakyReLU(negative_slope=0.2, inplace=True)
        #             (1): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #             (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (3): UnetSkipConnectionBlock(
        #               (model): Sequential(
        #                 (0): LeakyReLU(negative_slope=0.2, inplace=True)
        #                 (1): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #                 (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #                 (3): UnetSkipConnectionBlock(
        #                   (model): Sequential(
        #                     (0): LeakyReLU(negative_slope=0.2, inplace=True)
        #                     (1): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #                     (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #                     (3): UnetSkipConnectionBlock(
        #                       (model): Sequential(
        #                         (0): LeakyReLU(negative_slope=0.2, inplace=True)
        #                         (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #                         (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #                         (3): UnetSkipConnectionBlock(
        #                           (model): Sequential(
        #                             (0): LeakyReLU(negative_slope=0.2, inplace=True)
        #                             (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #                             (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #                             (3): UnetSkipConnectionBlock(
        #                               (model): Sequential(
        #                                 (0): LeakyReLU(negative_slope=0.2, inplace=True)
        #                                 (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #                                 (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #                                 (3): UnetSkipConnectionBlock(
        #                                   (model): Sequential(
        #                                     (0): LeakyReLU(negative_slope=0.2, inplace=True)
        #                                     (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #                                     (2): ReLU(inplace=True)
        #                                     (3): ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #                                     (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #                                   )
        #                                 )
        #                                 (4): ReLU(inplace=True)
        #                                 (5): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #                                 (6): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #                                 (7): Dropout(p=0.5, inplace=False)
        #                               )
        #                             )
        #                             (4): ReLU(inplace=True)
        #                             (5): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #                             (6): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #                             (7): Dropout(p=0.5, inplace=False)
        #                           )
        #                         )
        #                         (4): ReLU(inplace=True)
        #                         (5): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #                         (6): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #                         (7): Dropout(p=0.5, inplace=False)
        #                       )
        #                     )
        #                     (4): ReLU(inplace=True)
        #                     (5): ConvTranspose2d(1024, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #                     (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #                   )
        #                 )
        #                 (4): ReLU(inplace=True)
        #                 (5): ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #                 (6): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #               )
        #             )
        #             (4): ReLU(inplace=True)
        #             (5): ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #             (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #           )
        #         )
        #         (2): ReLU(inplace=True)
        #         (3): ConvTranspose2d(128, 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        #         (4): Tanh()
        #       )
        #     )
        #   )
        # )
        self.fake_B = self.netG(self.real_A)  # G(A)
        # pdb.set_trace()
        # (Pdb) pp self.fake_B.size()
        # torch.Size([10, 2, 256, 256])

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""

        # (Pdb) pp self.netD
        # DataParallel(
        #   (module): NLayerDiscriminator(
        #     (model): Sequential(
        #       (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        #       (1): LeakyReLU(negative_slope=0.2, inplace=True)
        #       (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #       (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (4): LeakyReLU(negative_slope=0.2, inplace=True)
        #       (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #       (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (7): LeakyReLU(negative_slope=0.2, inplace=True)
        #       (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
        #       (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (10): LeakyReLU(negative_slope=0.2, inplace=True)
        #       (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
        #     )
        #   )
        # )


        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero

        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # (Pdb) fake_AB.size()
        # torch.Size([10, 3, 256, 256])
        pred_fake = self.netD(fake_AB.detach())
        # (Pdb) pred_fake.size()
        # torch.Size([10, 1, 30, 30])
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # (Pdb) self.criterionGAN
        # GANLoss(
        #   (loss): BCEWithLogitsLoss()
        # )
        # (Pdb) self.loss_D_fake
        # tensor(0.7503, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

        self.optimizer_D.step()          # update D's weights
        # (Pdb) self.optimizer_D
        # Adam (
        # Parameter Group 0
        #     amsgrad: False
        #     betas: (0.5, 0.999)
        #     eps: 1e-08
        #     initial_lr: 0.0002
        #     lr: 0.0002
        #     weight_decay: 0
        # )

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # (Pdb) pp pred_fake.size()
        # torch.Size([10, 1, 30, 30])

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # (Pdb) pp self.opt.lambda_L1
        # 100.0
        # (Pdb) self.criterionL1
        # L1Loss()
        # (Pdb) self.loss_G_L1
        # tensor(17.2437, device='cuda:0', grad_fn=<MulBackward0>)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

        self.optimizer_G.step()             # udpate G's weights
        # pdb.set_trace()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        # # self.set_requires_grad(self.netD, True)  # enable backprop for D
        # # self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        # # self.optimizer_D.step()          # update D's weights

        # update G(Pdb) self.loss_G_L1
        # tensor(17.2437, device='cuda:0', grad_fn=<MulBackward0>)

        # # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        # # self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        # # self.optimizer_G.step()             # udpate G's weights
