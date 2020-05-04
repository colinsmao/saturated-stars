import time
import argparse
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import networks


class ModelOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--ngpu', type=int, default=0,
                                 help='number of GPUs: use 0 for CPU')
        self.parser.add_argument('--size', type=int, default=64,
                                 help='size of training images in pixels; all images will be resized to this size')
        self.parser.add_argument('--nz', type=int, default=128,
                                 help='size of latent vector')
        self.parser.add_argument('--input_nc', type=int, default=1,
                                 help='# of input image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--output_nc', type=int, default=1,
                                 help='# of output image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--G_levels', type=int, default=5,
                                 help='# of layers in generator')
        self.parser.add_argument('--D_levels', type=int, default=6,
                                 help='# of layers in discriminator')
        self.parser.add_argument('--D_pred', type=int, default=1,
                                 help='# of prediction layers in discriminator, for multi-scale')
        self.parser.add_argument('--numeric_features', type=int, default=13,
                                 help='# of numeric features')
        self.parser.add_argument('--ngf', type=int, default=64,
                                 help='# of gen filters in the last conv layer')
        self.parser.add_argument('--ndf', type=int, default=128,
                                 help='# of discrim filters in the first conv layer')
        self.parser.add_argument('--lr', type=float, default=0.001,  # 0.0002
                                 help='learning rate for Adam Aptimizer')
        self.parser.add_argument('--beta1', type=float, default=0.5,
                                 help='beta1 hyperparam for Adam optimizers')
        self.parser.add_argument('--lossD_weight', type=float, default=0.5,
                                 help='multiplier for lossD')
        self.parser.add_argument('--lambda_DL', type=float, default=50,
                                 help='lambda factor for discriminator L1/2 loss')
        self.parser.add_argument('--lambda_GL', type=float, default=0.5,
                                 help='lambda factor for generator L1/2 loss')
        self.parser.add_argument('--netG_name', type=str, default='cGAN',
                                 help='netG type')
        self.parser.add_argument('--netD_name', type=str, default='Multi',
                                 help='netD type')
        self.parser.add_argument('--distD', type=str, default='L1',
                                 help='distance type to use in disciminator loss (L1 or L2/MSE)')
        self.parser.add_argument('--distG', type=str, default='L1',
                                 help='distance type to use in generator loss (L1 or L2/MSE)')
        self.parser.add_argument('--norm_type', type=str, default='spectralD',
                                 help="normalization layer type (batch, spectralD, spectralDG, etc.)")
        self.parser.add_argument('--sampling_mode', type=int, default=1,
                                 help="sampling mode")

    def get_default_opts(self):
        return self.parser.parse_args([])

    def get_opts(self, opts):
        return self.parser.parse_known_args(opts)[0]


class BaseModel(ABC):
    def __init__(self, opt, fixed_label_noise=None, save_dir=None, training=False):
        """Create a (Conditional) GAN model.
        Arguments:
            * opt: an (arg)parser object containing relevant options
            * fixed_label_noise: (label, noise) to pass to netG for consistent generator tracking - if None, no tracking
            * save_dir: path to tensorboard logdir - can be None, in which case no tensorboard saving is done
            * training: whether the model is initialized in training mode or not
        """
        self.opt = opt
        # Make a unique name for the model, for saving purposes. Is saved and loaded along with the model.
        self.unique_name = self._get_unique_name()

        self.ngpu = self.opt.ngpu  # used often so create alias
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

        # Model
        self.netG = None                        # Generator
        self.netD = None                        # Discriminator
        self._init_nets()                       # Initialization of nets in mini-function

        # Send nets to relevant device
        self.netG = self.netG.to(self.device)
        self.netD = self.netD.to(self.device)

        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(self.ngpu)))
            self.netD = nn.DataParallel(self.netD, list(range(self.ngpu)))

        # Apply the _weights_init function (recursively) to randomly initialize all weights
        self.netG.apply(self._weights_init)
        self.netD.apply(self._weights_init)

        # Losses
        self.criterionGAN = networks.BCEWithLogitsLoss_expand   # BCELoss with sigmoid function, with broadcasting
        if self.opt.distG == 'L1':
            self.criterionGL = nn.L1Loss()      # L1 Loss
        elif self.opt.distG == 'L2' or self.opt.distG == 'MSE':
            self.criterionGL = nn.MSELoss()     # L2 Loss
        else:
            raise ValueError("Loss distance '{)' is not supported.".format(self.opt.distG))

        # Loss tensors (will have .backward() called on them)
        self.lossD = None           # Discriminator loss
        self.lossG_GAN = None       # Generator GAN (BCE) Loss
        self.lossG_L = None         # Generator L1/2 loss
        self.lossG = None           # Generator loss
        self.loss_tensors = ['lossD', 'lossG_GAN', 'lossG_L']

        # Adam optimizers for both G and D
        self.optD = optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optG = optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        # Set training / eval mode
        self.training = training
        self.train(self.training)

        # Training tensors
        self.real_flag = torch.tensor(1., device=self.device)   # real/true label for discriminator training, default 1
        self.fake_flag = torch.tensor(0., device=self.device)   # fake/false label for discriminator training, default 0
        self.label = None                   # real input, i.e. the labels (true fluxes / positions), for Conditional GAN
        self.noise = None                   # input noise to generator, i.e. latent space
        self.real_image = None              # real 'output', i.e. the training images
        self.fake_image = None              # fake output, i.e. netG(label, noise)
        self.sigmoid = nn.Sigmoid()

        # Model tracking
        self.fixed_label_noise = fixed_label_noise   # Fixed input tensor, for consistent evolution of G - can be None
        if self.fixed_label_noise is None:
            print('Warning: fixed_in is None, so generator evolution will not be tracked.')
        # Training info
        self.iters = 0                      # Iterations (incremented every call of optimize_parameters)
        self.start_time = None              # Starting wall time from time.time()
        # Tensors for tracking state: only stores the latest value
        self.D_x = None                     # D(x), should approach 0.5 as G improves
        self.D_G_z1 = None                  # D(G(z)) before updating G
        self.D_G_z2 = None                  # D(G(z)) after updating G
        # Collections of evaluated tensors for logging / history
        self.losses = []                    # Losses
        self.preds = []                     # Discriminator predictions
        self.grads = defaultdict(list)      # Gradients
        self.img_list = OrderedDict()       # Generated images, i.e. G(fixed_in) at different iters
        self.test_results = OrderedDict()   # Test results at different iters
        # Make tensorboard writer if needed, else None
        self.save_dir = save_dir
        self.writer = None
        if self.save_dir is not None and self.training:
            self._make_writer()
            if self.ngpu == 0:
                self.writer.add_graph(networks.OneNetwork(self.netG, self.netD),
                                      [torch.rand(2, self.opt.numeric_features, device=self.device),
                                       torch.rand(2, self.opt.nz, device=self.device)])

    @staticmethod
    def _get_unique_name():
        """Not technically unique but close enough"""
        import datetime
        time_str = datetime.datetime.now().strftime("%y%m%d%H%M%S%f")  # yymmddHHMMSSFFFFFF down to milliseconds
        rand_str = str(np.random.randint(1000, 10000))  # Random 4-digit identifier
        return time_str + rand_str

    @abstractmethod
    def _init_nets(self):
        pass

    def train(self, mode=True):
        self.training = mode
        self.netD.train(mode)
        self.netG.train(mode)

    def eval(self):  # Alias for self.train(False)
        self.train(False)

    def _make_writer(self):
        import os
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(self.save_dir, self.unique_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def _weights_init(m):
        """custom weights initialization called on netG and netD"""
        classname = m.__class__.__name__
        if classname == 'Conv2d' or classname == 'ConvTranspose2d':
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # 0.02
        elif classname == 'Linear':
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.normal_(m.bias.data, 0.0, 0.02)
        elif classname == 'BatchNorm2d':
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def set_input(self, input_):
        self.real_image = input_[0]
        self.label = input_[1]
        self.noise = input_[2]

    def forward(self):
        self.fake_image = self.netG(self.label, self.noise)

    @abstractmethod
    def _backward_D(self):
        pass

    @abstractmethod
    def _backward_G(self):
        pass

    def optimize_parameters(self):
        if not self.training:
            raise Exception('Model is not in training mode, cannot optimize parameters.')
        self.forward()                            # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)   # enable backprop for D
        self.optD.zero_grad()                     # set D's gradients to zero
        self._backward_D()                        # calculate gradients for D
        self.optD.step()                          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optG.zero_grad()                     # set G's gradients to zero
        self._backward_G()                        # calculate graidents for G
        self.optG.step()                          # udpate G's weights

        self.iters += 1                           # increment iteration counter

    def optimize_G_only(self, num=1):
        if not self.training:
            raise Exception('Model is not in training mode, cannot optimize parameters.')
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        for _ in range(num):
            self.forward()                        # compute fake images: G(A)
            self.optG.zero_grad()                 # set G's gradients to zero
            self._backward_G()                    # calculate graidents for G
            self.optG.step()                      # udpate G's weights

    def get_grads(self):
        """Get gradients of netD and netG."""
        numD, numG, parD, parG, gradD, gradG = [0]*6
        for name, param in self.netD.named_parameters():
            if name.find('batchnorm') == -1:  # is not batchnorm layer
                if param.grad is not None:
                    numD += np.prod(param.shape)
                    parD += param.abs().sum()
                    gradD += param.grad.abs().sum()
        for name, param in self.netG.named_parameters():
            if name.find('batchnorm') == -1:  # is not batchnorm layer
                if param.grad is not None:
                    numG += np.prod(param.shape)
                    parG += param.abs().sum()
                    gradG += param.grad.abs().sum()
        if numD == 0:  # To prevent div by 0
            numD = 1
        if numG == 0:
            numG = 1
        return np.array([[numD, numG], [parD, parG], [gradD, gradG]])

    @abstractmethod
    def get_formatted_loss(self):
        pass

    def get_formatted_time(self, epoch=None, num_epochs=None):
        elapsed_time = time.time() - self.start_time
        if epoch is None or num_epochs is None:
            epoch_str = ''
        else:
            epoch_str = '[{}/{:d}] ' \
                .format(str(epoch + 1).rjust(int(np.log10(num_epochs)) + 1), num_epochs)
        time_str = '{}i: {:5d} t: {:7.1f}s   '.format(epoch_str, self.iters, elapsed_time)

        return time_str

    def get_formatted_state(self):
        [[numD, numG], [parD, parG], [gradD, gradG]] = self.get_grads()
        loss_str = self.get_formatted_loss()
        state_str = '{}D(x/G(z)): {:.2f} / {:.2f} < {:.2f}   ' \
                    'grad (/1e-6): D: {:6.5g} / {:5.5g}   G: {:6.5g} / {:5.5g}' \
            .format(loss_str, self.D_x.item(), self.D_G_z1.item(), self.D_G_z2.item(),
                    1e6 * gradD / numD, 1e6 * parD / numD, 1e6 * gradG / numG, 1e6 * parG / numG)

        return state_str

    def set_start_time(self):
        self.start_time = time.time()

    def save(self, SAVE_PATH):
        save_dict = {
            'unique_name': self.unique_name,
            'iters': self.iters,
            'netD_state_dict': self.netD.state_dict(),
            'netG_state_dict': self.netG.state_dict(),
            'optD_state_dict': self.optD.state_dict(),
            'optG_state_dict': self.optG.state_dict()
        }
        # if self.opt.tensorboard_dir is None:  # below lists will be empty if None anyways
        if len(self.losses) > 0:
            save_dict['losses'] = self.losses
        if len(self.preds) > 0:
            save_dict['preds'] = self.preds
        if len(self.grads) > 0:
            save_dict['grads'] = self.grads
        if len(self.img_list) > 0:
            save_dict['img_list'] = self.img_list

        torch.save(save_dict, SAVE_PATH)

    def load(self, LOAD_PATH):
        checkpoint = torch.load(LOAD_PATH)
        self.unique_name = checkpoint['unique_name']
        self.iters = checkpoint['iters']
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.optD.load_state_dict(checkpoint['optD_state_dict'])
        self.optG.load_state_dict(checkpoint['optG_state_dict'])

        if 'losses' in checkpoint:
            self.losses = checkpoint['losses']
        if 'preds' in checkpoint:
            self.preds = checkpoint['preds']
        if 'grads' in checkpoint:
            self.grads = checkpoint['grads']
        if 'img_list' in checkpoint:
            self.img_list = checkpoint['img_list']

        if self.save_dir is not None:
            self._make_writer()

    def save_losses(self):
        """Save losses at current state, for tracking / plotting"""
        if self.writer is None:
            self.losses.append([getattr(self, x).item() for x in self.loss_tensors])
            self.preds.append([self.D_x.item(), self.D_G_z1.item(), self.D_G_z2.item()])
        else:
            for x in self.loss_tensors:
                self.writer.add_scalar('loss/{}'.format(x), getattr(self, x).item(), self.iters)
            self.writer.add_scalar('pred/D(x)', self.D_x.item(), self.iters)
            self.writer.add_scalar('pred/D(G(z1))', self.D_G_z1.item(), self.iters)
            self.writer.add_scalar('pred/D(G(z2))', self.D_G_z2.item(), self.iters)

    def save_grads(self):
        """Save gradients at current state, for tracking / plotting"""
        for name, param in self.netG.named_parameters():
            if name.find('batchnorm') == -1:  # is not batchnorm layer
                if param.grad is not None:
                    grad = param.grad.abs().mean()
                    if self.writer is None:
                        self.grads['netD.{}'.format(name)].append(grad)
                    else:
                        self.writer.add_scalar('grad/netG/{}'.format(name), grad, self.iters)
        for name, param in self.netD.named_parameters():
            if name.find('batchnorm') == -1:  # is not batchnorm layer
                if param.grad is not None:
                    grad = param.grad.abs().mean()
                    if self.writer is None:
                        self.grads['netD.{}'.format(name)].append(grad)
                    else:
                        self.writer.add_scalar('grad/netD/{}'.format(name), grad, self.iters)

    def generate_images(self, label, noise):
        if self.training:
            was_training = True
            self.eval()
        else:
            was_training = False
        images = self.netG(label, noise).detach().cpu()
        if was_training:
            self.train()
        return images

    def save_images(self):
        if self.fixed_label_noise is not None:
            images = self.generate_images(*self.fixed_label_noise)
            if self.writer is None:
                self.img_list[self.iters] = images
            else:
                self.writer.add_images('images', images, self.iters)


class ConditionalModel(BaseModel):
    def __init__(self, opt, **kwargs):
        """Create a Conditional GAN model."""
        super().__init__(opt, **kwargs)

    def _init_nets(self):
        if self.opt.netG_name == 'cGAN':
            self.netG = networks.ConditionalGenerator(
                numeric_features=self.opt.numeric_features,
                nz=self.opt.nz,
                output_nc=self.opt.output_nc,
                ngf=self.opt.ngf,
                levels=self.opt.G_levels,
                norm_type=self.opt.norm_type,
                mode=self.opt.sampling_mode)
        else:
            raise ValueError("netG type '{}' is not supported".format(self.opt.netG_name))

        if self.opt.netD_name == 'Multi':
            self.netD = networks.MultiScalePatchDiscriminator(
                input_nc=self.opt.input_nc,
                numeric_features=self.opt.numeric_features,
                output_preds=1,
                ndf=self.opt.ndf,
                conv_layers=self.opt.D_levels,
                pred_layers=self.opt.D_pred,
                norm_type=self.opt.norm_type,
                mode=self.opt.sampling_mode)
        else:
            raise ValueError("netD type '{}' is not supported".format(self.opt.netD_name))

    def _backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Real batch
        pred_real = self.netD(self.real_image, self.label)
        lossD_real = self.criterionGAN(pred_real, self.real_flag)
        # For logging, evaluate D(x), discriminator guess of true image
        self.D_x = self.sigmoid(pred_real).mean()

        # Fake batch; stop backprop to the generator by detaching fake_out
        pred_fake = self.netD(self.fake_image.detach(), self.label)
        lossD_fake = self.criterionGAN(pred_fake, self.fake_flag)
        # For logging, evaluate D(G(z1)), discriminator guess of generated image, before updating generator
        self.D_G_z1 = self.sigmoid(pred_fake).mean()

        # combine loss and calculate gradients
        self.lossD = lossD_real + lossD_fake
        self.lossD_weighted = self.lossD * self.opt.lossD_weight
        self.lossD_weighted.backward()

    def _backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Fake batch; no longer detached, as we are training the generator
        pred_fake = self.netD(self.fake_image, self.label)
        # Calculate G's GAN loss based on this output; considered as real_flag to invert loss (G wants to decrease D)
        self.lossG_GAN = self.criterionGAN(pred_fake, self.real_flag)
        # Calculate G's L1 loss compared to the true sample
        self.lossG_L = self.criterionGL(self.fake_image, self.real_image)
        # For logging, evaluate D(G(z1)), discriminator guess of generated image, after updating generator
        self.D_G_z2 = self.sigmoid(pred_fake).mean()

        # Calculate gradients for G
        self.lossG = self.lossG_GAN + self.lossG_L * self.opt.lambda_GL
        self.lossG.backward()

    def get_formatted_loss(self):
        loss_str = 'Loss D: {:6.3f}   G: {:.3f} + \u03BB*{:.3f} = {:6.3f}   ' \
            .format(self.lossD.item(), self.lossG_GAN.item(), self.lossG_L.item(), self.lossG.item())
        return loss_str


class AuxiliaryModel(BaseModel):
    def __init__(self, opt, num_pred, **kwargs):
        """Create an Auxiliary Classifier GAN model."""
        self.n = num_pred
        super().__init__(opt, **kwargs)
        self.criterionDL = nn.L1Loss()
        self.lossD_GAN = None
        self.lossD_L = None
        self.loss_tensors = ['lossD_GAN', 'lossD_L', 'lossG_GAN', 'lossG_L']

    def _init_nets(self):
        if self.opt.netG_name == 'cGAN':
            self.netG = networks.ConditionalGenerator(
                numeric_features=self.opt.numeric_features,
                nz=self.opt.nz,
                output_nc=self.opt.output_nc,
                ngf=self.opt.ngf,
                levels=self.opt.G_levels,
                norm_type=self.opt.norm_type,
                mode=self.opt.sampling_mode)
        else:
            raise ValueError("netG type '{}' is not supported".format(self.opt.netG_name))

        if self.opt.netD_name == 'Multi':
            self.netD = networks.MultiScalePatchDiscriminator(
                input_nc=self.opt.input_nc,
                numeric_features=self.opt.numeric_features - self.n,
                output_preds=1 + self.n,
                ndf=self.opt.ndf,
                conv_layers=self.opt.D_levels,
                pred_layers=self.opt.D_pred,
                norm_type=self.opt.norm_type,
                mode=self.opt.sampling_mode)
        else:
            raise ValueError("netD type '{}' is not supported".format(self.opt.netD_name))

    def _backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Real batch
        pred_real = self.netD(self.real_image, self.label[:, self.n:])
        lossD_GAN_real = self.criterionGAN(pred_real[:, 0], self.real_flag)
        lossD_L_real = self.criterionDL(pred_real[:, 1:], self.label[:, :self.n].unsqueeze(2))
        # For logging, evaluate D(x), discriminator guess of true image
        self.D_x = self.sigmoid(pred_real).mean()

        # Fake batch; stop backprop to the generator by detaching fake_out
        pred_fake = self.netD(self.fake_image.detach(), self.label[:, self.n:])
        lossD_GAN_fake = self.criterionGAN(pred_fake[:, 0], self.fake_flag)
        lossD_L_fake = self.criterionDL(pred_fake[:, 1:], self.label[:, :self.n].unsqueeze(2))
        # For logging, evaluate D(G(z1)), discriminator guess of generated image, before updating generator
        self.D_G_z1 = self.sigmoid(pred_fake).mean()

        # combine loss and calculate gradients
        self.lossD_GAN = lossD_GAN_real + lossD_GAN_fake
        self.lossD_L = lossD_L_real + lossD_L_fake
        self.lossD = self.lossD_GAN + self.lossD_L * self.opt.lambda_DL
        self.lossD_weighted = self.lossD * self.opt.lossD_weight
        self.lossD_weighted.backward()

    def _backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Fake batch; no longer detached, as we are training the generator
        pred_fake = self.netD(self.fake_image, self.label[:, self.n:])
        # Calculate G's GAN loss based on this output; considered as real_flag to invert loss (G wants to decrease D)
        self.lossG_GAN = self.criterionGAN(pred_fake[:, 0], self.real_flag)
        # Calculate G's L1 loss compared to the true sample
        self.lossG_L = self.criterionGL(self.fake_image, self.real_image)
        # For logging, evaluate D(G(z1)), discriminator guess of generated image, after updating generator
        self.D_G_z2 = self.sigmoid(pred_fake).mean()

        # Calculate gradients for G
        self.lossG = self.lossG_GAN + self.lossG_L * self.opt.lambda_GL
        self.lossG.backward()

    def get_formatted_loss(self):
        loss_str = 'Loss D: {:.3f} + \u03BB*{:.3f} = {:6.3f}   ' \
                   'G: {:.3f} + \u03BB*{:.3f} = {:6.3f}   ' \
            .format(self.lossD_GAN.item(), self.lossD_L.item(), self.lossD.item(),
                    self.lossG_GAN.item(), self.lossG_L.item(), self.lossG.item())
        return loss_str

    def evaluate_for_dataloader(self, dataloader):
        if self.training:
            was_training = True
            self.eval()
        else:
            was_training = False

        # Run through dataloader and evaluate L1 loss and k_m predictions
        l1_losses, fluxes, preds, gen_preds = [], [], [], []
        for data in iter(dataloader):
            image = data[0]
            label = data[1]
            noise = data[2]
            gen_image = self.netG(label, noise).detach()
            l1_loss = torch.mean(torch.abs(image - gen_image), dim=(1, 2, 3)).detach().numpy()
            l1_losses.append(l1_loss)
            fluxes.append(label[:, 0])
            preds.append(self.netD(image, label[:, 1:])[:, 1].detach().numpy())
            gen_preds.append(self.netD(gen_image, label[:, 1:])[:, 1].detach().numpy())
        l1_losses = np.concatenate(l1_losses, axis=0)
        fluxes = np.concatenate(fluxes, axis=0)
        preds = np.concatenate(preds, axis=0).squeeze()
        gen_preds = np.concatenate(gen_preds, axis=0).squeeze()

        # Sort fluxes and predictions
        sort_order = fluxes.argsort()
        fluxes = fluxes[sort_order]
        preds = preds[sort_order]
        gen_preds = gen_preds[sort_order]

        pred_diffs = preds - fluxes
        gen_diffs = gen_preds - fluxes

        """
        # Bin fluxes and predictions and calculate summary statistics (mean, std)
        N = 6
        step = 2 / N
        pred_means, pred_sigmas, gen_means, gen_sigmas = {}, {}, {}, {}
        for i in range(N):
            a, b = fluxes.searchsorted(-1 + i*step), fluxes.searchsorted(-1 + (i+1)*step)
            if i == N - 1:
                b = len(fluxes) + 1
            f = '{:.1f}'.format(-1 + (i+0.5)*step)
            pred_means[f] = pred_diffs[a:b].mean()
            pred_sigmas[f] = pred_diffs[a:b].std()
            gen_means[f] = gen_diffs[a:b].mean()
            gen_sigmas[f] = gen_diffs[a:b].std()
        """

        # Save results
        if self.writer is None:
            self.test_results[self.iters] = {
                'l1_losses': l1_losses,
                'fluxes': fluxes,
                'preds': preds,
                'gen_preds': gen_preds
            }
        else:
            self.writer.add_scalar('test/l1_loss', l1_losses.mean(), self.iters)
            self.writer.add_histogram('test/pred_diffs', pred_diffs, self.iters)
            self.writer.add_histogram('test/gen_diffs', gen_diffs, self.iters)

        if was_training:
            self.train()

        return 'Evaluation result: L1 loss: {:.3f}, k_m preds: {:.3f} +- {:.3f}, gen: {:.3f} +- {:.3f}'\
            .format(l1_losses.mean(), pred_diffs.mean(), pred_diffs.std(), gen_diffs.mean(), gen_diffs.std())
