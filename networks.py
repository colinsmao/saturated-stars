import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.utils.spectral_norm import SpectralNorm


# HELPER FUNCTIONS / MODULES #


def BCEWithLogitsLoss_expand(output, label):
    """Expand size label to output sized tensor, then calculate BCEWithLogitsLoss"""
    return nn.BCEWithLogitsLoss()(output, label.expand_as(output))


def get_nfs(nf, i, increasing=True, max_exp=3):
    """
    Return nf1 and nf2, the number of input/output features at depth i.
    Args:
        * nf: the base i.e. initial (for increasing) or final (for decreasing) feature depth
        * i: the current level
        * increasing: bool, whether nf1 <= nf2 or vice versa
        * max_exp: cap maximum nf, default at (2**3 == 8) * nf
    """
    nf1 = (2. ** min(i, max_exp)) * nf
    nf2 = (2. ** min(i + (1 if increasing else -1), max_exp)) * nf
    return int(nf1), int(nf2)


class Conv2dPlusLinear(nn.Module):
    """Convolutional layer that also takes a-dimensional numerical data"""
    def __init__(self, in_channels, out_channels, numeric_features,
                 kernel_size, stride=1, padding=0,
                 expand_factor=None, bias=True):
        super().__init__()
        self.ef = expand_factor
        if self.ef is None:
            # Standard convolution layer
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            # Linear layer to map numeric_features to out_channels, to be added to conv output
            self.linear = nn.Linear(in_features=numeric_features, out_features=out_channels, bias=False)
        else:
            # Expanded convolution layer to expand_factor * out_channels
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=self.ef*out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            # Linear layer to map numeric_features to expand_factor * out_channels, to be added to conv output
            self.linear = nn.Linear(in_features=numeric_features, out_features=self.ef*out_channels, bias=False)
            # 1x1 convolution layer to reduce dims from expand_factor * out_channels to out_channels
            self.conv11 = nn.Conv2d(in_channels=self.ef*out_channels, out_channels=out_channels,
                                    kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, image, label):
        image_out = self.conv(image)
        label_out = self.linear(label).unsqueeze(2).unsqueeze(3)
        output = image_out + label_out
        if self.ef is None:
            return output
        else:
            return self.conv11(output)


class GeneralizedSequential(nn.Sequential):
    """Generalization of nn.Sequential that takes arbitrary number of inputs"""
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, *input_):
        """Overrides the forward method in nn.Sequential, with *input for the first layer"""
        first = True
        for module in self:
            if first:
                input_ = module(*input_)
                first = False
            else:
                input_ = module(input_)
        return input_


class OneNetwork(nn.Module):
    """Merge Generator and Discriminator into one network, for tensorboard visualization"""
    def __init__(self, netG, netD, n=1):
        super().__init__()
        self.n = n
        self.netG = netG
        self.netD = netD

    def forward(self, label, noise):
        return self.netD(self.netG(label, noise), label[:, self.n:])


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    """Copied from https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html"""
    # Extra check of hasattr for classes with no 'weight'
    if hasattr(module, name):
        if dim is None:
            if isinstance(module, (torch.nn.ConvTranspose1d,
                                   torch.nn.ConvTranspose2d,
                                   torch.nn.ConvTranspose3d)):
                dim = 1
            else:
                dim = 0
        SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


# DISCRIMINATORS #

class MultiScalePatchDiscriminator(nn.Module):
    def __init__(self, input_nc, numeric_features, output_preds, ndf, conv_layers, pred_layers,
                 numeric_convs=None, numeric_preds=None, norm_type='batch', mode=0):
        """Construct a multi-scale PatchGAN discriminator
        Args:
            input_nc (int)          -- the number of channels in input images
            numeric_features (int)  -- the number of numeric (non image) features
            output_preds (int)      -- the number of predictions to output
            ndf (int)               -- the number of filters in the first conv layer
            conv_layers (int)       -- the number of conv layers in the primary conv net
            pred_layers (int)       -- the number of layers at end to apply prediction map to
            numeric_convs (list)    -- convolution layers to add linear layer with numeric input
            numeric_preds (list)    -- prediction layers to add linear layer with numeric input
            norm_type (str)         -- normalization layer type, default batchnorm
            mode (int)              -- downsampling mode (different kernel_size/stride/padding for Conv2d)
        """
        super().__init__()
        self.output_preds = output_preds
        # ModuleDicts to register layers properly
        self.primary_layers = nn.ModuleDict()
        self.prediction_layers = nn.ModuleDict()

        # Layers to add conv2d+linear, in the main network, and prediction subnetworks
        if numeric_features == 0:
            numeric_convs = []          # Blank out
            numeric_preds = []          # Blank out
        else:
            if numeric_convs is None:
                numeric_convs = [0]     # Default
            if numeric_preds is None:
                numeric_preds = []      # Default

        if mode in [0]:
            kernel_size = 3
            padding = 0
        elif mode in [1, 2]:
            kernel_size = 4
            padding = 1
        else:
            raise ValueError("Downsampling mode '{}' is not supported.".format(mode))

        for c in range(conv_layers):  # gradually increase the number of filters, up to 8
            conv_block = OrderedDict([])
            ndf1, ndf2 = get_nfs(ndf, c-1, increasing=True)
            if c == 0:
                ndf1 = input_nc

            # Create a primary conv layer
            if c in numeric_convs:
                # If there are numeric features, add linear layer
                conv_layer = Conv2dPlusLinear(
                    in_channels=ndf1, out_channels=ndf2, numeric_features=numeric_features,
                    kernel_size=kernel_size+(c == 0), stride=2-(c == 0), padding=padding+(c == 0),
                    expand_factor=2, bias=(norm_type != 'batch'))
            else:
                conv_layer = nn.Conv2d(
                    in_channels=ndf1, out_channels=ndf2,
                    kernel_size=kernel_size+(c == 0), stride=2-(c == 0), padding=padding+(c == 0),
                    bias=(norm_type != 'batch'))
            conv_block['conv' + str(c)] = conv_layer

            # Add relevant normalization
            if norm_type == 'spectralD' or norm_type == 'spectralDG':
                conv_block['conv'+str(c)].apply(spectral_norm)
            elif norm_type == 'batch':
                conv_block['batchnorm'+str(c)] = nn.BatchNorm2d(ndf2)
            else:
                raise ValueError("norm_type '{}' not supported.".format(norm_type))

            # Leaky ReLU activation
            conv_block['leaky_relu'+str(c)] = nn.LeakyReLU(0.2, True)

            if c in numeric_convs:
                self.primary_layers['conv{:d}'.format(c)] = GeneralizedSequential(conv_block)
            else:
                self.primary_layers['conv{:d}'.format(c)] = nn.Sequential(conv_block)

            # Create pred_layers prediction layers, consiting of two convolution layers to a feature depth of 1
            if c >= conv_layers - pred_layers:
                if c in numeric_preds:
                    pred_layer = Conv2dPlusLinear(ndf2, self.output_preds, kernel_size, 1, 0)
                else:
                    pred_layer = nn.Conv2d(ndf2, self.output_preds, kernel_size, 1, 0)

                if norm_type == 'spectralD' or norm_type == 'spectralDG':
                    pred_layer.apply(spectral_norm)

                self.prediction_layers['pred{:d}'.format(c)] = pred_layer

    def forward(self, image, label):
        output = image
        preds = []
        for c, conv_block in self.primary_layers.items():
            # Build the conv net
            if conv_block[0].__class__.__name__ == 'Conv2d':
                output = conv_block(output)
            elif conv_block[0].__class__.__name__ == 'Conv2dPlusLinear':
                output = conv_block(output, label)
            else:
                raise Exception

            # If pred, create pred vector
            p = 'pred' + c[4:]  # c is a string of the form 'convN'
            if p in self.prediction_layers:
                pred_layer = self.prediction_layers[p]
                if pred_layer.__class__.__name__ == 'Conv2d':
                    preds.append(pred_layer(output))
                elif pred_layer.__class__.__name__ == 'Conv2dPlusLinear':
                    preds.append(pred_layer(output, label))
                else:
                    raise Exception

        return torch.cat([p.view(p.size(0), self.output_preds, -1) for p in preds], dim=2)


# GENERATORS #

class UpsamplingLayer(nn.Module):
    """Upsampling layer for Generator: Increases dimensions from (batch, in, N, N) to (batch, out, 2*N, 2*N)"""
    def __init__(self, in_channels, out_channels, bias=True, mode=0, first=False):
        """Create an upsampling layer for the Generator, according to mode.
        Args:
            in/out_channels (int)   -- in and out channels
            bias (bool)             -- whether to add a bias
            mode (int)              -- upsampling mode (ConvTranspose2d vs ResizeConvolution, and stride size)
            first (bool)            -- for some modes, upsample by scale of 4 for first layer (i.e. 1x1 to 4x4)
        """
        super().__init__()

        if mode == 0:
            self.layer = nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 0, bias=bias)
        elif mode == 1:
            if first:
                stride, padding = 1, 0
            else:
                stride, padding = 2, 1
            self.layer = nn.ConvTranspose2d(in_channels, out_channels, 4, stride, padding, bias=bias)
        elif mode == 2:
            if first:
                self.layer = nn.ConvTranspose2d(in_channels, out_channels, 4, 1, 0, bias=bias)
            else:
                layer = [nn.Upsample(scale_factor=2, mode='nearest')]
                layer += [nn.ReflectionPad2d(1)]
                layer += [nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias=bias)]
                self.layer = nn.Sequential(*layer)
        else:
            raise ValueError("Upsampling mode '{}' is not supported.".format(mode))

    def forward(self, input_):
        return self.layer(input_)


class ConditionalGenerator(nn.Module):
    def __init__(self, numeric_features, nz, output_nc, ngf, levels, norm_type='batch', mode=0):
        """Construct a conditional generator
        Args:
            numeric_features (int)  -- number of numeric input features (i.e. input size [0])
            nz (int)                -- size of noise vector (i.e. input size [1])
            output_nc (int)         -- number of output channels
            ngf (int)               -- number of features in final (deep) layer
            levels (int)            -- number of levels
            norm_type (str)         -- normalization layer type, default batchnorm
            mode (int)              -- upsampling mode
        """
        super().__init__()
        if levels < 2:
            raise ValueError("Conditional Generator requires 'levels' >= 2.")
        if numeric_features == 0:
            raise ValueError("Conditional Generator requires 'numeric_features' > 0.")

        ngf1, ngf2 = get_nfs(ngf, levels - 1, increasing=False)
        # Linear layer, before first layer of deconvolution
        self.linear = nn.Linear(numeric_features, ngf1)

        if mode in [0, 1]:
            name = 'convT'
        elif mode in [2]:
            name = 'reconv'
        else:
            raise ValueError("Upsampling mode '{}' is not supported.".format(mode))

        # Deconvolution net
        model = OrderedDict([])
        for i in range(levels):
            ngf1, ngf2 = get_nfs(ngf, levels - 1 - i, increasing=False)

            if i == 0:
                # Add first upsampling layer
                model[name+str(i)] = UpsamplingLayer(ngf1 + nz, ngf2, bias=False, mode=mode, first=True)
                # Add spectral normalization if required
                if norm_type == 'spectralDG':
                    model[name+str(i)].apply(spectral_norm)
            else:
                # Add batchnorm if not spectral normalization
                if norm_type != 'spectralDG':
                    model['batchnorm'+str(i-1)] = nn.BatchNorm2d(ngf1)
                # Add leaky ReLU layer
                model['relu'+str(i-1)] = nn.ReLU(True)

                if i < levels - 1:  # Remaining upsampling layers
                    model[name+str(i)] = UpsamplingLayer(ngf1, ngf2, bias=False, mode=mode)
                    if norm_type == 'spectralDG':
                        model[name+str(i)].apply(spectral_norm)
                else:  # Final layer (doesn't change image dimensions)
                    model['pad'+str(i)] = nn.ReflectionPad2d(2)
                    model['conv' + str(i)] = nn.Conv2d(ngf1, output_nc, 5, 1, 0, bias=True)
                    if norm_type == 'spectralDG':
                        model['conv'+str(i)].apply(spectral_norm)

        model['tanh'] = nn.Tanh()
        self.model = nn.Sequential(model)

    def forward(self, label, noise):
        vector = self.linear(label)
        vector = torch.cat((vector, noise), dim=-1).unsqueeze(2).unsqueeze(3)
        return self.model(vector)

