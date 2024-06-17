import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *

class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data
                param.data = self.shadow[name].data.clone()

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy
    
    # Sub-function for ema_copy
    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class Conv1dWithInitialization(nn.Module):
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = nn.Conv1d(**kwargs)
        nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ConvolutionBlock, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.convolution = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation)
    
    def forward(self, x):
        outputs = self.leaky_relu(x)
        outputs = self.convolution(outputs)
        return outputs


class BasicModulationBlock(nn.Module):
    
    # Linear modulation part of UBlock, represented by sequence of the following layers:
    #    - Feature-wise Affine
    #    - LReLU
    #    - 3x1 Conv
    
    def __init__(self, n_channels, dilation):
        super(BasicModulationBlock, self).__init__()
        self.featurewise_affine = FeatureWiseAffine()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.convolution = Conv1dWithInitialization(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation
        )

    def forward(self, x, scale, shift):
        outputs = self.featurewise_affine(x, scale, shift)
        outputs = self.leaky_relu(outputs)
        outputs = self.convolution(outputs)
        return outputs


class FeatureWiseAffine(nn.Module):
    def __init__(self):
        super(FeatureWiseAffine, self).__init__()

    def forward(self, x, scale, shift):
        outputs = scale * x + shift
        return outputs


class FeatureWiseLinearModulation(nn.Module):
    def __init__(self, in_channels, out_channels, input_dscaled_by):
        super(FeatureWiseLinearModulation, self).__init__()
        self.signal_conv = nn.Sequential(*[
            Conv1dWithInitialization(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(0.2)
        ])
        self.positional_encoding = PositionalEncoding(in_channels)
        self.scale_conv = Conv1dWithInitialization(in_channels=in_channels, out_channels=out_channels, kernel_size=3,stride=1, padding=1)
        self.shift_conv = Conv1dWithInitialization(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, noise_level):
        outputs = self.signal_conv(x)
        outputs = outputs + self.positional_encoding(noise_level).unsqueeze(-1)
        scale, shift = self.scale_conv(outputs), self.shift_conv(outputs)
        return scale, shift


class PositionalEncoding(nn.Module):
    def __init__(self, n_channels):
        super(PositionalEncoding, self).__init__()
        self.n_channels = n_channels
        self.linear_scale = 5e3

    def forward(self, noise_level):
        if len(noise_level.shape) > 1:
            noise_level = noise_level.squeeze(-1)
        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, dtype=torch.float32).to(noise_level) / float(half_dim)
        exponents = 1e-4 ** exponents
        exponents = self.linear_scale * noise_level.unsqueeze(1) * exponents.unsqueeze(0)
        return torch.cat([exponents.sin(), exponents.cos()], dim=-1)  


class InterpolationBlock(nn.Module):
    def __init__(self, scale_factor, mode='linear', align_corners=False, downsample=False):
        super(InterpolationBlock, self).__init__()
        self.mode = mode
        self.downsample = downsample
        self.scale_factor = scale_factor
        self.align_corners = align_corners
    
    def forward(self, x):
        if (self.downsample):
            size = x.shape[-1] // self.scale_factor
        else:
            size = x.shape[-1] * self.scale_factor

        outputs = nn.functional.interpolate(x,
            size = size, mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=False
        )
        return outputs


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor, dilations):
        super(UpsamplingBlock, self).__init__()
        self.first_block_main_branch = nn.ModuleDict({
            'upsampling': nn.Sequential(*[
                nn.LeakyReLU(0.2),
                InterpolationBlock(scale_factor=factor,mode='linear',align_corners=False),
                Conv1dWithInitialization(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=dilations[0],dilation=dilations[0])
            ]),
            'modulation': BasicModulationBlock(out_channels, dilation=dilations[1])
        })
        
        self.first_block_residual_branch = nn.Sequential(*[
            Conv1dWithInitialization(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1),
            InterpolationBlock(scale_factor=factor,mode='linear',align_corners=False)
        ])
        
        self.second_block_main_branch = nn.ModuleDict({
            f'modulation_{idx}': BasicModulationBlock(out_channels, dilation=dilations[2 + idx]) for idx in range(2)
        })

    def forward(self, x, scale, shift):
        # First upsampling residual block
        outputs = self.first_block_main_branch['upsampling'](x)
        outputs = self.first_block_main_branch['modulation'](outputs, scale, shift)
        outputs = outputs + self.first_block_residual_branch(x)

        # Second residual block
        residual = self.second_block_main_branch['modulation_0'](outputs, scale, shift)
        outputs = outputs + self.second_block_main_branch['modulation_1'](residual, scale, shift)
        return outputs


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor, dilations):
        super(DownsamplingBlock, self).__init__()
        in_sizes = [in_channels] + [out_channels for _ in range(len(dilations) - 1)]
        out_sizes = [out_channels for _ in range(len(in_sizes))]
        self.main_branch = nn.Sequential(*(
            [InterpolationBlock(scale_factor=factor, mode='linear', align_corners=False, downsample=True)] + 
            [ConvolutionBlock(in_size, out_size, dilation) 
             for in_size, out_size, dilation in zip(in_sizes, out_sizes, dilations)
        ]))
        
        self.residual_branch = nn.Sequential(*[
            Conv1dWithInitialization(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            InterpolationBlock(scale_factor=factor, mode='linear', align_corners=False, downsample=True)
        ])

    def forward(self, x):
        outputs = self.main_branch(x)
        outputs = outputs + self.residual_branch(x)
        return outputs


class WaveGradNN(nn.Module):
    def __init__(self, config):
        super(WaveGradNN, self).__init__()
        
        # Building upsampling branch (starting from mels -> signal)
        self.ublock_preconv = Conv1dWithInitialization(in_channels=len(AVE_CHANNELS_NAME), 
                                                       out_channels=config.upsampling_preconv_out_channels,
                                                       kernel_size=1282, stride=1, padding=0)
        
        upsampling_in_sizes = [config.upsampling_preconv_out_channels] + config.upsampling_out_channels[:-1]
        
        self.ublocks = nn.ModuleList([
            UpsamplingBlock(in_channels=in_size, out_channels=out_size, factor=factor,dilations=dilations) 
            for in_size, out_size, factor, dilations in 
                zip(upsampling_in_sizes, config.upsampling_out_channels, config.factors, config.upsampling_dilations)
        ])
        
        
        self.ublock_postconv = Conv1dWithInitialization(in_channels=config.upsampling_out_channels[-1], 
                                                        out_channels=len(AVE_CHANNELS_NAME),
                                                        kernel_size=3, stride=1, padding=1)
        
        self.dblock_preconv = Conv1dWithInitialization(in_channels=len(AVE_CHANNELS_NAME),
                                                       out_channels=config.downsampling_preconv_out_channels,
                                                       kernel_size=5, stride=1, padding=2)
        self.dblock_label = nn.Embedding(2, config.downsampling_preconv_out_channels)
        
        downsampling_in_sizes = [config.downsampling_preconv_out_channels] + config.downsampling_out_channels[:-1]
        self.dblocks = nn.ModuleList([
            DownsamplingBlock(in_channels=in_size, out_channels=out_size, factor=factor, dilations=dilations) 
            for in_size, out_size, factor, dilations in 
                zip(downsampling_in_sizes, config.downsampling_out_channels, config.factors[1:][::-1], config.downsampling_dilations)
        ])

        # Building FiLM connections (in order of downscaling stream)
        film_in_sizes = [config.downsampling_preconv_out_channels] + config.downsampling_out_channels
        film_out_sizes = config.upsampling_out_channels[::-1]
        film_factors = [1] + config.factors[1:][::-1]
        
        # for proper positional encodings initialization
        self.films = nn.ModuleList([
            FeatureWiseLinearModulation(in_channels=in_size, out_channels=out_size, input_dscaled_by=np.product(film_factors[:i+1])) 
            for i, (in_size, out_size) in enumerate(zip(film_in_sizes, film_out_sizes))
        ])

    def forward(self, x_0_dwt, x_0, noise_level, label):
        """
        Computes forward pass of neural network.
        :param x_0_dwt (torch.Tensor): shape [B, n_mels, T//hop_length]
        :param x_0 (torch.Tensor): noised signal `y_n` of shape [B, T]
        :param noise_level (float): level of noise added by diffusion
        :param label(torch.Tensor): label for the signal  (-1 imply no annotation, 1 imply annotation, 0 imply nothings)
        :return (torch.Tensor): epsilon noise
        """
        # Prepare inputs
        assert len(x_0_dwt.shape) == 3  # Batch_size, 19, DURATION*NEW_FREQUENCY
        assert len(x_0.shape)     == 3  # Batch_size, 19, >=DURATION*NEW_FREQUENCY
        assert len(label.shape)   == 1  # Batch_size,
        
        
        # Pass in label information to turn diffusion model to classifier-free guidance diffusion model
        statistics = []
        dblock_outputs = self.dblock_preconv(x_0)
        if label[0] != 2:    # If it is = 2, no label information passes  dblock_outputs = dblock_outputs + 0 
            dblock_labels  = self.dblock_label(label)[:,:,None]     # Add 1 more dim and become (Batch_size, 19, 1)
            dblock_outputs = dblock_outputs + dblock_labels         # Add in the information of the label    
        
        # Downsampling stream + Linear Modulation statistics calculation
        scale, shift = self.films[0](x=dblock_outputs, noise_level=noise_level)
        
        statistics.append([scale, shift])
        for dblock, film in zip(self.dblocks, self.films[1:]):
            dblock_outputs = dblock(dblock_outputs)
            scale, shift = film(x=dblock_outputs, noise_level=noise_level)
            statistics.append([scale, shift])
        statistics = statistics[::-1]
        
        # Upsampling stream
        ublock_outputs = self.ublock_preconv(x_0_dwt) # (n, 18, 1282) --> (n, 768, 1)
        for i, ublock in enumerate(self.ublocks): 
            scale, shift = statistics[i]
            ublock_outputs = ublock(x=ublock_outputs, scale=scale, shift=shift)
        outputs = self.ublock_postconv(ublock_outputs)
        
        return outputs
    
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        num_channel = len(AVE_CHANNELS_NAME)
        self.cnn = nn.Sequential(
            Conv1dWithInitialization(in_channels=num_channel, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            Conv1dWithInitialization(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            Conv1dWithInitialization(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            Conv1dWithInitialization(in_channels=32, out_channels=16, kernel_size=3),
            nn.BatchNorm1d(16), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            Conv1dWithInitialization(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            Conv1dWithInitialization(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            Conv1dWithInitialization(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        neuron_each_channel = NEW_FREQUENCY*DURATION
        for _ in range(7):
            neuron_each_channel = (neuron_each_channel - 2) // 2

        self.linear_layer = nn.Linear(128*neuron_each_channel,1)
        
        self.apply(self._init_weights)
        torch.nn.init.xavier_normal_(self.linear_layer.weight, gain = nn.init.calculate_gain('sigmoid'))
        
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        x = self.linear_layer(x)
        return x

    # For initialize the weight
    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            torch.nn.init.xavier_normal_(module.weight, gain = nn.init.calculate_gain('relu'))
            if module.bias is not None:  # Initialize the bias as 1
                module.bias.data.fill_(1) 

    

# Reference from https://spotintelligence.com/2023/01/31/self-attention/
class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim=None):
        super(SelfAttention, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.query = nn.Linear(input_dim, output_dim)      
        self.key   = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.attention = nn.MultiheadAttention(output_dim, 
                                               1,
                                               batch_first=True)

        
    def forward(self, x, need_weights=False):
        queries = self.query(x)       # N, 32n, x
        keys    = self.key(x)
        values  = self.value(x)
        if need_weights:
            return self.attention(queries, keys, values, need_weights=need_weights) # Return both output and weight
        return self.attention(queries, keys, values, need_weights=need_weights)[0]  # Only return the output


  
class SateLight(nn.Module):
    def __init__(self):
        super(SateLight, self).__init__()
        fs_divide_2         = DURATION*NEW_FREQUENCY //2
        num_channel         = len(AVE_CHANNELS_NAME)
        self.two_con2D = nn.Sequential(
            nn.Conv2d(1, 16, (1, fs_divide_2)),
            nn.Conv2d(16, 32, (num_channel, 1), groups=16)
        )
        self.batchNorm  = nn.BatchNorm1d(32)
        self.relu       = nn.ReLU()
        self.dropout    = nn.Dropout(0.2)
        self.maxpooling = nn.MaxPool1d(4)
        
        # Self Attention with Residual Blocks
        self.attention_blocks         = []
        self.fully_connections_blocks = []
        for n in range(1,4):
            self.attention_blocks.append(nn.Sequential(
                SelfAttention(32*n),
                nn.BatchNorm1d(fs_divide_2//4//4**(n-1)),
                nn.Dropout(0.2)
            ))
            self.fully_connections_blocks.append(nn.Sequential(
                nn.Linear(32*n, 32*(n+1)),
                nn.BatchNorm1d(fs_divide_2//4//4**(n-1)),
                nn.ReLU(),
                nn.Dropout(0.2)
            ))
        self.attention_blocks         = nn.ModuleList(self.attention_blocks)
        self.fully_connections_blocks = nn.ModuleList(self.fully_connections_blocks)
        
        
        self.classification_head = nn.Linear(256, 1)
        
        self.apply(self._init_weights)
        torch.nn.init.xavier_normal_(self.classification_head.weight, gain = nn.init.calculate_gain('sigmoid'))

    def forward(self,x):
        # Make a downsample with a factor of 8
        x = x[:,None,:,:]      # Extend 1 dimension, shape = (BATCH_SIZE, 1, num_channel, DURATION*NEW_FREQUENCY)
        x = self.two_con2D(x)  # Output Shape = (BATCH_SIZE, 32, 1, DURATION*NEW_FREQUENCY//4)
        x = torch.squeeze(x,2) # Remove extra dimension at dim 2
        x = self.batchNorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpooling(x)
        x = x.permute(0,2,1)  # Output Shape = (BATCH_SIZE, DURATION*NEW_FREQUENCY//4, 32]

        for i in range(3):
            x1 = self.attention_blocks[i](x)
            x  = x+x1                              # Output shape when i=0,1,2 = [[8, 160, 32], [8, 40, 64], [8, 10, 96]]
            
            x1 = self.fully_connections_blocks[i](x)
            x  = F.pad(x, (16,16), "constant", 0)  # Add padding at last dimension with left padding = 16, right padding=16
            x  = x+x1                              # Output shape when i=0,1,2 = [[8, 160, 64], [8, 40, 96], [8, 10, 128]]
            
            x  = self.maxpooling(x.permute(0,2,1))
            x  = x.permute(0,2,1)                   # Output shape when i=0,1,2 = [[8, 40, 64], [8, 10, 96], [8, 2, 128]]

        x = x.reshape(x.shape[0], -1)      # Output Shape = (BATCH_SIZE, 256)
        x = self.classification_head(x) # Output Shape = (BATCH_SIZE, 1)
        
        return x

    # For initialize the weight
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_normal_(module.weight, gain = nn.init.calculate_gain('relu'))
            if module.bias is not None:  # Initialize the bias as 1
                module.bias.data.fill_(1) 

# Features gotten when b = [1,2,3,4,5] are
# 512 * 16, 256 * 80, 128 * 320, 128 * 1280, 19  * 1280
# Num of trainable parameters: 1,869,185  1,537,089  1,770,881  2,348,865  1,545,153
class DiffusionClassifier(nn.Module):
    def __init__(self, config, diffusion_model, device, t_value, b):
        super(DiffusionClassifier, self).__init__()
        self.diffusion_model = diffusion_model
        self.t_value         = t_value
        self.b               = b
        self.device          = device
        
        input_dim            = np.prod(config.factors[:self.b+1])
        if b == 1: 
            self.attentionhead = nn.Sequential(Conv1dWithInitialization(in_channels=config.upsampling_out_channels[b],out_channels=256,
                                                                    kernel_size=3,stride=1,padding="same"), 
                                        nn.LeakyReLU(0.2),
                                        Conv1dWithInitialization(in_channels=256,out_channels=256,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        Conv1dWithInitialization(in_channels=256,out_channels=128,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        Conv1dWithInitialization(in_channels=128,out_channels=128,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        Conv1dWithInitialization(in_channels=128,out_channels=64,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        Conv1dWithInitialization(in_channels=64,out_channels=64,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.Flatten(),
                                        SelfAttention(64 * input_dim, 256)
                                        )
        elif b == 2: 
            self.attentionhead = nn.Sequential(Conv1dWithInitialization(in_channels=config.upsampling_out_channels[b],out_channels=128,
                                                                    kernel_size=3,stride=1,padding="same"), 
                                                nn.LeakyReLU(0.2),
                                                Conv1dWithInitialization(in_channels=128,out_channels=128,
                                                                            kernel_size=3,stride=1,padding="same"), 
                                                nn.LeakyReLU(0.2),
                                                Conv1dWithInitialization(in_channels=128,out_channels=128,
                                                                            kernel_size=3,stride=1,padding="same"), 
                                                nn.LeakyReLU(0.2),
                                                nn.MaxPool1d(2),
                                                Conv1dWithInitialization(in_channels=128,out_channels=64,
                                                                            kernel_size=3,stride=1,padding="same"),
                                                nn.LeakyReLU(0.2),
                                                Conv1dWithInitialization(in_channels=64,out_channels=64,
                                                                            kernel_size=3,stride=1,padding="same"), 
                                                nn.LeakyReLU(0.2),
                                                Conv1dWithInitialization(in_channels=64,out_channels=64,
                                                                            kernel_size=3,stride=1,padding="same"), 
                                                nn.LeakyReLU(0.2),
                                                nn.MaxPool1d(2), 
                                                nn.Flatten(),
                                                SelfAttention(64 * input_dim // 4, 256)
                                                )
        elif b == 3: 
            self.attentionhead = nn.Sequential(Conv1dWithInitialization(in_channels=config.upsampling_out_channels[b],out_channels=256,
                                                                    kernel_size=3,stride=1,padding="same"), 
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=256,out_channels=256,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=256,out_channels=128,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        Conv1dWithInitialization(in_channels=128,out_channels=128,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=128,out_channels=64,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        Conv1dWithInitialization(in_channels=64,out_channels=64,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        nn.Flatten(),
                                        SelfAttention(64 * input_dim // 16, 256)
                                        )
        elif b == 4: 
            self.attentionhead = nn.Sequential(Conv1dWithInitialization(in_channels=config.upsampling_out_channels[b],out_channels=256,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2), 
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=256,out_channels=512,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=512,out_channels=256,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=256,out_channels=128,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=128,out_channels=128,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=128,out_channels=64,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        nn.Flatten(),
                                        SelfAttention(64 * input_dim // 64, 256)
                                        )
        elif b == 5: 
            self.attentionhead = nn.Sequential(Conv1dWithInitialization(in_channels=len(AVE_CHANNELS_NAME),out_channels=32,
                                                                    kernel_size=3,stride=1,padding="same"), 
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=32,out_channels=64,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=64,out_channels=128,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=128,out_channels=256,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=256,out_channels=128,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        Conv1dWithInitialization(in_channels=128,out_channels=64,
                                                                    kernel_size=3,stride=1,padding="same"),
                                        nn.LeakyReLU(0.2),
                                        nn.MaxPool1d(2),
                                        nn.Flatten(),
                                        SelfAttention(64 * input_dim // 64, 256)
                                        )
        else:
            raise(f"Invalid value for b - {b}, should be in range (1-5)")
        self.classifierhead1  = nn.Sequential(nn.Linear(256,128),
                                             nn.Linear(128,64),
                                             nn.Linear(64,32), 
                                             nn.Linear(32, 16)) 
        self.classifierhead2   =  nn.Linear(16,1) 
        
        self.apply(self._init_weights)
        torch.nn.init.xavier_normal_(self.classifierhead2.weight, gain = nn.init.calculate_gain('sigmoid'))
        

    
    def forward(self, x_0_dwt, x_0, label_no=1):
        batch_size = x_0.shape[0]   
        eps        = torch.randn_like(x_0)                          # Sample a random error
        
        # Random draw sample from uniform distribution with low = alphas_prod_p_sqrt[t-1], high = alphas_prod_p_sqrt[t] with size = (batch_size,1)
        continuous_sqrt_alpha_cumprod = self.sample_continuous_noise_level(batch_size, self.t_value).to(self.device)

        x_noisy = self.q_sample(x_0, continuous_sqrt_alpha_cumprod, eps) # Diffuse the signal to x_t, given x_0, alpha & random error
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.reshape(batch_size, 1)

        features = self.get_b_blocks_features(x_0_dwt, 
                                              x_noisy, 
                                              continuous_sqrt_alpha_cumprod, 
                                              torch.full((batch_size, ) ,label_no, dtype=torch.int64).to(device))

        x        = self.attentionhead(features)       
        x        = self.classifierhead1(x)
        x        = self.classifierhead2(x)
        
        return x
    
    def get_b_blocks_features(self, x_0_dwt, x_0, noise_level, label):
        # Downsampling stream + Linear Modulation statistics calculation
        statistics = []

        dblock_outputs = self.diffusion_model.dblock_preconv(x_0)
        if label[0] != 2:    # If it is = 2, no label information passes  dblock_outputs = dblock_outputs + 0 
            dblock_labels  = self.diffusion_model.dblock_label(label)[:,:,None]     # Add 1 more dim and become (Batch_size, 19, 1)
            dblock_outputs = dblock_outputs + dblock_labels         # Add in the information of the label    
            
        scale, shift = self.diffusion_model.films[0](x=dblock_outputs, noise_level=noise_level)
        
        statistics.append([scale, shift])
        for dblock, film in zip(self.diffusion_model.dblocks, self.diffusion_model.films[1:]):
            dblock_outputs = dblock(dblock_outputs)
            scale, shift = film(x=dblock_outputs, noise_level=noise_level)
            statistics.append([scale, shift])
        statistics = statistics[::-1]
        
        
        # Upsampling stream
        ublock_outputs = self.diffusion_model.ublock_preconv(x_0_dwt) # (n, 18, 1282) --> (n, 768, 1)
        for i, ublock in enumerate(self.diffusion_model.ublocks): 
            scale, shift = statistics[i]
            ublock_outputs = ublock(x=ublock_outputs, scale=scale, shift=shift)
            if i == self.b:
                break
        
        if self.b == 5:    
            ublock_outputs = self.diffusion_model.ublock_postconv(ublock_outputs)
        
        
        return ublock_outputs
    
    # For initialize the weight
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain = nn.init.calculate_gain('relu'))
            if module.bias is not None:  # Initialize the bias as 1
                module.bias.data.fill_(1) 
        
    
    
    def sample_continuous_noise_level(self, batch_size, t_value):
        t                             = np.full((batch_size), t_value)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(np.random.uniform(alphas_prod_p_sqrt[t-1], alphas_prod_p_sqrt[t], size=batch_size))
        return continuous_sqrt_alpha_cumprod.unsqueeze(-1).unsqueeze(-1)  # Add 1 more dimension (BATCH_SIZE, 1)

    def q_sample(self, x_0, continuous_sqrt_alpha_cumprod, eps):
        return continuous_sqrt_alpha_cumprod * x_0 + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * eps

