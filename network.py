import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import math

SEQ_LEN = 40000

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=9, padding=4,bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=9, padding=4,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(4),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=4, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=1),
            # torch.nn.LeakyReLU(),
            # torch.nn.Conv1d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, start=0, dropout=0.1, max_len=SEQ_LEN,factor=1.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.factor = factor

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)
        self.start = start
    # @torchsnooper.snoop()
    def forward(self, x):
        x = x + self.factor*self.pe[:,:,self.start:(self.start+x.size(2))]
        x = self.dropout(x)
        return x
    
class PositionalUNet(nn.Module):
    def __init__(self):
        super(PositionalUNet, self).__init__()
        self.bilinear = True
        
        multi = 40
        ADC_channel = 256 #ABRA ADC Channel
        Embedding_dim = 32

        self.embedding = nn.Embedding(ADC_channel, Embedding_dim, scale_grad_by_freq=True)
        self.inc = DoubleConv(Embedding_dim, multi)
        self.down1 = Down(multi, multi*2)
        self.down2 = Down(multi*2, multi*4)
        self.down3 = Down(multi*4, multi*8)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(multi*8, multi*16 // factor)
              
        self.up1 = Up(multi*16, multi*8 // factor, self.bilinear)
        self.up2 = Up(multi*8, multi*4 // factor, self.bilinear)
        self.up3 = Up(multi*4, multi*2 // factor, self.bilinear)
        self.up4 = Up(multi*2, multi // factor, self.bilinear)
        self.outc = OutConv(multi // factor, ADC_channel)
        self.fc_mean = torch.nn.Conv1d(multi*16 // factor, multi*16 // factor,1)
        self.fc_var = torch.nn.Conv1d(multi*16 // factor, multi*16 // factor,1)
        
        self.pe0 = PositionalEncoding(Embedding_dim)
        self.pe1 = PositionalEncoding(multi)
        self.pe2 = PositionalEncoding(multi*2)
        self.pe3 = PositionalEncoding(multi*4)
        self.pe4 = PositionalEncoding(multi*8)
        self.pe5 = PositionalEncoding(multi*16//factor)
        self.pe6 = PositionalEncoding(multi*8// factor,start=multi*4)
        self.pe7 = PositionalEncoding(multi*4// factor,start=multi*2)
        self.pe8 = PositionalEncoding(multi*2// factor,start=multi*2)
        self.pe9 = PositionalEncoding(multi// factor,start=0,factor=1.0)
        
    
    def reparametrize(self, mu,logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(mu)
        return eps.mul(std).add_(mu)
    
#     @torchsnooper.snoop()
    def forward(self, x):
        '''
        Input: time series data with dimension (time,) or (batch, time,)
        IMPORTANT: all values of the tensor must be POSITIVE INTEGER BETWEEN 0 and 255
        This is to make sure it can go through the embedding layer, for more about embedding, read
        https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

        Output: dimension (time, 256) or (batch, time, 256)
        the last dimension with 256 elements is a "classification dimension", i.e. where should
        we output the time series value at this given time index.

        This network should be trained with a CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss
        '''

        x  = self.pe0(self.embedding(x).transpose(-1,-2))
        x1 = self.pe1(self.inc(x))
        x2 = self.pe2(self.down1(x1))
        x3 = self.pe3(self.down2(x2))
        x4 = self.pe4(self.down3(x3))
        x5 = self.pe5(self.down4(x4))
#         x5 = self.pe5(self.reparametrize(self.fc_mean(x5), self.fc_var(x5)))
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        
#         assert 0
        return output

class FocalLoss1D(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss1D, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Assume inputs and targets are 1D with shape [batch_size, num_classes, length]
        # inputs are logits and targets are indices of the correct class
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        
        # convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 2, 1).float()
        
        # Calculate Focal Loss
        alpha_t = self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot)
        loss = -alpha_t * ((1 - pt) ** self.gamma) * log_pt
        loss = (targets_one_hot * loss).sum(dim=1)  # only keep loss where targets are not zero
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
        
class TransformerModel(nn.Module):

    def __init__(self):
        super(TransformerModel, self).__init__()
        
        
        ADC_channel = 256 #ABRA ADC Channel
        Embedding_dim = 32
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(Embedding_dim)
        # d_model, nhead, d_hid, dropout
        encoder_layers = TransformerEncoderLayer(Embedding_dim, 2, 128, 0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)
        self.embedding = nn.Embedding(ADC_channel, Embedding_dim, scale_grad_by_freq=True)
        self.d_model = Embedding_dim
        self.linear = nn.Linear(Embedding_dim, ADC_channel)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
#     @torchsnooper.snoop()
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = src.transpose(0,1)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.permute(1,2,0)).permute(2,0,1)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output).permute(1,2,0)
        return output
    
class AE(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        scale_factor = [0.1,0.01,0.001]

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, int(input_size*scale_factor[0])),
            torch.nn.ReLU(),
            torch.nn.Linear(int(input_size*scale_factor[0]), int(input_size*scale_factor[1])),
            torch.nn.ReLU(),
            torch.nn.Linear(int(input_size*scale_factor[1]), int(input_size*scale_factor[2])),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(int(input_size*scale_factor[2]), int(input_size*scale_factor[1])),
            torch.nn.ReLU(),
            torch.nn.Linear(int(input_size*scale_factor[1]), int(input_size*scale_factor[0])),
            torch.nn.ReLU(),
            torch.nn.Linear(int(input_size*scale_factor[0]), input_size)
        )

    def forward(self, input_seq):
        # Encode the input sequence
        encoder_outputs = self.encoder(input_seq.float())

        # Decode the encoded sequence
        decoder_outputs= self.decoder(encoder_outputs)

        return decoder_outputs