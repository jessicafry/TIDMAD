import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import math
import torchsnooper
from torch.utils.checkpoint import checkpoint

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
    # @torchsnooper.snoop()
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

class CausalConv1d(nn.Module):
    """Causal convolution that ensures no future information leakage"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)
        
    def forward(self, x):
        x = self.conv(x)
        # Remove future padding to maintain causality
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class WaveNetBlock(nn.Module):
    """Single WaveNet residual block with dilated causal convolution"""
    def __init__(self, residual_channels, gate_channels, skip_channels, kernel_size, dilation):
        super(WaveNetBlock, self).__init__()
        
        # Dilated causal convolution
        self.causal_conv = CausalConv1d(residual_channels, gate_channels, 
                                        kernel_size, dilation)
        
        # Gated activation (tanh and sigmoid gates)
        self.gate_conv = nn.Conv1d(gate_channels // 2, gate_channels // 2, 1)
        self.filter_conv = nn.Conv1d(gate_channels // 2, gate_channels // 2, 1)
        
        # Output projections
        self.residual_conv = nn.Conv1d(gate_channels // 2, residual_channels, 1)
        self.skip_conv = nn.Conv1d(gate_channels // 2, skip_channels, 1)
        
    def forward(self, x):
        # Store residual
        residual = x
        
        # Dilated causal convolution
        x = self.causal_conv(x)
        
        # Split for gated activation
        filter_gate, gate_gate = torch.chunk(x, 2, dim=1)
        
        # Gated activation: tanh(filter) * sigmoid(gate)
        filter_gate = torch.tanh(self.filter_conv(filter_gate))
        gate_gate = torch.sigmoid(self.gate_conv(gate_gate))
        x = filter_gate * gate_gate
        
        # Skip connection
        skip = self.skip_conv(x)
        
        # Residual connection
        residual_out = self.residual_conv(x)
        if residual_out.size(-1) != residual.size(-1):
            # Handle size mismatch due to convolutions
            residual = residual[:, :, :residual_out.size(-1)]
        
        return residual + residual_out, skip

class SimpleWaveNet(nn.Module):
    """Simplified WaveNet model"""
    def __init__(self, 
                 input_channels=16,
                 residual_channels=32,
                 gate_channels=64,
                 skip_channels=32,
                 output_channels=256,
                 kernel_size=12,
                 num_blocks=10):
        super(SimpleWaveNet, self).__init__()
        

         # Embedding layer
        self.embedding = nn.Embedding(256, input_channels)

        # Input projection
        self.input_conv = nn.Conv1d(input_channels, residual_channels, 1)
        
        # WaveNet blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        dilations = [2**i for i in range(num_blocks)]
        
        for dilation in dilations:
            block = WaveNetBlock(residual_channels, gate_channels, 
                                skip_channels, kernel_size, dilation)
            self.blocks.append(block)
        
        # Output layers
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, output_channels, 1)
    # @torchsnooper.snoop()
    def forward(self, x):

        # x = x.float()
        # Handle both (batch, sequence_length) and (batch, sequence_length, 1) inputs
        # if x.dim() == 2:
        #     # Input: (batch, sequence_length) -> add channel dimension
        #     x = x.unsqueeze(-1)  # (batch, sequence_length, 1)
        
        x = self.embedding(x.int())


        # Expected input: (batch, sequence_length, 1)
        # Convert to (batch, channels, sequence_length) for conv1d
        x = x.transpose(1, 2)  # (batch, 1, sequence_length)
        
        # Input projection
        x = self.input_conv(x)
        
        # Accumulate skip connections
        skip_connections = None
        
        # Process through WaveNet blocks
        for block in self.blocks:
            x, skip = block(x)
            
            if skip_connections is None:
                skip_connections = skip
            else:
                # Handle different sequence lengths
                min_len = min(skip_connections.size(-1), skip.size(-1))
                skip_connections = skip_connections[:, :, :min_len] + skip[:, :, :min_len]
        
        # Output processing
        x = F.relu(skip_connections)
        x = F.relu(self.output_conv1(x))
        x = self.output_conv2(x)
        
        # Convert back to (batch, sequence_length, channels)
        # x = x.transpose(1, 2)
        # assert 0
        return x

class Seq2SeqEncoder(nn.Module):
    """Encoder RNN that processes the input sequence"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.1):
        super(Seq2SeqEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input sequence (batch, seq_len)
        Returns:
            outputs: All hidden states (batch, seq_len, hidden_dim)
            hidden: Final hidden state (num_layers, batch, hidden_dim)
            cell: Final cell state (num_layers, batch, hidden_dim)
        """
        # Embed input
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        
        return outputs, hidden, cell

class Seq2SeqDecoder(nn.Module):
    """Decoder RNN that generates the output sequence"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.1):
        super(Seq2SeqDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Embedding layer (can be same as encoder or different)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden, cell):
        """
        Single step forward (for inference)
        Args:
            input_token: Current input token (batch, 1)
            hidden: Hidden state from previous step
            cell: Cell state from previous step
        Returns:
            output: Logits for next token (batch, 1, num_classes)
            hidden: Updated hidden state
            cell: Updated cell state
        """
        # Embed input token
        embedded = self.embedding(input_token)  # (batch, 1, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.dropout(output)
        
        # Project to output classes
        logits = self.output_proj(output)  # (batch, 1, num_classes)
        
        return logits, hidden, cell
    
    def forward_sequence(self, input_sequence, hidden, cell):
        """
        Process entire sequence at once (for training with teacher forcing)
        Args:
            input_sequence: Input tokens (batch, seq_len)
            hidden: Initial hidden state
            cell: Initial cell state
        Returns:
            outputs: All logits (batch, seq_len, num_classes)
        """
        # Embed input sequence
        embedded = self.embedding(input_sequence)  # (batch, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        outputs = self.dropout(outputs)
        
        # Project to output classes
        logits = self.output_proj(outputs)  # (batch, seq_len, num_classes)
        
        return logits

# Simplified version for your specific use case (input and output same length)
class RNNSeq2Seq(nn.Module):
    """
    Simplified Seq2Seq for same-length input/output
    Input: (batch, seq_len)
    Output: (batch, seq_len, num_classes)
    """
    def __init__(self,
                 vocab_size=256,
                 num_classes=256,
                 embedding_dim=128,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.1):
        super(RNNSeq2Seq, self).__init__()

        self.encoder = Seq2SeqEncoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = Seq2SeqDecoder(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout)

    # @torchsnooper.snoop()
    def forward(self, x):
        """
        For your specific case: same length input/output
        Args:
            x: Input sequence (batch, seq_len)
        Returns:
            outputs: Output logits (batch, seq_len, num_classes)
        """
        batch_size, seq_len = x.shape

        # Encode
        encoder_outputs, hidden, cell = self.encoder(x)

        # Decode: use input sequence shifted as decoder input
        # This is a simplification - in practice you might want different logic
        decoder_input = x  # Using same sequence as decoder input
        outputs = self.decoder.forward_sequence(decoder_input, hidden, cell)

        return outputs.transpose(1,2)


# ============================================================
# S4D State Space Model Components
# Adapted from: https://github.com/chreissel/neutrino_project
# ============================================================

class DropoutNd(nn.Module):
    """N-dimensional dropout with tied mask across sequence dimensions."""

    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be in [0, 1), but got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, X):
        """X: (batch, dim, lengths...) when transposed=True."""
        if self.training:
            # Mask shape ties across sequence length when tie=True
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
        return X


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters.

    Uses Vandermonde multiplication with learnable log-dt, C, A parameters
    to produce a length-L convolution kernel in O(N log L) time via FFT.
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self._register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * torch.arange(N // 2).unsqueeze(0).expand(H, -1).float()
        self._register("log_A_real", log_A_real, lr)
        self._register("A_imag", A_imag, lr)

    def forward(self, L):
        """Returns kernel of shape (H, L)."""
        dt = torch.exp(self.log_dt)                          # (H,)
        C = torch.view_as_complex(self.C)                    # (H, N/2)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N/2)

        # Vandermonde multiplication to build the kernel
        dtA = A * dt.unsqueeze(-1)                           # (H, N/2)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H, N/2, L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn,hnl->hl", C, torch.exp(K)).real  # (H, L)
        return K

    def _register(self, name, tensor, lr=None):
        """Register a tensor with optional per-parameter learning rate and zero weight decay."""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    """S4D layer: efficient SSM with diagonal parameterization and spectral convolution.

    The forward pass computes the SSM convolution kernel in the frequency domain
    (via FFT), multiplies with the input spectrum, and transforms back via iFFT.
    A skip connection (D term) and pointwise output projection with GLU gating
    complete the layer.

    Input/output shape: (B, H, L) when transposed=True (default).
    """

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        # Skip connection parameter
        self.D = nn.Parameter(torch.randn(self.h))

        # SSM kernel generator
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise activation and dropout
        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # Output projection with GLU gating
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):
        """
        Args:
            u: Input tensor of shape (B, H, L) [transposed=True] or (B, L, H).
        Returns:
            y: Output tensor of same shape as u.
            None: Placeholder for state (compatible with recurrent interfaces).
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM convolution kernel in frequency domain (spectral convolution)
        k = self.kernel(L=L)                         # (H, L)
        k_f = torch.fft.rfft(k, n=2 * L)            # (H, L+1) complex
        u_f = torch.fft.rfft(u, n=2 * L)            # (B, H, L+1) complex
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B, H, L)

        # D-term skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)

        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None


class MixtureMSESpectralLoss(nn.Module):
    """Hybrid loss combining time-domain MSE and frequency-domain spectral loss.

    Computes MSE on raw predictions and MSE on the magnitude of the real FFT,
    then combines them with a weighting parameter alpha:
        loss = alpha * MSE(time_domain) + (1 - alpha) * MSE(|FFT|)

    Args:
        alpha: Weight for the time-domain MSE term (default 0.5).
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(inputs, targets)

        # Spectral loss: compare magnitude spectra along the last (sequence) dimension
        inputs_mag = torch.abs(torch.fft.rfft(inputs, dim=-1))
        targets_mag = torch.abs(torch.fft.rfft(targets, dim=-1))
        spectral_loss = self.mse(inputs_mag, targets_mag)

        return self.alpha * mse_loss + (1.0 - self.alpha) * spectral_loss


class S4DSeq2SeqModel(nn.Module):
    """S4D sequence-to-sequence model for denoising.

    Unlike classification models that pool over time, this model preserves the
    full sequence length and maps each timestep to d_output channels, making it
    suitable for time-series denoising tasks.

    Input shape:  (B, L, d_input)
    Output shape: (B, L, d_output)
    """

    def __init__(self, d_input, d_output, d_model=128, n_layers=6,
                 dropout=0.0, prenorm=False, gradient_checkpointing=False):
        super().__init__()
        self.prenorm = prenorm
        self.gradient_checkpointing = gradient_checkpointing

        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout) if dropout > 0.0 else nn.Identity())

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Args:
            x: (B, L, d_input) float tensor.
        Returns:
            (B, L, d_output) float tensor.
        """
        x = self.encoder(x)       # (B, L, d_model)
        x = x.transpose(-1, -2)   # (B, d_model, L) — transposed format for S4D

        for layer, norm, drop in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            if self.gradient_checkpointing:
                z = checkpoint(lambda inp: layer(inp)[0], z, use_reentrant=False)
            else:
                z, _ = layer(z)
            z = drop(z)
            x = z + x             # residual connection
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)   # (B, L, d_model)
        x = self.decoder(x)        # (B, L, d_output)
        return x


class S4DenoisModel(nn.Module):
    """TIDMAD-compatible wrapper around S4DSeq2SeqModel for ABRA data denoising.

    Accepts the same float input format as the AE/FCNet models (values 0-255
    representing ADC counts), normalizes internally to [0, 1], applies an
    S4D sequence-to-sequence denoising network, and returns output in the
    original 0-255 scale.

    The spectral convolution inside S4D allows the model to capture both
    short- and long-range frequency dependencies, which is important for
    separating narrow-band axion signals from broadband SQUID noise.

    Input/output shape: (B, L) float, values in [0, 255].
    Loss: MixtureMSESpectralLoss (combined time-domain + frequency-domain).
    """

    def __init__(self, d_model=128, n_layers=6, dropout=0.0):
        super().__init__()
        self.s4_model = S4DSeq2SeqModel(
            d_input=1,
            d_output=1,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, x):
        """
        Args:
            x: (B, L) float tensor with ADC values in [0, 255].
        Returns:
            (B, L) float tensor with denoised values in [0, 255].
        """
        x_norm = x / 255.0               # normalize to [0, 1]
        x_in = x_norm.unsqueeze(-1)      # (B, L, 1)
        out = self.s4_model(x_in)        # (B, L, 1)
        return out.squeeze(-1) * 255.0   # (B, L) in [0, 255]
