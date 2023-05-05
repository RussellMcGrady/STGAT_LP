
import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
import numpy as np

from torch import nn, Tensor
from layer.positional_encoder import PositionalEncoder
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.nn import SAGEConv


class TimeSeriesTransformer(nn.Module):

    """
    This class implements a transformer model that can be used for times series
    forecasting. This time series transformer model is based on the paper by
    Wu et al (2020) [1]. The paper will be referred to as "the paper".

    A detailed description of the code can be found in my article here:

    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e

    In cases where the paper does not specify what value was used for a specific
    configuration/hyperparameter, this class uses the values from Vaswani et al
    (2017) [2] or from PyTorch source code.

    Unlike the paper, this class assumes that input layers, positional encoding 
    layers and linear mapping layers are separate from the encoder and decoder, 
    i.e. the encoder and decoder only do what is depicted as their sub-layers 
    in the paper. For practical purposes, this assumption does not make a 
    difference - it merely means that the linear and positional encoding layers
    are implemented inside the present class and not inside the 
    Encoder() and Decoder() classes.

    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020). 
    'Deep Transformer Models for Time Series Forecasting: 
    The Influenza Prevalence Case'. 
    arXiv:2001.08317 [cs, stat] [Preprint]. 
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).

    [2] Vaswani, A. et al. (2017) 
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint]. 
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).

    """

    def __init__(self, 
        input_size: int,
        batch_first: bool,
        dec_seq_len: int=58,
        out_seq_len: int=58,
        dim_val: int=512,  
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2, 
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        num_predicted_features: int=1
        ): 

        """
        Args:

            input_size: int, number of input variables. 1 if univariate.

            dec_seq_len: int, the length of the input sequence fed to the decoder

            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val

            n_encoder_layers: int, number of stacked encoder layers in the encoder

            n_decoder_layers: int, number of stacked encoder layers in the decoder

            n_heads: int, the number of attention heads (aka parallel attention layers)

            dropout_encoder: float, the dropout rate of the encoder

            dropout_decoder: float, the dropout rate of the decoder

            dropout_pos_enc: float, the dropout rate of the positional encoder

            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder

            dim_feedforward_decoder: int, number of neurons in the linear layer 
                                     of the decoder

            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """

        super().__init__() 

        self.dec_seq_len = dec_seq_len

        #print("input_size is: {}".format(input_size))
        #print("dim_val is: {}".format(dim_val))

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size, 
            out_features=dim_val 
            )

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
            )  
        
        self.linear_mapping = nn.Linear(
            in_features=dim_val, 
            out_features=num_predicted_features
            )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
            )
        # # Create positional decoder
        self.positional_decoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
            )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
            )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerEncoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(dim_val)
            )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
            )

        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerDecoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=n_decoder_layers,
            norm=nn.LayerNorm(dim_val)
            )
        
        # diverse decoder trial
        
        self.fcn = nn.Conv1d(dim_val, num_predicted_features, kernel_size=1)
        self.conv_init(self.fcn)
        
        self.lineardecoder = nn.Linear(dim_val,num_predicted_features)
        self.init_weights()

    def conv_init(self, module):
        # he_normal
        n = module.out_channels
        for k in module.kernel_size:
            n = n*k
        module.weight.data.normal_(0, math.sqrt(2. / n))

    def init_weights(self):
        initrange = 0.1    
        self.lineardecoder.bias.data.zero_()
        self.lineardecoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None, 
                tgt_mask: Tensor=None, linear_decoder: bool=False) -> Tensor:
        """
        Returns a tensor of shape:

        [target_sequence_length, batch_size, num_predicted_features]
        
        Args:

            src: the encoder's output sequence. Shape: (S,E) for unbatched input, 
                 (S, N, E) if batch_first=False or (N, S, E) if 
                 batch_first=True, where S is the source sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)

            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input, 
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if 
                 batch_first=True, where T is the target sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)

            src_mask: the mask for the src sequence to prevent the model from 
                      using data points from the target sequence

            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence


        """

        #print("From model.forward(): Size of src as given to forward(): {}".format(src.size()))
        #print("From model.forward(): tgt size = {}".format(tgt.size()))

        # Pass throguh the input layer right before the encoder
        src = self.encoder_input_layer(src) # src shape: [src length, batch_size, dim_val] regardless of number of input features
        #print("From model.forward(): Size of src after input layer: {}".format(src.size()))

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(src) # src shape: [src length, batch_size, dim_val] regardless of number of input features
        #print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length. 
        # (https://github.com/huggingface/transformers/issues/4083)
        src = self.encoder( # src shape: [enc_seq_len, batch_size, dim_val]
            src=src
            )
        #print("From model.forward(): Size of src after encoder: {}".format(src.size()))

        if linear_decoder:
            decoder_output = self.lineardecoder(src)

            # 1 dimension conv
            # decoder_output = self.fcn(src.permute(0,2,1)).permute(0,2,1)
            # decoder_output = F.avg_pool1d(decoder_output, 1)
            # decoder_output = decoder_output.view(decoder_output.size(-1,3,2))
        else:
            # Pass decoder input through decoder input layer
            decoder_output = self.decoder_input_layer(tgt) # src shape: [target sequence length, batch_size, dim_val] regardless of number of input features
            #print("From model.forward(): Size of decoder_output after linear decoder layer: {}".format(decoder_output.size()))

            # Pass through the positional encoding layer
            decoder_output = self.positional_decoding_layer(decoder_output) # src shape: [src length, batch_size, dim_val] regardless of number of input features
            #print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))

            #if src_mask is not None:
                #print("From model.forward(): Size of src_mask: {}".format(src_mask.size()))
            #if tgt_mask is not None:
                #print("From model.forward(): Size of tgt_mask: {}".format(tgt_mask.size()))

            # Pass throguh decoder - output shape: [target seq len, batch_size, dim_val]
            decoder_output = self.decoder(
                tgt=decoder_output,
                memory=src,
                tgt_mask=tgt_mask,
                memory_mask=src_mask
                )

            #print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))

            # Pass through linear mapping
            decoder_output = self.linear_mapping(decoder_output) # shape [target seq len, batch_size, feature_dim]
            #print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))

        return decoder_output


class GATAE(torch.nn.Module):
    """
    Two GATv2 layer, https://github.com/tech-srl/how_attentive_are_gats
    """
    def __init__(self, in_channels, hidden_channels, out_channels, output_seq_len=1, nheads=8, dropout=0.2, concat=True):
        super(GATAE, self).__init__()
        self.conv1 = GATv2Conv(output_seq_len*in_channels, hidden_channels, heads=nheads, dropout=dropout, concat=concat)
        self.conv2 = GATv2Conv(nheads * hidden_channels, output_seq_len*out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = F.elu(self.conv1(x, edge_index))
        x = self.conv1(x, edge_index)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        z = F.normalize(x, p=2, dim=1)  # Apply L2 normalization
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
    

# class GraphAttentionLayer(nn.Module):
#     """
#     Two GATv2 layer, https://github.com/tech-srl/how_attentive_are_gats
#     """

#     def __init__(self, in_features, hidden_features, out_features, nheads=8, dropout=0.2, alpha=0.2, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat

#         self.conv1 = GATv2Conv(in_features, hidden_features, heads=nheads, dropout=dropout, concat=concat)
#         self.conv2 = GATv2Conv(hidden_features * nheads, out_features, heads=nheads, concat=concat)

#         self.leakyrelu = nn.LeakyReLU(self.alpha)


#     def forward(self, input, edge_index):
#         #batch_size = input.size(0)
#         #h = torch.bmm(input, self.W.expand(batch_size, self.in_features, self.out_features))
#         #f_1 = torch.bmm(h, self.a1.expand(batch_size, self.out_features, 1))
#         #f_2 = torch.bmm(h, self.a2.expand(batch_size, self.out_features, 1))
#         #e = self.leakyrelu(f_1 + f_2.transpose(2,1))
#         # add by xyk
#         #attention = torch.mul(adj, e)
#         #attention = F.softmax(attention, dim=1)
#         #attention = F.dropout(attention, self.dropout, training=self.training)
#         #h_prime = torch.bmm(attention, h) + self.bias.expand(batch_size, self.num_nodes, self.out_features)
#         #if input.shape[-1] != h_prime.shape[-1]:
#             #input = self.downsample(input.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
#             #h_prime = h_prime + input
#         #else:
#             #h_prime = h_prime + input

#         # obtain the indices of the non-zero elements
#         # row, col = torch.where(adj != 0)
#         # concatenate row and col tensors to obtain edge index
#         # edge_index = torch.stack([row, col], dim=0)

#         # Convert input tensor to 2D feature matrix
#         in_channels, batch_size, num_features = input.shape # in_channels refer to the seq_len
#         features = input.view(batch_size, in_channels * num_features)

#         h = self.conv1(features, edge_index)
#         #x = torch.relu(x)
#         h_prime = self.conv2(h, edge_index)

#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class STGATBlock(nn.Module):
    def __init__(self, cuda, in_channels, feature_channels, hidden_channels, output_seq_len=1,
                dropout=0.2, nheads=8, concat=True, batch_first=False):
        super(STGATBlock, self).__init__()
        self.output_seq_len = output_seq_len
        self.nheads = nheads
        self.concat = concat
        self.cuda = cuda
        self.batch_first = batch_first

        self.temporal = [TimeSeriesTransformer(
            input_size=in_channels,
            batch_first=batch_first,
            num_predicted_features=feature_channels # 1 if univariate
            )]
        self.spatial = [GATAE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=feature_channels,
            output_seq_len=output_seq_len,
            nheads=nheads,
            dropout=dropout,
            concat=concat
            )]

        if cuda:
            self.temporal.to(self.cuda)
            self.spatial.to(self.cuda)
   
    def forward(self, src, trg, src_mask, tgt_mask, edge_index): # edge_index of size (2, num_edges)
        # residual = X #todo
        if self.batch_first == False:
            B, N, T, E = trg.size() # B == 1 train on a graph for each batch
        else:
            T, B, N, E = trg.size()

        t = self.temporal(
            src=src,
            tgt=trg,
            src_mask=src_mask,
            tgt_mask=tgt_mask
            )
        if self.batch_first == False:
            t = t.permute(1, 0, 2)
        t = t.view(B,N,T,E)
        t = t.contiguous().view(-1, T*E)

        A_pred, st = self.spatial(t, edge_index)

        return A_pred, st


class GatedLinearUnits(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels=16, kernel_size=2, dilation=1, groups=4, activate=False):
        super(GatedLinearUnits, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activate = activate

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=True, groups=groups)
        nn.init.xavier_uniform_(self.conv.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.conv.bias, 0.1)
        self.gate = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=True, groups=groups)
        nn.init.xavier_uniform_(self.gate.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.gate.bias, 0.1)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1, bias=True)
        nn.init.xavier_uniform_(self.downsample.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.downsample.bias, 0.1)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

        self.sigmod = nn.Sigmoid()
        
    def forward(self, X):
        res = X
        gate = X
        
        X = nn.functional.pad(X, ((self.kernel_size-1)*self.dilation, 0, 0, 0))
        out = self.conv(X)
        if self.activate:
            out = F.tanh(out)

        gate = nn.functional.pad(gate, ((self.kernel_size-1)*self.dilation, 0, 0, 0))
        gate = self.gate(gate)
        gate = self.sigmod(gate)

        out = torch.mul(out, gate)
        ones = torch.ones_like(gate)

        if res.shape[1] != out.shape[1]:
            res = self.downsample(res)
        
        res = torch.mul(res, ones-gate)
        out = out + res
        out = self.relu(self.bn(out))
        return out


class EndConv(nn.Module):
    def __init__(self, in_channels, out_channels, nhid_channels, layer=3):
        super(EndConv, self).__init__()
        layers = []
        for i in range(layer):
            # print('in_channels', in_channels)
            if i == 0:
                layers.append(GatedLinearUnits(in_channels, nhid_channels, kernel_size=1, dilation=1, groups=1))
            else:
                layers.append(GatedLinearUnits(nhid_channels, nhid_channels, kernel_size=3, dilation=1, groups=1))
        layers.append(nn.Conv1d(nhid_channels, out_channels, 1)) ## todo
        self.units = nn.Sequential(*layers)
    
    def forward(self, X):
        out = self.units(X)
        return out
    

class TransformerGAT(nn.Module):
    def __init__(self, cuda, in_channels, feature_channels, hidden_channels, output_seq_len=1,
                 dropout=0.2, nheads=8, concat=True, batch_first=False, layers=1):
        super(TransformerGAT, self).__init__()
        self.cuda_device = cuda
        self.layers = layers
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(STGATBlock(cuda, in_channels, feature_channels, hidden_channels, output_seq_len=output_seq_len,
                dropout=dropout, nheads=nheads, concat=concat, batch_first=batch_first)
                )
        self.output = EndConv(hidden_channels, feature_channels, hidden_channels)
        


    
    def forward(self, src, trg, src_mask, tgt_mask, edge_index):
        if self.batch_first == False:
            B, N, T, E = trg.size() # B == 1 train on a graph for each batch
        else:
            T, B, N, E = trg.size()
    
        for i in range(self.layers):
            A_pred, st = self.blocks[i](src, trg, src_mask, tgt_mask, edge_index)
        emb = st
        emb = emb.reshape((B, N, T, E)).unsqueeze().permute(0, 2, 1)

        logits = self.output(emb).permute(0, 2, 1).unsqueeze(dim=3)
        if self.training:
            return A_pred, logits
        else:
            return st
        

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super(GraphSAGEModel, self).__init__()
        self.sage_conv1 = SAGEConv(in_channels, hidden_channels)
        self.dropout = dropout
        self.sage_conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sage_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage_conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    