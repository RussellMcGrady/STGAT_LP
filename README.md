# How to code a GAT-based autoencoder model using time series data for link prediction in PyTorch
## PyTorch implementation of STGAT model for link on heterogeneous nodes with multiple features"

This is the repo of the STGAT model for link prediction on KG

The sandbox*.py file shows how to use the model conduct link prediction on the data from the .csv file in "/data".

The inference_sandbox*.py file contains the function that takes care of inference. 

Noteworthy, the learning rate to be smaller than 1e-5 if use the transformer decoder, otherwise may lead to overfitting.

## install the required package in requirements.txt
pip install -r requirements.txt

## train
python sandboxGATAE-LP_oneBatch

## test
python inference_sandboxGATAE-LP_oneBatch.py

## pseudocode

## Define the input dimensions

    seq_len = 42
    embedding_dim = 512
    feature_dim = 12

## Define the transformer model

class TimeSeriesTransformer(nn.Module):

        super().__init__() 

        self.dec_seq_len = dec_seq_len

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
        # Create positional decoder
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


## Define the GAT autoencoder
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