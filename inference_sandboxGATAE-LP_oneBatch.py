"""
code-ish example of how to use the inference function to do validation
during training. 

The validation loop can be used as-is for model testing as well.

NB! You cannot use this script as is. This is merely an example to show the overall idea - 
not something you can copy paste and expect to work. For instance, see "sandbox.py" 
for example of how to instantiate model and generate dataloaders.

If you have never before trained a PyTorch neural network, I suggest you look
at some of PyTorch's beginner-level tutorials.
"""
import torch
import torch.nn.functional as F
import argparse
import util.inference as inference
import util.utils as utils
import layer.TransformerGAT as tst
import numpy as np
import util.dataset as ds
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from util.dataset import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
# from torch_geometric.nn import GATv2Conv
from layer.TransformerGAT import GATAE
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit


torch.manual_seed(42)
np.random.seed(42)

    

def eval(model_temporal, model_spatial, test_time_data, test_data, src_mask, tgt_mask, batch_first, input_size, SCALER, LINEAR_DECODER):
    # Set the model to evaluation mode
    model_temporal.eval()
    model_spatial.eval()

    for _, batch in enumerate(test_time_data):
        # output = torch.Tensor([]).to(device).requires_grad_(True)
        src, trg, trg_y = batch
        B, N, T, E = trg_y.size()
        if input_size == 1: # feature size = 1
            trg_y.unsqueeze(2)
        src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)
        

        # Permute from shape [batch size, node size, seq len, num features] to [seq len, batch size, node size, num features]
        if batch_first == False:
            src = src.permute(2, 0, 1, 3)
            trg = trg.permute(2, 0, 1, 3)
            trg_y = trg_y.permute(2, 0, 1, 3)

        # inference on the length of the output window
        # [seq len, batch size*node size, num features] 
        # Node dimension is put inside the batch, in order to process each node along the time separately
        src_B = src.view(src.size()[0], src.size()[1] * src.size()[2], src.size()[3])
        trg_B = trg.view(trg.size()[0], trg.size()[1] * trg.size()[2], trg.size()[3])
        trg_y_B = trg_y.view(trg_y.size()[0], trg_y.size()[1] * trg_y.size()[2], trg_y.size()[3])

        prediction = model_temporal(
            src=src_B,
            tgt=trg_B,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            linear_decoder=LINEAR_DECODER
        )
        # outputTmp = torch.cat((trg_B, prediction), dim=0) # try [trg||prediction]
        # outputTmp = prediction
        # output = torch.cat((output, outputTmp), dim=1)
        output = prediction + trg_B

        # # reverse scaler
        # prediction = scaler.inverse_transform(prediction.view(-1, input_size)) # .detach().cpu()
        # prediction = prediction.view(T, -1, E)
        # trg_y = scaler.inverse_transform(trg_y.contiguous().view(-1, input_size))
        # trg_y = trg_y.view(T, -1, E)

        # if batch_first == False:
        #     prediction = prediction.permute(1, 0, 2)
        #     trg_y_B = trg_y_B.permute(1, 0, 2)

        if batch_first == False:
            output = output.permute(1, 0, 2)
        output = output.view(B,N,T,E).contiguous().view(-1, T*E)
        
        auc, ap = test(model_spatial, test_data, output, device)
        #predict_link(model_spatial, test_data, output, 0, 5, device)
        print('| auc {:3f} | ap {:3f} '.format(auc, ap))
        


def test(model, data, x, device):
    model.eval()

    with torch.no_grad():
        _, z = model(x.to(device), data.edge_index.to(device))
        pos_score = torch.sigmoid((z[data.edge_index[0]] * z[data.edge_index[1]]).sum(dim=1))
        neg_edge_index = negative_sampling(data.edge_index, num_nodes=data.num_nodes)
        neg_score = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))

    y_true = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    y_score = torch.cat([pos_score, neg_score]).cpu().numpy()
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return auc, ap

def predict_link(model, data, x, node_a, node_b, device):
    model.eval()
    with torch.no_grad():
        _, z = model(x.to(device), data.edge_index.to(device))
        score = torch.sigmoid((z[node_a] * z[node_b]).sum())
    return score.item()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training"
    )
    argparser.add_argument("--data_file", type=str, default='forecast_cyclical_data_v1',
                           help="file name wo/ suffix")
    argparser.add_argument("--projection_map_file", type=str, default='projection_map_v1',
                           help="file name wo/ suffix")
    argparser.add_argument("--edgeIdx_file", type=str, default='edge_v1',
                           help="file name wo/ suffix")
    argparser.add_argument("--SCALER", type=bool, default=True)
    argparser.add_argument("--PLOT_BIAS", type=bool, default=True)
    argparser.add_argument("--PLOT_PREDICT", type=bool, default=True)
    argparser.add_argument("--LINEAR_DECODER", type=bool, default=False)
    argparser.add_argument("--test_size", type=float, default=0.2)
    argparser.add_argument("--batch_size", type=int, default=1)
    argparser.add_argument("--dim_val", type=int, default=512)
    argparser.add_argument("--n_heads", type=int, default=1)
    argparser.add_argument("--n_decoder_layers", type=int, default=4)
    argparser.add_argument("--n_encoder_layers", type=int, default=4)
    argparser.add_argument("--enc_seq_len", type=int, default=4,
                           help="length of input given to encoder 153")
    argparser.add_argument("--dec_seq_len", type=int, default=2,
                           help="length of input given to decoder 48. Must equal to output_seq_len")
    argparser.add_argument("--output_seq_len", type=int, default=2,
                           help="target sequence length. If hourly data and length = 48, you predict 2 days ahead 48")
    argparser.add_argument("--forecast_step", type=int, default=2,
                           help="window you forecast in future")
    argparser.add_argument("--nodeIdx", type=int, default=1, # 400
                           help="nodeIdx for forecasting plot")
    argparser.add_argument("--step_size", type=int, default=1,
                           help="Step size, i.e. how many time steps does the moving window move at each step")
    argparser.add_argument("--in_features_encoder_linear_layer", type=int, default=2048)
    argparser.add_argument("--in_features_decoder_linear_layer", type=int, default=2048)
    argparser.add_argument("--batch_first", type=bool, default=False)
    argparser.add_argument("--target_col_name", type=str, default="Reliability")
    argparser.add_argument("--timestamp_col", type=str, default="Timestamp")
    argparser.add_argument("--node_col", type=str, default="Node")
    argparser.add_argument("--sort_col", type=str, default="Node Index")
    argparser.add_argument("--label_col", type=str, default="Node Label")
    argparser.add_argument("--exogenous_vars", type=str, default="Flexibility,Service,Infrastructure Quality,Freight,order_rating,Utilization Rate,Political Stability,Economic Stability",
                           help="split by comma, should contain strings. Each string must correspond to a column name")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = torch.device("cuda:%d" % args.gpu)
    else:
        device = torch.device("cpu")

    # Only use data from this date and onwards
    # cutoff_date = datetime.datetime(2017, 1, 1) 

    # Define input variables 
    if args.LINEAR_DECODER:
        args.output_seq_len = args.enc_seq_len
    window_size = args.enc_seq_len + args.output_seq_len # used to slice data into sub-sequences
    exogenous_vars = args.exogenous_vars.split(',')
    input_variables = [args.target_col_name] + exogenous_vars

    # Read data
    # Input x
    # (batch_size, nodes, sequentials, features)
    data, slice_size = utils.read_data(file_name=args.data_file, node_col_name=args.node_col, timestamp_col_name=args.timestamp_col, sort_col=args.sort_col)

    # Get test data from dataset
    ratio = round(slice_size*args.test_size)
    first_round = data.iloc[slice_size-ratio:slice_size, :]
    for i in range(1,round(len(data)//slice_size)+1):
        first_round = pd.concat([first_round, data.iloc[slice_size*(i+1)-ratio:slice_size*(i+1), :]], axis=0)
    test_time_data = first_round
    test_slice_size = ratio
    # test_time_data = data[-(round(len(data)*test_size)):]

    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc. 
    # Should be test data indices only
    test_indices = utils.get_indices_input_target(
        input_len=window_size,
        step_size=window_size,
        slice_size=test_slice_size
    )

    # looks like normalizing input values curtial for the model
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = StandardScaler()
    # Recover the original values
    # original_data = scaler.inverse_transform(scaled_data)
    map_series = test_time_data[input_variables].values
    labels = test_time_data[args.label_col].values
    
    # dic for label wise feature projection, e.g., OrderedDict([(0, 3), (1, 2))])
    dic = utils.read_projection_map(file_name=args.projection_map_file)
    series = np.zeros((len(map_series), sum(dic.values()))) # 0 avoid the impact of -1 values for the scaler func.
    # series = np.full((len(map_series), sum(dic.values())), -1.) # -1 denotes the absence feature of each node
    for i in range(len(series)):
        given_index = labels[i]
        index = utils.index_for_feature_projection(dic, given_index)
        series[i][index:index+dic[given_index]] = map_series[i][map_series[i] != -1]

    # edge index including the start and end nodes.
    edge_list = utils.read_edgeIdx(file_name=args.edgeIdx_file)
    edge_list = edge_list[["startIdx","endIdx"]].values.T
    # labels = np.random.randint(0, 3, 25) # num_classes

    if args.SCALER:
        scaler = StandardScaler(mean=torch.FloatTensor(series).to(device).mean(axis=0), std=torch.FloatTensor(series).to(device).std(axis=0))
        amplitude = scaler.fit_transform(torch.FloatTensor(series).to(device))
    else:
        amplitude = torch.FloatTensor(series).to(device)

    # prepare graph data for link prediction
    input_size = amplitude.size(-1)
        # Create a PyTorch Geometric Data object
    output_data = Data(x=amplitude.view(-1,test_slice_size*input_size),
                edge_index=torch.tensor(edge_list, dtype=torch.long),
                y=torch.tensor(labels, dtype=torch.long))
    # Split edges using RandomLinkSplit
    split = RandomLinkSplit()
    train_data, val_data, test_data = split(output_data)

    # Making instance of custom dataset class
    test_time_data = ds.TransformerDataset(
        data=test_data.x.float().view(-1,input_size),
        indices=test_indices,
        enc_seq_len=args.enc_seq_len,
        dec_seq_len=args.dec_seq_len,
        target_seq_len=args.output_seq_len,
        slice_size=test_slice_size
        )

    # Making dataloader
    test_time_data = DataLoader(test_time_data, args.batch_size)

    # Make src mask for decoder with size:
    # [batch_size*n_heads, dec_seq_len, enc_seq_len]
    src_mask = utils.generate_square_subsequent_mask(
        dim1=args.dec_seq_len,
        dim2=args.enc_seq_len
        ).to(device)

    # Make tgt mask for decoder with size:
    # [batch_size*n_heads, dec_seq_len, dec_seq_len]
    tgt_mask = utils.generate_square_subsequent_mask( 
        dim1=args.dec_seq_len,
        dim2=args.dec_seq_len
        ).to(device)

    # Initialize the model with the same architecture and initialization as when it was saved
    model_temporal = tst.TimeSeriesTransformer(
        input_size=input_size,
        batch_first=args.batch_first,
        n_heads=args.n_heads,
        num_predicted_features=input_size # 1 if univariate
        ).to(device)

    model_spatial = GATAE(in_channels=input_size, hidden_channels=args.dim_val, out_channels=input_size, output_seq_len=args.output_seq_len, nheads=args.n_heads).to(device)

    # Define the file path, same as the forecast_step
    PATH_temporal = 'model/model4D_{}_{}.pth'.format(args.enc_seq_len, args.output_seq_len)
    PATH_spatial = 'model/modelSpatial_{}_{}.pth'.format(args.enc_seq_len, args.output_seq_len)

    # Load the saved state dictionary into the model
    model_temporal.load_state_dict(torch.load(PATH_temporal))
    model_spatial.load_state_dict(torch.load(PATH_spatial))
    # Load the state dict into the model
    # state_dict  = torch.load(PATH, map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)


    # Iterate over all (x,y) pairs in validation dataloader
    with torch.no_grad():
        eval(model_temporal, model_spatial, test_time_data, test_data, src_mask, tgt_mask, args.batch_first, input_size, args.SCALER, args.LINEAR_DECODER)

