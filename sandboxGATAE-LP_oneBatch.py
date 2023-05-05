"""
Showing how to use the model with some time series data.

NB! This is not a full training loop. You have to write the training loop yourself. 

I.e. this code is just a starting point to show you how to initialize the model and provide its inputs

If you do not know how to train a PyTorch model, it is too soon for you to dive into transformers imo :) 

You're better off starting off with some simpler architectures, e.g. a simple feed forward network, in order to learn the basics
"""

import torch
import torch.nn.functional as F
import argparse
import datetime
import time
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '1'
import math
import pandas as pd
import util.dataset as ds
import util.utils as utils
import layer.TransformerGAT as tst

# from torch_geometric.nn import GATv2Conv
from layer.TransformerGAT import GATAE
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, get_linear_schedule_with_warmup
from util.dataset import StandardScaler
# from sklearn.preprocessing import MinMaxScaler, StandardScaler


torch.manual_seed(42)
np.random.seed(42)


def train(model_temporal, optimizer_teporal, model_spatial, optimizer_spatial, training_time_data, train_data, src_mask, tgt_mask, loss_fn, scheduler, batch_first, input_size, LINEAR_DECODER=False):
    model_spatial.train()
    model_temporal.train() # Turn on the train mode \o/
    start_time = time.time()
    total_loss = 0.


    for step, batch in enumerate(training_time_data):
        # output = torch.Tensor([]).to(device).requires_grad_(True)
        
        src, trg, trg_y = batch
        if batch_first == False:
            B, N, T, E = trg_y.size()
        else:
            T, B, N, E = trg_y.size()
        if input_size == 1: # todo
            trg_y.unsqueeze(-1) # feature size = 1
        src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)


        # Permute from shape [series batch size, node minibatch size, seq len, num features] to [seq len, series batch size*node minibatch size, num features]
        # Node dimension is put inside the batch, in order to process each node along the time separately
        if batch_first == False:
            src = src.permute(2, 0, 1, 3)
            src = src.view(src.size()[0], src.size()[1] * src.size()[2], src.size()[3])
            # print("src shape changed from {} to {}".format(shape_before, src.shape))

            trg = trg.permute(2, 0, 1, 3)
            trg = trg.view(trg.size()[0], trg.size()[1] * trg.size()[2], trg.size()[3])

            trg_y = trg_y.permute(2, 0, 1, 3)
            trg_y = trg_y.view(trg_y.size()[0], trg_y.size()[1] * trg_y.size()[2], trg_y.size()[3])

        optimizer_teporal.zero_grad()
        optimizer_spatial.zero_grad()

        prediction = model_temporal(
            src=src,
            tgt=trg,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            linear_decoder=LINEAR_DECODER
        )
        # outputTmp = torch.cat((trg, prediction), dim=0) # try [trg||prediction]
        # outputTmp = prediction
        # output = torch.cat((output, outputTmp), dim=1)
        output = prediction + trg 
        
        # # reverse scaler
        # prediction = scaler.inverse_transform(prediction.view(-1, input_size)) # .detach().cpu()
        # prediction = prediction.view(T, -1, E)
        # trg_y = scaler.inverse_transform(trg_y.contiguous().view(-1, input_size))
        # trg_y = trg_y.view(T, -1, E)

        # if batch_first == False:
        #     prediction = prediction.permute(1, 0, 2)
        #     trg_y = trg_y.permute(1, 0, 2)
            
        # loss_temporal = loss_fn(prediction, trg_y)

        if batch_first == False:
            output = output.permute(1, 0, 2)
        output = output.view(B,N,T,E).contiguous().view(-1, T*E)

        A_pred, z = model_spatial(output.to(device), train_data.edge_index.to(device))
        pos_score = torch.sigmoid((z[train_data.edge_index[0]] * z[train_data.edge_index[1]]).sum(dim=1))
        neg_edge_index = negative_sampling(train_data.edge_index, num_nodes=train_data.num_nodes)
        neg_score = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))
        # obtain the adj via edge index
        adj_label = utils.generate_adj_from_edgeidx(train_data.edge_index, len(z))
        adj_label = adj_label.to(device)

        # # link prediction loss
        loss_pre = F.binary_cross_entropy(torch.cat([pos_score, neg_score]), 
                                    torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device))
        # structure reconstruction loss
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1)) # binary_cross_entropy_with_logits
        loss_spatial = loss_pre + re_loss #+ loss_temporal
        loss_spatial.backward()
        optimizer_teporal.step()
        optimizer_spatial.step()
        total_loss += loss_spatial.item()
        
    ave_batch_loss = total_loss / (step+1)
    elapsed = time.time() - start_time
    print('| epoch {:3d} | lr {:02.8f} | {:5.2f} ms | '
            'step loss {:5.5f} | ppl {:8.2f}'.format(
            epoch, scheduler.get_last_lr()[0], # get_lr()
            elapsed, ave_batch_loss, math.exp(ave_batch_loss))) # math.log(cur_loss)
    
    total_loss = 0
    start_time = time.time()
    scheduler.step()

    return z, val_data, test_data

def test(model, data, device):
    model.eval()

    with torch.no_grad():
        z = model(data.x.to(device), data.edge_index.to(device))
        pos_score = torch.sigmoid((z[data.edge_index[0]] * z[data.edge_index[1]]).sum(dim=1))
        neg_edge_index = negative_sampling(data.edge_index, num_nodes=data.num_nodes)
        neg_score = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))

    y_true = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    y_score = torch.cat([pos_score, neg_score]).cpu().numpy()
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return auc, ap

def predict_link(model, data, node_a, node_b, device):
    model.eval()
    with torch.no_grad():
        z = model(data.x.to(device), data.train_pos_edge_index.to(device))
        score = torch.sigmoid((z[node_a] * z[node_b]).sum())
    return score.item()


def expectile_loss(pred, target, expectile_level):
    """
    taking the maximum of two terms:(expectile_level - 1) * abs_errors and expectile_level * errors.
    """
    errors = target - pred
    abs_errors = torch.abs(errors)
    expectile_loss = torch.mean(torch.max((expectile_level - 1) * abs_errors, expectile_level * errors))
    return expectile_loss


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
    argparser.add_argument("--LINEAR_DECODER", type=bool, default=False)
    argparser.add_argument("--num_epochs", type=int, default=50)
    argparser.add_argument("--test_size", type=float, default=0.2)
    argparser.add_argument("--batch_size", type=int, default=1)
    argparser.add_argument("--dim_val", type=int, default=512)
    argparser.add_argument("--n_heads", type=int, default=8)
    argparser.add_argument("--n_decoder_layers", type=int, default=4)
    argparser.add_argument("--n_encoder_layers", type=int, default=4)
    argparser.add_argument("--enc_seq_len", type=int, default=4,
                           help="length of input given to encoder 153")
    argparser.add_argument("--dec_seq_len", type=int, default=2,
                           help="length of input given to decoder 92")
    argparser.add_argument("--output_seq_len", type=int, default=2,
                           help="target sequence length. If hourly data and length = 48, you predict 2 days ahead 48")
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

    # Remove test data from dataset for each node
    ratio = round(slice_size*(1-args.test_size))
    first_round = data.iloc[0:ratio, :]
    for i in range(1,round(len(data)//slice_size)+1):
        first_round = pd.concat([first_round, data.iloc[slice_size*i:slice_size*i+ratio, :]], axis=0)
    training_time_data = first_round
    training_slice_size = ratio

    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc. 
    # Should be training time series data indices only
    training_indices = utils.get_indices_entire_sequence(
        window_size=window_size, 
        step_size=args.step_size,
        slice_size=training_slice_size)

    # looks like normalizing input values curtial for the model
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = StandardScaler()
    # Recover the original values
    # original_data = scaler.inverse_transform(scaled_data)
    map_series = training_time_data[input_variables].values
    labels = training_time_data[args.label_col].values
    
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
    output_data = Data(x=amplitude.view(-1,training_slice_size*input_size),
                edge_index=torch.tensor(edge_list, dtype=torch.long),
                y=torch.tensor(labels, dtype=torch.long))
    # Split edges using RandomLinkSplit
    split = RandomLinkSplit()
    train_data, val_data, test_data = split(output_data)

    # Making instance of custom dataset class
    training_time_data = ds.TransformerDataset(
        data=train_data.x.float().view(-1,input_size),
        indices=training_indices,
        enc_seq_len=args.enc_seq_len,
        dec_seq_len=args.dec_seq_len,
        target_seq_len=args.output_seq_len,
        slice_size=training_slice_size
        )

    # Making dataloader
    training_time_data = DataLoader(training_time_data, args.batch_size, shuffle=False) #cannot shuffle time series

    model_temporal = tst.TimeSeriesTransformer(
        input_size=input_size,
        batch_first=args.batch_first,
        num_predicted_features=input_size # 1 if univariate
        ).to(device)
    
    model_spatial = GATAE(in_channels=input_size, hidden_channels=args.dim_val, out_channels=input_size, output_seq_len=args.output_seq_len).to(device)


    # Make src mask for decoder with size:
    # [batch_size*n_heads, output_seq_len, enc_seq_len]
    src_mask = utils.generate_square_subsequent_mask(
        dim1=args.output_seq_len,
        dim2=args.enc_seq_len
        ).to(device)

    # Make tgt mask for decoder with size:
    # [batch_size*n_heads, output_seq_len, output_seq_len]
    tgt_mask = utils.generate_square_subsequent_mask(
        dim1=args.output_seq_len,
        dim2=args.output_seq_len
        ).to(device)

    # loss_fn = torch.nn.HuberLoss().to(device)
    loss_fn = torch.nn.MSELoss().to(device)

    optimizer_temporal = torch.optim.AdamW(model_temporal.parameters(), lr=1e-5)
    optimizer_spatial = torch.optim.AdamW(model_spatial.parameters(), lr=0.005) # , weight_decay=5e-4

    # Define the warm-up schedule
    # total_steps = len(training_time_data) * num_epochs
    # Create the scheduler
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=num_epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_temporal, args.num_epochs//10, gamma=0.95) # args.num_epochs//10


    for epoch in range(args.num_epochs):
        output, val_data, test_data = train(model_temporal, optimizer_temporal, model_spatial, optimizer_spatial, training_time_data, train_data, src_mask, tgt_mask, loss_fn, scheduler, args.batch_first, input_size, args.LINEAR_DECODER)

        if epoch == args.num_epochs-1:
            print('hidden embeddings of epoch {}: {}'.format(epoch, scaler.inverse_transform(output.view(-1, input_size)))) # .detach().cpu()
                
        if (epoch+1) % 10 == 0:
            # Save the model
            torch.save(model_temporal.state_dict(), 'model/model4D_{}_{}.pth'.format(args.enc_seq_len, args.output_seq_len))
            torch.save(model_spatial.state_dict(), 'model/modelSpatial_{}_{}.pth'.format(args.enc_seq_len, args.output_seq_len))
            # model.load_state_dict(torch.load('model.pth'))

            # # evaluation
            # auc, ap = test(model, test_data, device)
            # print(f'Epoch: {epoch}, AUC: {auc:.4f}, AP: {ap:.4f}')
