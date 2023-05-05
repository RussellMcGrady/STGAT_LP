import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from collections import OrderedDict
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 


    Return:

        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def get_indices_input_target(input_len, step_size, slice_size):
    """
    Produce all the start and end index positions of all sub-sequences.
    The indices will be used to split the data into sub-sequences on which 
    the models will be trained. 

    Returns a tuple with four elements:
    1) The index position of the first element to be included in the input sequence
    2) The index position of the last element to be included in the input sequence
    3) The index position of the first element to be included in the target sequence
    4) The index position of the last element to be included in the target sequence

    
    Args:
        num_obs (int): Number of observations in the entire dataset for which
                        indices must be generated.

        input_len (int): Length of the input sequence (a sub-sequence of 
                            of the entire data sequence)

        step_size (int): Size of each step as the data sequence is traversed.
                            If 1, the first sub-sequence will be indices 0-input_len, 
                            and the next will be 1-input_len.

        forecast_horizon (int): How many index positions is the target away from
                                the last index position of the input sequence?
                                If forecast_horizon=1, and the input sequence
                                is data[0:10], the target will be data[11:taget_len].

        target_len (int): Length of the target / output sequence.

        slice_size (int): num of series slice for each node.
    """
    input_len = round(input_len) # just a precaution
    start_position = 0
    stop_position = slice_size
    
    subseq_first_idx = start_position
    subseq_last_idx = start_position + input_len
    # target_first_idx = subseq_last_idx + forecast_horizon
    # target_last_idx = target_first_idx + target_len 
    # print("target_last_idx is {}".format(target_last_idx))
    print("stop_position is {}".format(stop_position))
    indices = []
    while subseq_last_idx <= stop_position:
        # indices.append((subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx))
        indices.append((subseq_first_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_last_idx += step_size
        # target_first_idx = subseq_last_idx + forecast_horizon
        # target_last_idx = target_first_idx + target_len

    return indices

    # todo: cannot slice series for different nodes
    # input_len = round(input_len) # just a precaution
    # indices = []
    
    # for i in range(round(num_obs//slice_size)):
    #     sub_indices = []
    #     stop_position = slice_size*(i+1) # for each node
        
    #     # Start the first sub-sequence at index position 0
    #     subseq_first_idx = 0 + i*slice_size
    #     subseq_last_idx = input_len + i*slice_size
    #     # pred_first_idx = subseq_last_idx + forecast_horizon
    #     # pred_last_idx = pred_first_idx + target_len 
    #     # print("target_last_idx is {}".format(target_last_idx))
    #     # print("stop_position is {}".format(stop_position))

    #     while subseq_last_idx <= stop_position:
    #         # indices.append((subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx))
    #         sub_indices.append((subseq_first_idx, subseq_last_idx))
    #         subseq_first_idx += step_size
    #         subseq_last_idx += step_size
    #         # target_first_idx = subseq_last_idx + forecast_horizon
    #         # target_last_idx = target_first_idx + target_len

    #     indices.append(sub_indices)

    # return indices

def get_indices_entire_sequence(window_size: int, step_size: int, slice_size: int) -> list:
    """
    Produce all the start and end index positions that is needed to produce
    the sub-sequences. 

    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
    sequence. These tuples should be used to slice the dataset into sub-
    sequences. These sub-sequences should then be passed into a function
    that slices them into input and target sequences. 
    
    Args:
        num_obs (int): Number of observations (time steps) in the entire 
                        dataset for which indices must be generated, e.g. 
                        len(data)

        window_size (int): The desired length of each sub-sequence. Should be
                            (input_sequence_length + target_sequence_length)
                            E.g. if you want the model to consider the past 100
                            time steps in order to predict the future 50 
                            time steps, window_size = 100+50 = 150

        step_size (int): Size of each step as the data sequence is traversed 
                            by the moving window.
                            If 1, the first sub-sequence will be [0:window_size], 
                            and the next will be [1:window_size].

        slice_size (int): num of series slice for each node.

    Return:
        indices: a list of tuples
    """
    stop_position = slice_size
    
    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    
    subseq_last_idx = window_size
    
    indices = []
    
    while subseq_last_idx <= stop_position:

        indices.append((subseq_first_idx, subseq_last_idx))
        
        subseq_first_idx += step_size
        
        subseq_last_idx += step_size

    return indices

    # todo: cannot slice series for different nodes
    # indices = []

    # for i in range(round(len(data)//slice_size)):
    #     sub_indices = []
    #     stop_position = slice_size + i*slice_size # 1- because of 0 indexing len(data)-1
        
    #     # Start the first sub-sequence at index position 0
    #     subseq_first_idx = 0 + i*slice_size
        
    #     subseq_last_idx = window_size + i*slice_size
        
    #     while subseq_last_idx <= stop_position:

    #         sub_indices.append((subseq_first_idx, subseq_last_idx))
            
    #         subseq_first_idx += step_size
            
    #         subseq_last_idx += step_size

    #     indices.append(sub_indices)

    # return indices


def read_data(data_dir: Union[str, Path] = "data", file_name: str="dfs_merged_upload",
    node_col_name: str="Node", timestamp_col_name: str="timestamp", sort_col: str="Node Index") -> pd.DataFrame:
    """
    Read data from csv file and return pd.Dataframe object

    Args:

        data_dir: str or Path object specifying the path to the directory 
                  containing the data

        target_col_name: str, the name of the column containing the target variable

        timestamp_col_name: str, the name of the column or named index 
                            containing the timestamps
    """

    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Read csv file
    csv_files = list(data_dir.glob(file_name+".csv"))
    
    if len(csv_files) > 1:
        # raise ValueError("data_dir contains more than 1 csv file. Must only contain 1")
        pass
    elif len(csv_files) == 0:
         raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = csv_files[0]

    print("Reading file in {}".format(data_path))

    data = pd.read_csv(
        data_path,
        parse_dates=[timestamp_col_name],
        index_col=[timestamp_col_name],
        infer_datetime_format=True,
        low_memory=False
    )

    ## reset the columns of "node index" and "Time Period index" divided by the len of period, 52 here
    # for i in range(len(data)//52):
    #     data["Node Index"][0+52*i:52*(i+1)] = i
    #     data["Time Period"][0+52*i:52*(i+1)] = np.arange(1,53)
    # data.to_csv(data_path, index=True)

    # Make sure all "n/e" values have been removed from df. 
    if is_ne_in_df(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    # Make sure all "n/e" values have been removed from df. 
    if has_sa_pe(data, node_col_name):
        raise ValueError("data frame contains different time period of nodes.")
    
    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=[sort_col], inplace=True)

    slice_size = data.loc[data[node_col_name] == data[node_col_name].unique()[0]].index.size

    return data, slice_size

def is_ne_in_df(df:pd.DataFrame):
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """
    
    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False

def has_sa_pe(df:pd.DataFrame, node_col_name:str):
    # Get the unique nodes
    nodes = df[node_col_name].unique()

    # Loop over each pair of nodes
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            
            # Select the rows corresponding to the two nodes
            node1_rows = df.loc[df[node_col_name] == nodes[i]]
            node2_rows = df.loc[df[node_col_name] == nodes[j]]
        
            # Check if the two nodes have the same values in column 'Time Period'
            if (node1_rows.index == node2_rows.index).all():
                return False
    return True

def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df

def read_projection_map(data_dir: Union[str, Path] = "data", file_name: str="dfs_merged_upload") -> OrderedDict:
    
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Read csv file
    csv_files = list(data_dir.glob(file_name+".csv"))
    
    if len(csv_files) == 0:
         raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = csv_files[0]

    print("Reading file in {}".format(data_path))

    df = pd.read_csv(data_path)

    # Create an OrderedDict from the DataFrame:
    d = OrderedDict(zip(df.values[:,0], df.values[:,1]))
    return d

def read_edgeIdx(data_dir: Union[str, Path] = "data", file_name: str="dfs_merged_upload") -> list[list[int]]:
    
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Read csv file
    csv_files = list(data_dir.glob(file_name+".csv"))
    
    if len(csv_files) == 0:
         raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = csv_files[0]

    print("Reading file in {}".format(data_path))

    df = pd.read_csv(data_path)

    # # set and write the edge index randomly
    # edge_index = torch.randint(1505, (2, 5000))
    # for i in range(edge_index.size()[1]):
    #     if i<=len(df):
    #         df["startIdx"][i] = edge_index[0][i].numpy()
    #         df["endIdx"][i] = edge_index[1][i].numpy()
    #     else:
    #         # append a new row with index 3
    #         new_row = pd.Series({'startIdx': edge_index[0][i].numpy(), 'endIdx': edge_index[1][i].numpy()})
    #         df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    # df.to_csv(data_path, index=False)
    
    return df


def index_for_feature_projection(d, given_index):
    sum_values = 0
    for index, (key, value) in enumerate(d.items()):
        if index < given_index:
            sum_values += value
    return sum_values


def generate_adj_from_edgeidx(edge_index: Tensor, num_nodes) -> Tensor:
    """
    generate adj given the edge index
    the edge_index tensor to be in the compressed sparse row (CSR) format, which requires the column indices to be sorted.
    Example edge_index tensor edge_index = torch.tensor([[0, 0, 1, 1, 2, 3, 3, 4], [1, 3, 0, 2, 3, 1, 4, 3]])
    """

    # # Compute number of nodes from edge_index tensor
    # num_nodes = edge_index.max().item() + 1

    # Create a sparse COO tensor with ones at the locations specified by the edge_index tensor
    adj_matrix = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), 
                                        (num_nodes, num_nodes))

    # Convert sparse tensor to dense tensor
    adj_matrix = adj_matrix.to_dense()

    # return adjacency matrix
    return adj_matrix

def generate_edgeidx_from_adj(adj: Tensor) -> Tensor:
    # obtain the indices of the non-zero elements
    row, col = torch.where(adj != 0)
    # concatenate row and col tensors to obtain edge index
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def plot(output, truth, step, plot_length, loss):
    plot_length = plot_length
    # Define the colors list with a gradient
    # colors = plt.cm.jet(np.linspace(0, 1, output.size()[1]*3))
    print('-'*12)
    print('validation loss : {:5.5f}'.format(loss))
    print('-'*12)

    for i in range(output.size()[1]):
        plt.plot(output[:plot_length,i], color='red')
        plt.plot(truth[:plot_length,i], color='blue')
        plt.plot(output[:plot_length,i]-truth[:plot_length,i], color='green')
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.savefig('output/transformer-step%d-feat%d.png'%(step,i))
        plt.close()
    return True

# predict the next n steps based on the input data 
def predict_future(src, prediction, nodeIdx):       
    # (batch-size, seq-len , features-num)
    # input : [ m,m+1,...,m+n ] -> [m,m+1,..., m+n+output_window_size]
    obs_length = src.size(1)
    output = torch.cat((src[nodeIdx], prediction[nodeIdx]), dim=0).cpu()

    for i in range(output.size()[1]):
        # I used this plot to visualize if the model pics up any long therm structure within the data.
        plt.plot(output[:,i],color="red")       
        plt.plot(output[:obs_length,i],color="blue")    
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.savefig('output/transformer-future-feat%d.png'%i)
        # plt.show()
        plt.close()
    return True



def expectile_loss(pred, target, expectile_level):
    """
    taking the maximum of two terms:(expectile_level - 1) * abs_errors and expectile_level * errors.
    """
    errors = target - pred
    abs_errors = torch.abs(errors)
    expectile_loss = torch.mean(torch.max((expectile_level - 1) * abs_errors, expectile_level * errors))
    return expectile_loss

# evaluating eval accuracy
def masked_huber_loss(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss_square = 0.5 * loss * loss
    loss = torch.where(loss < 1, loss_square, loss)
    return torch.mean(loss)

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


# evaluating forecasing accuracy
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / ((y_true + y_pred) / 2))) * 100

def evaluate_forecast(y_true, y_pred):
    """
    Mean Absolute Error (MAE): This metric computes the average of the absolute differences between the predicted and actual values.
    It gives an idea of the magnitude of the errors without considering the direction (positive or negative) of the errors.
    Lower values indicate better accuracy.

    Mean Squared Error (MSE): This metric calculates the average of the squared differences between the predicted and actual values.
    By squaring the errors, it emphasizes larger errors more than smaller ones, making it sensitive to outliers.
    Lower values indicate better accuracy.

    Root Mean Squared Error (RMSE): This metric is the square root of the Mean Squared Error (MSE).
    RMSE has the same unit as the original values, which makes it easier to interpret than MSE.
    Lower values indicate better accuracy.

    Mean Absolute Percentage Error (MAPE): This metric computes the average of the absolute percentage differences between the predicted and actual values.
    It provides an error measurement in percentage terms, which can be easier to understand and compare across different scales.
    Lower values indicate better accuracy. However, it has limitations when dealing with zero or near-zero values in the actual data.

    Symmetric Mean Absolute Percentage Error (sMAPE): This metric is a modified version of MAPE, which addresses some of its limitations.
    It calculates the average of the absolute percentage differences between the predicted and actual values, relative to the average of the predicted and actual values.
    This makes it symmetric and more robust when dealing with zero or near-zero values. Lower values indicate better accuracy.
    """
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    # print("Mean Absolute Percentage Error (MAPE):", mape)
    print("Symmetric Mean Absolute Percentage Error (sMAPE):", smape)
