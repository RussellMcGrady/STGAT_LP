import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple


class TransformerDataset(Dataset):
    """
    Dataset class used for transformer models.
    
    """
    def __init__(self, 
        data: torch.tensor,
        indices: list, 
        enc_seq_len: int, 
        dec_seq_len: int, 
        target_seq_len: int,
        slice_size: int
        # scaler: bool,
        ) -> None:

        """
        Args:

            data: tensor, the entire train, validation or test data sequence 
                        before any slicing. If univariate, data.size() will be 
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.

            indices: a list of tuples. Each tuple has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence. 
                     The sub-sequence is split into src, trg and trg_y later.  

            enc_seq_len: int, the desired length of the input sequence given to the
                     the first layer of the transformer model.

            target_seq_len: int, the desired length of the target sequence (the output of the model)

            target_idx: The index position of the target variable in data. Data
                        is a 2D tensor.

        """
        
        super().__init__()

        self.indices = indices

        self.data = data

        # print("From get_src_trg: data size = {}".format(data.size()))

        self.enc_seq_len = enc_seq_len

        self.dec_seq_len = dec_seq_len

        self.target_seq_len = target_seq_len

        self.slice_size = slice_size


    def __len__(self):
        
        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """

        src, trg, trg_y = [], [], []

        node_num = round(len(self.data)//self.slice_size)
        for i in range(node_num):
            # Get the first element of the i'th tuple in the list self.indicesasdfas
            start_idx = self.indices[index][0]

            # Get the second (and last) element of the i'th tuple in the list self.indices
            end_idx = self.indices[index][1]

            sequence = self.data[start_idx+i*self.slice_size:end_idx+i*self.slice_size]

            #print("From __getitem__: sequence length = {}".format(len(sequence)))

            src_i, trg_i, trg_y_i = self.get_src_trg(
                sequence=sequence,
                enc_seq_len=self.enc_seq_len,
                dec_seq_len=self.dec_seq_len,
                target_seq_len=self.target_seq_len
                )
            # stack the tensors along a new dimension
            src.append(src_i.tolist())
            trg.append(trg_i.tolist())
            trg_y.append(trg_y_i.tolist())

        return torch.tensor(src), torch.tensor(trg), torch.tensor(trg_y)
        
        # for i in self.indices[index]: # todo: all node for batch period

        #     # Get the first element of the i'th tuple in the list self.indices for each node
        #     start_idx = i[0] # -index*self.slice_size

        #     # Get the second (and last) element of the i'th tuple in the list self.indices for each node
        #     end_idx = i[1] # -index*self.slice_size

        #     sequence = self.data[start_idx:end_idx] # torch.tensor().float()

        #     #print("From __getitem__: sequence length = {}".format(len(sequence)))

        #     src_i, trg_i, trg_y_i = self.get_src_trg(
        #         sequence=sequence,
        #         enc_seq_len=self.enc_seq_len,
        #         dec_seq_len=self.dec_seq_len,
        #         target_seq_len=self.target_seq_len
        #         )
        #     # stack the tensors along a new dimension
        #     src.append(src_i.tolist())
        #     trg.append(trg_i.tolist())
        #     trg_y.append(trg_y_i.tolist())
        
        # return torch.tensor(src), torch.tensor(trg), torch.tensor(trg_y)
    
    def get_src_trg(
        self,
        sequence: torch.Tensor, 
        enc_seq_len: int, 
        dec_seq_len: int, # same as target_seq_len deprecated
        target_seq_len: int
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence. 

        Args:

            sequence: tensor, a 1D tensor of length n where 
                    n = encoder input length + target sequence length  

            enc_seq_len: int, the desired length of the input to the transformer encoder

            target_seq_len: int, the desired length of the target sequence (the 
                            one against which the model output is compared)

        Return: 

            src: tensor, 1D, used as input to the transformer model

            trg: tensor, 1D, used as input to the transformer model

            trg_y: tensor, 1D, the target sequence against which the model output
                is compared when computing loss. 
        
        """
        assert len(sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"
        
        # encoder input
        src = sequence[:enc_seq_len] 
        
        # decoder input. As per the paper, it must have the same dimension as the 
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)

        # trg = sequence[enc_seq_len-1:len(sequence)-1] # seen the future for embedding
        trg = sequence[enc_seq_len-target_seq_len:enc_seq_len] # unseen the future in forecasting task
        
        assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        trg_y = sequence[-target_seq_len:]

        assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

        return src, trg, trg_y # .squeeze(-1) change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len] if feature size = 1


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return (data * (self.std + 1e-8)) + self.mean