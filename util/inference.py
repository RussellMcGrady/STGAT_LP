"""
Code for running inference with transformer
"""

import torch.nn as nn 
import torch
import util.utils as utils

def run_encoder_decoder_inference(
    model: nn.Module, 
    src: torch.Tensor,
    dec_seq_len: int,
    forecast_step: int,
    device,
    batch_first: bool=False
    ) -> torch.Tensor:

    """
    NB! This function is currently only tested on models that work with 
    batch_first = False
    
    This function is for encoder-decoder type models in which the decoder requires
    an input, tgt, which - during training - is the target sequence. During inference,
    the values of tgt are unknown, and the values therefore have to be generated
    iteratively.  
    
    This function returns a prediction of length forecast_step for each batch in src
    
    NB! If you want the inference to be done without gradient calculation, 
    make sure to call this function inside the context manager torch.no_grad like:
    with torch.no_grad:
        run_encoder_decoder_inference()
        
    The context manager is intentionally not called inside this function to make
    it usable in cases where the function is used to compute loss that must be 
    backpropagated during training and gradient calculation hence is required.
    
    If use_predicted_tgt = True:
    To begin with, tgt is equal to the last value of src. Then, the last element
    in the model's prediction is iteratively concatenated with tgt, such that 
    at each step in the for-loop, tgt's size increases by 1. Finally, tgt will
    have the correct length (target sequence length) and the final prediction
    will be produced and returned.
    
    Args:
        model: An encoder-decoder type model where the decoder requires
               target values as input. Should be set to evaluation mode before 
               passed to this function.
               
        src: The input to the model
        
        forecast_horizon: The desired length of the model's output, e.g. 58 if you
                         want to predict the next 58 hours of FCR prices.
                           
        batch_size: batch size
        
        batch_first: If true, the shape of the model input should be 
                     [batch size, input sequence length, number of features].
                     If false, [input sequence length, batch size, number of features]
    
    """

    # Dimension of a batched model input that contains the target sequence values
    target_seq_dim = 0 if batch_first == False else 1

    # Take the last value of thetarget variable in all batches in src and make it tgt
    # as per the Influenza paper
    tgt = src[-dec_seq_len:, :, :] if batch_first == False else src[:, -dec_seq_len, :] # shape [dec_seq_len, batch_size, feature_size]

    final_prediction = torch.Tensor([]).to(device)
    # Iteratively concatenate tgt with the first element in the prediction
    for _ in range(forecast_step):

        # Create masks
        dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

        dim_b = src.shape[1] if batch_first == True else src.shape[0]

        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_a
            ).to(device)

        src_mask = utils.generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_b
            ).to(device)

        # Make prediction, Just predict all the vals in a whole
        prediction = model(src, tgt, src_mask, tgt_mask)

        # Detach the predicted element from the graph and concatenate with 
        # src, tgt, and final prediction in dimension 1 or 0 based on the step of last predicted value
        final_prediction = torch.cat((final_prediction, prediction.detach()), target_seq_dim)
        tmp_sequence = torch.cat((src, tgt), target_seq_dim)
        tmp_sequence = torch.cat((tmp_sequence, prediction), target_seq_dim)
        src = tmp_sequence[dec_seq_len:-dec_seq_len,:,:] if batch_first == False else tmp_sequence[:,dec_seq_len:-dec_seq_len,:]
        tgt = tmp_sequence[-dec_seq_len:,:,:] if batch_first == False else tmp_sequence[:,-dec_seq_len:,:]
    
    # # Create masks
    # dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

    # dim_b = src.shape[1] if batch_first == True else src.shape[0]

    # tgt_mask = utils.generate_square_subsequent_mask(
    #     dim1=dim_a,
    #     dim2=dim_a
    #     ).to(device)

    # src_mask = utils.generate_square_subsequent_mask(
    #     dim1=dim_a,
    #     dim2=dim_b
    #     ).to(device)

    # # Make final prediction
    # final_prediction = model(src, tgt, src_mask, tgt_mask)

    return final_prediction
