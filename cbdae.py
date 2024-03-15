import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import torch.nn.functional as F
import os
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from enum import Enum
from enum import auto

# Handling the data to make it suitable for the model containing preprocessing
class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for handling time series data.
    This class prepares the data for sequence-based models by creating sequences of a specified length from the input data.

    Attributes:
        df : pandas.DataFrame
            The input data as a DataFrame.
        sequence_length : int
            The length of the sequences to be created from the data.
        timestamp_column : str, optional
            The name of the column in df that contains the timestamps. By default, this is 'Datetime'.
        feature_list : list of str, optional
            A list of the names of the columns in df that contain the features to be used. If this is None, all columns in df are used.

    Methods:
        __len__()
            Returns the number of sequences that can be created from the data.
        __getitem__(idx)
            Returns the sequence and the corresponding timestamps at the given index.
    """

    def __init__(self, df, sequence_length, timestamp_column='Datetime', feature_list=None):
        self.sequence_length = sequence_length
        self.timestamp_column = timestamp_column
        self.df = df.copy(deep=True)
        if feature_list is not None:
            features_timestamp = feature_list + [timestamp_column]
            self.df = self.df[features_timestamp]

        # Convert timestamp column to numerical format
        self.df[self.timestamp_column] = (self.df[self.timestamp_column] - pd.Timestamp("1970-01-01")) // pd.Timedelta(
            '1s')
        self.df_without_timestamp = self.df.drop(columns=[self.timestamp_column])

        # Use unfold to create sequences
        self.sequences_without_timestamp = torch\
            .from_numpy(self.df_without_timestamp.values)\
            .unfold(0, self.sequence_length, 1)\
            .permute(0, 2, 1)
        self.timestamps = torch\
            .from_numpy(self.df[self.timestamp_column].values)\
            .unfold(0, self.sequence_length, 1)\
            .unsqueeze(-1)

    def __len__(self):
        return len(self.sequences_without_timestamp)

    def __getitem__(self, idx):
        return self.sequences_without_timestamp[idx], self.timestamps[idx]


# Just for data loading and batching
class HardNegativeSelector(DataLoader):
    """
    A PyTorch DataLoader for handling hard negative samples in a dataset. This class is used to create batches of data,
    where each batch contains a specified ratio of negative samples.

    Attributes:
        dataset : torch.utils.data.Dataset
            The input dataset.
        batch_size : int
            The size of the batches to be created from the data.
        negative_ratio : float, optional
            The ratio of negative samples in each batch. By default, this is 0.5.
        mode : str, optional
            The mode in which the DataLoader is operating. This should be either 'train' or 'val'. By default, this is 'train'.
        r : int
            The number of negative samples in each batch.
        s : int
            The number of positive samples in each batch.

    Methods:
        collate_fn_train(batch)
            Returns a batch of data for training. The batch contains both positive and negative samples.
        collate_fn_val(batch)
            Returns a batch of data for validation. The batch contains only positive samples.
    """

    def __init__(self, dataset, batch_size, negative_ratio=0.5, mode='train', *args, **kwargs):
        self.r = int(batch_size * negative_ratio)
        self.s = batch_size - self.r
        self.negative_ratio = negative_ratio
        self.dataset = dataset
        self.mode = mode
        super().__init__(self.dataset,
                         batch_size=self.s if mode == 'train' else batch_size,
                         collate_fn=self.collate_fn_train if mode == 'train' else self.collate_fn_val,
                         *args,
                         **kwargs)

    def collate_fn_train(self, batch):
        Bs = torch.stack([b[0] for b in batch])
        Bs_time = torch.stack([b[1] for b in batch])

        indices = random.sample(range(len(self.dataset)), self.r)
        BR = [self.dataset[i] for i in indices]
        Br = torch.stack([b[0] for b in BR])
        Br_time = torch.stack([b[1] for b in BR])

        return Bs, Br, Bs_time, Br_time

    def collate_fn_val(self, batch):
        B = torch.stack([b[0] for b in batch])
        B_time = torch.stack([b[1] for b in batch])

        return B, B_time


class CBDAE(nn.Module):
    """
    A PyTorch Module for a Contrastive Blind Denoising Auto Encoder (CBDAE).

    Attributes:
        N : int
            The number of features in the input data.
        hidden_size : int
            The number of features in the hidden state.
        num_layers : int
            The number of layers in the GRU cells.
        G1 : int
            The number of features in the output of the first intermediate linear transformation.
        G : int
            The number of features in the output of the second intermediate linear transformation.

    Methods:
        forward(x)
            Performs a forward pass of the CBDAE on the input data x.
    """

    def __init__(self, N, hidden_size, num_layers, G1, G):
        super(CBDAE, self).__init__()

        # Encoder GRU parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder GRU cells for all layers
        self.gru_cells_enc = nn.ModuleList(
            [nn.GRUCell(N if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

        # Intermediate Linear transformation for encoder output
        self.Wg1 = nn.Linear(hidden_size, G1, bias=False)
        self.Wg2 = nn.Linear(G1, G, bias=False)

        # Decoder GRU cells for all layers
        self.gru_cells_dec = nn.ModuleList(
            [nn.GRUCell(N if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

        # Linear transformation o_dec
        self.W_out = nn.Linear(hidden_size, N)

        # Sampling parameter
        self.p_d = 1

    def forward(self, x):
        # General setup
        B, T, N = x.size()
        device = x.device
        # Convert to float32 before sending to device
        x = x.float().to(device)

        # Commonly used zero tensor
        # h1(0) is zeros: https://arxiv.org/pdf/1406.1078.pdf
        zeros = torch.zeros(B, self.hidden_size, dtype=x.dtype).to(device)

        # Encoder
        # Empty list(not tensor) for initiation(to avoid inplace operation)
        h_enc_layers = []
        for layer in range(self.num_layers):
            h_enc_time_steps = []
            for j in range(T):
                p = x[:, j, :] if layer == 0 else h_enc_layers[layer - 1][:, j, :]
                h_enc_time_step = self.gru_cells_enc[layer](p, zeros if j == 0 else h_enc_time_steps[-1])
                h_enc_time_steps.append(h_enc_time_step)

            # Stack time steps
            h_enc_layer = torch.stack(h_enc_time_steps, dim=1)
            h_enc_layers.append(h_enc_layer)

        # Stack all layers
        h_enc = torch.stack(h_enc_layers, dim=0)

        # Intermediate Linear transformations
        zT = self.Wg2(torch.relu(self.Wg1(h_enc[-1, :, -1, :])))  # shape: (B, G)

        # Decoder
        # Initialize d_prev with h(T) from encoder
        d_prev = h_enc[:, :, -1, :].clone()  # shape: (num_layers, B, hidden_size)

        # Create a list to store the intermediate outputs
        y_hat_list = []
        d = [None for _ in range(self.num_layers)]
        for j in range(T):  # Iterating through time steps

            # Select the input for the first layer of the decoder
            if j > 0:
                if random.random() - (not self.training) <= self.p_d:
                    # Scheduled sampling: Use the output from the previous time step
                    y_input = y_hat_list[-1].squeeze(dim=1)
                else:
                    # Use the actual input from the previous time step
                    y_input = x[:, j - 1, :]

            else:
                # y(0) for first layer's input is zeros: https://arxiv.org/pdf/1406.1078.pdf
                y_input = torch.zeros(B, N).to(device)

            # Loop through each layer in the decoder
            for layer in range(self.num_layers):
                # Calculate the new hidden state using the GRU cell
                d[layer] = self.gru_cells_dec[layer](y_input,
                                                     d_prev[layer] if j == 0 else d[layer])  # Update the hidden state

                # The output of the current layer will be the input to the next layer
                y_input = d[layer]

            # Apply the linear transformation to the output of the last layer
            y_hat_t = self.W_out(d[-1]).unsqueeze(1)  # Add time dimension
            y_hat_list.append(y_hat_t)

        # Concatenate all the outputs in y_hat_list to create y_hat
        y_hat = torch.cat(y_hat_list, dim=1)

        # zT shape: (B, G)
        # y_hat shape: (B, T, N)
        return zT, y_hat


def cbdae_loss_cosine(B_in, B_out, Zs_out, Zr_out, beta):
    """
    Computes the loss for a CBDAE.

    This function calculates the loss as the sum of the Autoencoder (AE) loss and the Noise Contrastive Estimation (NCE) loss.
    The AE loss is computed as the mean squared error between the input and output of the autoencoder.
    The NCE loss is computed using cosine similarity between the output of the encoder and a negative sample.

    Args:
        B_in : torch.Tensor
            The input to the autoencoder. Shape: (B, T, N), where B is the batch size, T is the sequence length, and N is the number of features.
        B_out : torch.Tensor
            The output of the autoencoder. Shape: (B, T, N), same as B_in.
        Zs_out : torch.Tensor
            The output of the encoder for the positive samples. Shape: (s_length, G), where s_length is the length of the first dimension and G is the number of features in the output.
        Zr_out : torch.Tensor
            The output of the encoder for the negative samples. Shape: (r_length, G), where r_length is the length of the first dimension and G is the same as in Zs_out.
        beta : float
            The weight for the NCE loss in the overall loss calculation.

    Returns:
        torch.Tensor
            The computed loss. This is a scalar tensor.
    """
    device = B_in.device  # Get the device of the input tensor

    s_length = Zs_out.shape[0]  # the length of the first dimension of Zs_out
    r_length = Zr_out.shape[0]  # the length of the first dimension of Zr_out

    def l(i, j, s):
        L = i.shape[0]  # the length of the input tensors i, j, s

        # Zi, Zj are now tensors of shape (L, G)
        Zi = Zs_out[i]  # shape: (L, G)
        Zj = Zs_out[j]  # shape: (L, G)

        # numerator is now a tensor of shape (L,)
        numerator = torch.exp(F.cosine_similarity(Zi, Zj, dim=1))  # shape: (L,)

        # Create masks to exclude the index 's' from the calculations
        mask_Zs = torch.ones((L, s_length), dtype=torch.bool).to(device)  # shape: (L, s)
        mask_Zs[torch.arange(L), i] = 0
        valid_s = (s >= 0) & (s < s_length)
        s = s.to(device)
        mask_Zs[torch.arange(L).to(device)[valid_s.to(device)], s[valid_s].to(device)] = 0
        mask_Zr = torch.ones((L, r_length), dtype=torch.bool).to(device)  # shape: (L, r)

        # Compute the negative similarities for Zs_out and Zr_out
        negative_similarities_Zs = torch.exp(
            F.cosine_similarity(Zi.unsqueeze(1), Zs_out.unsqueeze(0), dim=-1))  # shape: (L, s)
        negative_similarities_Zr = torch.exp(
            F.cosine_similarity(Zi.unsqueeze(1), Zr_out.unsqueeze(0), dim=-1))  # shape: (L, r)

        # Apply masks
        negative_similarities_Zs = negative_similarities_Zs * mask_Zs
        negative_similarities_Zr = negative_similarities_Zr * mask_Zr

        # Compute the denominator by summing the negative similarities
        denominator = torch.sum(negative_similarities_Zs, dim=1) + torch.sum(negative_similarities_Zr,
                                                                             dim=1)  # shape: (L,)

        result = -torch.log(numerator / denominator)  # shape: (L,)
        return result

    # Create tensors for indices for the first part of NCE_loss
    i1 = torch.arange(0, s_length - 1).to(device)  # k
    j1 = torch.arange(1, s_length).to(device)  # k+1
    s1 = torch.arange(-1, s_length - 2).to(device)  # k-1

    # Create tensors for indices for the second part of NCE_loss
    i2 = torch.arange(1, s_length).to(device)  # k+1
    j2 = torch.arange(0, s_length - 1).to(device)  # k
    s2 = torch.arange(2, s_length + 1).to(device)  # k+2

    # Calculate NCE loss
    NCE_loss = (torch.sum(l(i1, j1, s1)) + torch.sum(l(i2, j2, s2))) / (2 * (s_length - 1))

    AE_loss = F.mse_loss(B_in, B_out)
    overall_loss = AE_loss + beta * NCE_loss
    return overall_loss

def cbdae_loss_euclidean(B_in, B_out, Zs_out, Zr_out, beta):
    """
    Computes the loss for a CBDAE.

    This function calculates the loss as the sum of the Autoencoder (AE) loss and the Noise Contrastive Estimation (NCE) loss.
    The AE loss is computed as the mean squared error between the input and output of the autoencoder.
    The NCE loss is computed using cosine similarity between the output of the encoder and a negative sample.

    Args:
        B_in : torch.Tensor
            The input to the autoencoder. Shape: (B, T, N), where B is the batch size, T is the sequence length, and N is the number of features.
        B_out : torch.Tensor
            The output of the autoencoder. Shape: (B, T, N), same as B_in.
        Zs_out : torch.Tensor
            The output of the encoder for the positive samples. Shape: (s_length, G), where s_length is the length of the first dimension and G is the number of features in the output.
        Zr_out : torch.Tensor
            The output of the encoder for the negative samples. Shape: (r_length, G), where r_length is the length of the first dimension and G is the same as in Zs_out.
        beta : float
            The weight for the NCE loss in the overall loss calculation.

    Returns:
        torch.Tensor
            The computed loss. This is a scalar tensor.
    """
    device = B_in.device  # Get the device of the input tensor

    s_length = Zs_out.shape[0]  # the length of the first dimension of Zs_out
    r_length = Zr_out.shape[0]  # the length of the first dimension of Zr_out

    def l(i, j, s):
        L = i.shape[0]  # the length of the input tensors i, j, s

        # Zi, Zj are now tensors of shape (L, G)
        Zi = Zs_out[i]  # shape: (L, G)
        Zj = Zs_out[j]  # shape: (L, G)

        # numerator is now a tensor of shape (L,)
        numerator = torch.exp(-F.pairwise_distance(Zi, Zj, p=2, eps=0))  # shape: (L,)

        # Create masks to exclude the index 's' from the calculations
        mask_Zs = torch.ones((L, s_length), dtype=torch.bool).to(device)  # shape: (L, s)
        mask_Zs[torch.arange(L), i] = 0
        valid_s = (s >= 0) & (s < s_length)
        s = s.to(device)
        mask_Zs[torch.arange(L).to(device)[valid_s.to(device)], s[valid_s].to(device)] = 0
        mask_Zr = torch.ones((L, r_length), dtype=torch.bool).to(device)  # shape: (L, r)

        # Compute the negative similarities for Zs_out and Zr_out 
        ### !TODO: Replace cosine_similarity to euclidean distance 
        negative_similarities_Zs = torch.exp(
            -torch.linalg.vector_norm(Zi.unsqueeze(1) - Zs_out.unsqueeze(0), dim=-1))  # shape: (L, s)
        negative_similarities_Zr = torch.exp(
            -torch.linalg.vector_norm(Zi.unsqueeze(1) - Zr_out.unsqueeze(0), dim=-1))  # shape: (L, r)

        # Apply masks
        negative_similarities_Zs = negative_similarities_Zs * mask_Zs
        negative_similarities_Zr = negative_similarities_Zr * mask_Zr

        # Compute the denominator by summing the negative similarities
        denominator = torch.sum(negative_similarities_Zs, dim=1) + torch.sum(negative_similarities_Zr,
                                                                             dim=1)  # shape: (L,)

        result = -torch.log(numerator / denominator)  # shape: (L,)
        return result

    # Create tensors for indices for the first part of NCE_loss
    i1 = torch.arange(0, s_length - 1).to(device)  # k
    j1 = torch.arange(1, s_length).to(device)  # k+1
    s1 = torch.arange(-1, s_length - 2).to(device)  # k-1

    # Create tensors for indices for the second part of NCE_loss
    i2 = torch.arange(1, s_length).to(device)  # k+1
    j2 = torch.arange(0, s_length - 1).to(device)  # k
    s2 = torch.arange(2, s_length + 1).to(device)  # k+2

    # Calculate NCE loss
    NCE_loss = (torch.sum(l(i1, j1, s1)) + torch.sum(l(i2, j2, s2))) / (2 * (s_length - 1))

    AE_loss = F.mse_loss(B_in, B_out)
    overall_loss = AE_loss + beta * NCE_loss
    return overall_loss

class Similarity(Enum):
    EUCLIDEAN = cbdae_loss_cosine
    COSINE = cbdae_loss_euclidean

def train_and_save(dataframe, features, savefile_path_name='temp', val_ratio=0.3, timestamp_column='Datetime',
                   sequence_length=60, batch_size=64, negative_ratio=0.2, hidden_size=200, num_layers=2, G1=50, G=50,
                   lr=0.01, patience=7, factor=0.5, num_epochs=700, k_d=0.5, c_d=0.015, loss_beta=1.5, nce_loss=Similarity.EUCLIDEAN):
    """
    Trains a CBDAE model on the provided training data.

    Args:
        dataframe (pandas.DataFrame): The training data.
        features (list): The list of features to use for training.
        savefile_path_name (str): Filename to save model.
        val_ratio (float): The ratio of data to use for validation. Defaults to 0.3.
        timestamp_column (str, optional): The name of the timestamp column in the data. Defaults to 'Datetime'.
        sequence_length (int, optional): The length of the sequences for the TimeSeriesDataset. Defaults to 60.
        batch_size (int, optional): The batch size for training. Defaults to 64.
        negative_ratio (float, optional): The ratio of negative samples for the HardNegativeSelector. Defaults to 0.2.
        hidden_size (int, optional): The number of hidden units in the GRU cells. Defaults to 200.
        num_layers (int, optional): The number of layers in the GRU. Defaults to 2.
        G1 (int, optional): The size of the first linear transformation in the CBDAE model. Defaults to 50.
        G (int, optional): The size of the second linear transformation in the CBDAE model. Defaults to 50.
        lr (float, optional): The initial learning rate for the Adam optimizer. Defaults to 0.01.
        patience (int, optional): The patience for the learning rate scheduler. Defaults to 7.
        factor (float, optional): The factor by which the learning rate is reduced when the scheduler is triggered. Defaults to 0.5.
        num_epochs (int, optional): The number of epochs to train for. Defaults to 700.
        k_d (float, optional): The initial value for the scheduled sampling parameter. Defaults to 0.5.
        c_d (float, optional): The increment for the scheduled sampling parameter. Defaults to 0.015.
        loss_beta (float, optional): The beta parameter for the loss function. Defaults to 1.5.
        nce_loss (Enum, optional): The type of similarity function used is NCE loss.

    Returns:
        model (CBDAE): The trained CBDAE model.
    """
    loss_function = nce_loss
    N = len(features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_beta = torch.tensor(loss_beta)
    model = CBDAE(N, hidden_size, num_layers, G1, G).to(device)

    architecture = {
        "N": N,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "G1": G1,
        "G": G,
        "features": features
    }

    with open(f"{savefile_path_name}.json", "w") as f:
        json.dump(architecture, f)

    train_ratio = 1 - val_ratio
    train_idx = int(len(dataframe) * train_ratio)
    train_data = dataframe.iloc[:train_idx]
    validation_data = dataframe.iloc[train_idx:]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=factor, verbose=True)

    dataset = TimeSeriesDataset(train_data, sequence_length, timestamp_column=timestamp_column, feature_list=features)
    data_loader = HardNegativeSelector(dataset, batch_size, negative_ratio=negative_ratio)

    validation_dataset = TimeSeriesDataset(validation_data, sequence_length, timestamp_column=timestamp_column,
                                           feature_list=features)
    validation_loader = HardNegativeSelector(validation_dataset, batch_size, mode='val')

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        model.p_d = min(1, k_d + c_d * epoch)
        train_losses = []
        for batch in data_loader:
            optimizer.zero_grad()

            Bs, Br, Bs_time, Br_time = batch
            Bs = Bs.to(device).to(torch.float32)
            Br = Br.to(device).to(torch.float32)
            B = torch.cat((Bs, Br), dim=0)
            Z, B_hat = model(B)
            Z = Z.to(device)
            B_hat = B_hat.to(device)

            s_size = Bs.size(dim=0)
            Zs = Z[:s_size, :]
            Zr = Z[s_size:, :]
            loss = loss_function(B, B_hat, Zs, Zr, loss_beta)
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                B_val, B_time_val = batch
                B_val = B_val.to(device).to(torch.float32)
                Z_val, B_hat_val = model(B_val)
                Z_empty = torch.empty((0, Z_val.shape[1]), dtype=torch.float32).to(device)

                loss_val = loss_function(B_val, B_hat_val, Z_val, Z_empty, loss_beta)
                val_losses.append(loss_val.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), f"{savefile_path_name}.pth")
            best_val_loss = avg_val_loss

    return model


def load(savefile_path_name='temp'):
    """
    Load a CBDAE model's architecture and weights from files.

    The function reads a JSON file to retrieve the model's architecture (hyperparameters) and then uses these parameters to instantiate the model.
    It then loads the model's weights from a .pth file. The model is switched to evaluation mode before it is returned.

    Args:
        savefile_path_name (str): The base name of the files to load from. The function expects to find a .json file and a .pth file with this base name in the current directory.
                             The .json file should contain a dictionary with the keys "N", "hidden_size", "num_layers", "G1", and "G",
                             and the .pth file should contain the model's state dict. Defaults to 'temp'.

    Returns:
        model (CBDAE): The loaded CBDAE model, ready for evaluation.
        features (list):
    """
    with open(f"{savefile_path_name}.json", "r") as f:
        architecture = json.load(f)
    features = architecture["features"]

    # Instantiate the model with the correct architecture
    model = CBDAE(architecture["N"], architecture["hidden_size"], architecture["num_layers"], architecture["G1"],
                  architecture["G"])

    # Load the model's weights
    model.load_state_dict(torch.load(f"{savefile_path_name}.pth", map_location=torch.device('cpu')))
    model.eval()

    return model, features

def run_model(model, input_data):
    """
    Run a model on the given data

    Args:
        model (CBDAE): The model to run. This should be trained CBDAE model
        input_data (torch.tensor): Shape of (B-T+1, T, N). The data to run the model on.

    Returns:
        output (torch.tensor): The output of the model for input data
    """
    model.eval()
    with torch.no_grad():
        zT, y_hat = model(input_data)
    return zT, y_hat[:, -1, :]