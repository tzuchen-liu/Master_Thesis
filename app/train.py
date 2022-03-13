"""
Author: Tzuchen Liu
Supervisor: Faras Brumand
Master Thesis
The entry point of the training process of the autoencoder model.
Run this file in order to start training.
"""

import random
import datetime as dt
import typing
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from app.machine_learning.ae import AutoencoderIndividualWindow
from app.helper_func.config import TrainConfig
from app.helper_func.utils import Saver, \
    slice_seq_to_windows, calculate_mse_loss


def train(config: TrainConfig):
    """
    This function trains autoencoder models. It does the follow:
    1. Read in previously normalized and sliced data
    2. Shuffle the data before splitting to training and testing dataset
    3. Split to training and testing dataset
    4. Set-up autoencoder model
    5. Set-up optimizer and saver for logging
    6. For each epoch: train model, evaluate model, and log model and data
    """
    if config.print_on_screen:
        print(dt.datetime.now(), 'training starts')

    # Read in previously normalized and sliced data
    data_sliced: typing.List[torch.Tensor] = []
    for seq_id in config.use_seq_id:
        filename = config.data_path / config.filename.replace('*', str(seq_id))
        seq_tensor = torch.tensor(pd.read_csv(
            filename, usecols=config.use_columns).values)
        seq_windows = slice_seq_to_windows(
            seq_tensor, config.window_length, config.window_overlap_length,
            config.irregular_window_at_end)
        data_sliced += seq_windows

    # Shuffle the data before splitting to training and testing dataset
    random.shuffle(data_sliced)

    # Split to training and testing dataset
    training_data, testing_data = train_test_split(
        data_sliced, test_size=config.test_ratio)
    train_windows_count = len(training_data)
    test_windows_count = len(testing_data)
    train_loader = DataLoader(training_data,
                              batch_size=config.batch_size,
                              shuffle=config.shuffle)
    test_loader = DataLoader(testing_data,
                             batch_size=config.batch_size,
                             shuffle=config.shuffle)

    # Set-up autoencoder model
    ae_model = AutoencoderIndividualWindow()
    load_saved_model = config.load_saved_model
    if load_saved_model:
        state_dict = torch.load(config.saved_model_path)
        ae_model.load_state_dict(state_dict)

    # Set-up optimizer and saver for logging
    optimizer = torch.optim.Adam(
        ae_model.parameters(),
        lr=config.learning_rate,
    )
    saver = Saver(config.log_dir)
    saver.backup_files(config.backup_file_list)

    ae_model.train()
    # For each epoch: train model, evaluate model, and log model and data
    for each_epoch in range(1, config.epochs + 1):
        """
        Here I used range(1, config.epochs + 1) instead of range(config.epochs)
        because in the "Log model and data" section I used
        (each_epoch % a_number) to decide whether to log data in this epoch.
        For example, if I train with 100 Epochs and I want to log data every 5
        epochs, using range(config.epochs) leads to logging data in the 0th,
        5th, 10th, ..., 95th epoch (the last epoch is the 99th epoch but it is
        not going to be logged). Using range(1, config.epochs + 1) leads to
        logging data in the 5th, 10th, ..., 100th epoch.
        """

        # The training part
        train_loss = config.starting_loss
        latent_mean_each_window = []
        for windows in train_loader:
            # forward
            reconstructed_windows, latent_mean = \
                ae_model(windows)
            latent_mean_each_window.append(latent_mean)

            reconstruction_loss = calculate_mse_loss(
                windows, reconstructed_windows)
            train_loss += reconstruction_loss.item()

            # backward
            optimizer.zero_grad()
            reconstruction_loss.backward()
            optimizer.step()

        # The evaluation part
        with torch.no_grad():
            ae_model.eval()
            test_loss = config.starting_loss
            for windows in test_loader:
                reconstructed_windows, _ = \
                    ae_model(windows)
                reconstruction_loss = calculate_mse_loss(
                    windows,
                    reconstructed_windows)

                test_loss += reconstruction_loss.item()

        # Log model and data
        save_model_now = each_epoch % config.save_model_interval == 0
        if save_model_now:
            saver.save_model(ae_model, config.model_name, each_epoch,
                             config.print_when_saving_model)
        save_loss_now = \
            each_epoch % config.save_loss_interval == 0
        if save_loss_now:
            saver.save_loss(each_epoch, train_loss / train_windows_count,
                            config.train_loss_filename,
                            config.print_when_saving_loss)
            saver.save_loss(each_epoch, test_loss / test_windows_count,
                            config.test_loss_filename,
                            config.print_when_saving_loss)
        save_latent_values_now = \
            each_epoch % config.save_latent_values_interval == 0
        if save_latent_values_now:
            saver.save_latent_values(each_epoch,
                                     torch.cat(latent_mean_each_window),
                                     config.latent_mean_variable_name,
                                     config.print_when_saving_latent_variables)


if __name__ == '__main__':
    CONFIGURATION = TrainConfig()
    # TrainConfig can be found in app/helper_func/config.py
    train(CONFIGURATION)
