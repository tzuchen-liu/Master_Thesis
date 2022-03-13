"""
This file provides helpful functions and class that are directly used by other
scripts in this project, or generate files that are used by other scripts in
this project.
These methods are collected here because they are used by multiple scripts,
or they have the potential to be used outside this project.
This file includes:
function:
    seq_for_recon_plot: generate a csv file that stores sequence id with max,
     min, median reconstruction loss under each autoencoder model
    calculate_mse_loss: calculate mean square error of original data and
     reconstructed data
    slice_by_pattern: take a raw data file as input, and slice it into data
     sequences by the pattern of a target column
    slice_seq_to_windows: slice a data sequence into multiple windows
    windows_to_seq: combine windows back to a data sequence
    dataloader_different_seq_len: the dataloader of pytorch does not accept
     data with different sequence lenght. This function can be used in that
     case.
    add_velocity_to_file: add velocity column to a file that has a column of
     position data, the velocity is calculated by the derivative of the
     position data.
class:
    Saver: this class helps the user to log important data and files e.g.
     training/testing loss, trained models, latent values, config file and
     training scripts
"""
from typing import List
import os
import datetime
import random
from pathlib import Path
import numpy as np
import pandas as pd
from app.helper_func.config import VisualizationConfig, AEConfig
from app.machine_learning.ae import AutoencoderIndividualWindow
import torch


def seq_for_recon_plot(model_dir: str, output_dir: str, model_list: List[str],
                       input_filename: str, data_dir: str,
                       seq_id_list: List[int]):
    """
    This function feed every data sequence to each trained autoencoder model,
     and calculate the reconstruction loss of each data sequence when
     reconstructed by each autoencoder model. For each autoencoder model, the
     sequence id of the sequences that result in the maximum reconstruction
     loss, minimum reconstruction loss, and median reconstruction loss are
     recorded.
    The output csv file contains 4 columns:
     Model: the model name
     min_loss_seq_id: the sequence id of the data sequence that has the
      minimum loss when reconstructed by this model
    median_loss_seq_id: the sequence id of the data sequence that has the
      median loss of all data sequences when reconstructed by this model
    max_loss_seq_id: the sequence id of the data sequence that has the
      maximum loss when reconstructed by this model

    :param model_dir: the directory that the autoencoder models were stored
    :param model_list: a python list of target model names
    :param output_dir: the directory that the output plots will be stored
    :param input_filename: the filename of a data sequence. This filename
    includes symbol '*', and this symbol will be replaced by sequence ID.
    For example, input_filename = 'dataset_*.csv'. When this function plots
    the reconstruction figures of sequence ID 100, this function will read
    the data by reading file 'dataset_100.csv'.
    :param data_dir: the directory that the input_filename locates
    :param seq_id_list: the list of ids of data sequences
    """
    visual_config = VisualizationConfig()
    ae_config = AEConfig()
    with open(output_dir + 'seq_for_recon_plot.csv', 'a') as output_file:
        output_file.write(
            'Model,min_loss_seq_id,median_loss_seq_id,max_loss_seq_id\n')
        output_file.flush()
        for model_name in model_list:
            print(model_name)
            model_name_split = model_name.split('_')
            ae_config.model_type = model_name_split[0].lower()
            ae_config.latent_dim = int(model_name_split[2][1:])
            if model_name_split[1] == 'F2':
                ae_config.n_features = 2
                usecols = ['x', 'valve_status']
            else:
                ae_config.n_features = 3
                usecols = ['x', 'valve_status', 'velocity']

            ae_config.update()
            ae_model = AutoencoderIndividualWindow(ae_config=ae_config)
            state_dict = torch.load(model_dir + 'AE_' + model_name + '.pth')
            ae_model.load_state_dict(state_dict)

            recon_loss_all_seq = []
            for seq_id in seq_id_list:
                data = torch.tensor(
                    pd.read_csv(data_dir + input_filename.replace(
                        '*', str(seq_id)),
                                usecols=usecols).values)

                original_windows = torch.stack(slice_seq_to_windows(
                    data, visual_config.window_length,
                    visual_config.window_overlap_length,
                    visual_config.irregular_window_at_end))

                with torch.no_grad():
                    ae_model.eval()
                    reconstructed_windows, _ = ae_model(original_windows)
                    reconstruction_loss = calculate_mse_loss(
                        original_windows, reconstructed_windows)
                    recon_loss_all_seq.append(reconstruction_loss.item())

            min_loss_seq_id = seq_id_list[np.argmin(recon_loss_all_seq)]
            median_loss_seq_id = seq_id_list[
                np.argsort(recon_loss_all_seq)[len(recon_loss_all_seq) // 2]]
            max_loss_seq_id = seq_id_list[np.argmax(recon_loss_all_seq)]
            output_file.write('%s,%s,%s,%s\n' % (
                model_name, min_loss_seq_id, median_loss_seq_id,
                max_loss_seq_id))
            output_file.flush()


def calculate_mse_loss(windows: torch.Tensor,
                       reconstructed_windows: torch.Tensor
                       ) -> torch.nn.MSELoss:
    # sourcery skip: inline-immediately-returned-variable
    """ Compute Mean Square Error between data and reconstructed_data.
    In this project, this function is used to calculate the loss of
     autoencoder models."""
    reconstructed_window_remove_batch_size = torch.squeeze(
        reconstructed_windows)
    mse = torch.nn.MSELoss()
    reconstruction_loss = mse(
        windows.float(), reconstructed_window_remove_batch_size.float()
    )
    return reconstruction_loss


class Saver:
    """
    To save models and to load the models after saving.
    It can also save training values regarding the model.
    """

    def __init__(self, log_dir):
        self.start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir_this_training = log_dir / self.start_time
        self.log_dir_this_training.mkdir(parents=True)

    def save_model(self, model, model_name: str, epoch: int,
                   print_on_screen: bool) -> None:
        """ This method is used to save the torch model. """
        now = datetime.datetime.now()
        model_name_with_time = '{}_{}'.format(model_name, self.start_time)
        file_name = "{}_epoch-{}.pth".format(model_name_with_time, epoch)
        torch.save(model.state_dict(), self.log_dir_this_training / file_name)
        if print_on_screen:
            print("Now is {}. Saved {} model in epoch {}".format(
                now, model_name, epoch))

    def load_model(self, model, model_name: str, epoch: int,
                   print_on_screen: bool) -> None:
        """ This method is used to load the torch model. """
        file_name = "{}_epoch-{}.pth".format(model_name, epoch)
        model_saved = torch.load(self.log_dir_this_training / file_name)
        model.load_state_dict(model_saved)
        if print_on_screen:
            print("Loaded {}".format(model_name))

    def save_loss(self, epoch: int, loss: float,
                  filename: str, print_on_screen: bool) -> None:
        """ This method is used to save training loss and testing loss
        of the torch model. """

        file_path = self.log_dir_this_training / filename
        if not file_path.exists():
            with open(file_path, 'w') as file:
                file.write('epoch,loss\n')
        with open(file_path, 'a') as file:
            file.write('%d,%s\n' % (epoch, loss))
        now = datetime.datetime.now()
        if print_on_screen:
            print("Now is {}. Saved loss in epoch {} in file {}".format(
                now, epoch, filename))

    def save_latent_values(self, epoch: int, values: torch.Tensor,
                           variable_name: str, print_on_screen: bool) -> None:
        """ This method is used to save values (e.g. latent variables
        of the vae model). """
        filename = variable_name + '_%d.pt' % epoch
        torch.save(values, self.log_dir_this_training / filename)
        now = datetime.datetime.now()
        if print_on_screen:
            print("Now is {}. Saved values in epoch {} in file {}".format(
                now, epoch, filename))

    def load_training_values(self, filename: Path) -> pd.DataFrame:
        # sourcery skip: inline-immediately-returned-variable
        """ This method is used to load saved training or testing loss. """
        training_values = pd.read_csv(self.log_dir_this_training,
                                      filename, delimiter=',')
        return training_values

    def backup_files(self, filenames_list: List[str]):
        """
        This method is used to back up some files to the log directory as well.
        """
        for filename in filenames_list:
            os.system('cp %s %s/' % (
                str(filename), self.log_dir_this_training))


def slice_by_pattern(original_data: pd.DataFrame, check_pattern_column,
                     pattern_end: int, pattern_start: int,
                     remove_pattern_column: bool, seq_len: int = None,
                     output_filename: str = None,
                     all_seq_len_output_filename=None):
    """ Cut out every data sequence from the DataFrame by the given patterns.
    This method examines the check_pattern_column row by row. If the value in
    this column changes from pattern_end to pattern_start, meaning that
    a data sequence terminated and a new data sequence starts.
    remove_pattern_column: If this is True, the returned data will not
        contain the check_pattern_column.
    seq_len: If this is not None, sequences with length shorter than this
    number would be padded with zero after the original data sequence, forcing
    every data sequence to be in the same length as the seq_len specified.
    output_filename: the filename should contain "*", which will be replaced
    with numbers. For example, output_filename = /example/example_*.csv,
    the output sequence files would be /example/example_1.csv,
    /example/example_2.csv, /example/example_3.csv, ... etc."""
    # "all_seq_len" is only for calculating statistics of sequence lengths
    all_seq_len = []
    return_data = []
    sequence_start = 0
    sequence_count = 0
    previous_check_value = original_data.loc[0, check_pattern_column]
    for row_count in range(1, original_data.shape[0]):
        current_check_value = original_data.loc[
            row_count, check_pattern_column]
        if previous_check_value == pattern_end and \
                current_check_value == pattern_start:
            selected_dataframe = original_data[sequence_start:row_count]
            if remove_pattern_column:
                selected_dataframe = selected_dataframe.drop(
                    columns=[check_pattern_column])
            all_seq_len.append(row_count - sequence_start)
            if output_filename:
                output_path = output_filename.replace('*', str(sequence_count))
                sequence_count += 1
                selected_dataframe.to_csv(output_path, index=False)
            else:
                selected_tensor = torch.tensor(selected_dataframe.values)
                if seq_len is None:
                    selected_data = selected_tensor
                else:
                    selected_tensor_shape = selected_tensor.shape
                    zero_tensor = torch.zeros(
                        seq_len - selected_tensor_shape[0],
                        selected_tensor_shape[1])
                    selected_data = torch.cat((selected_tensor, zero_tensor))
                return_data.append(selected_data)
            sequence_start = row_count
        previous_check_value = current_check_value
    if all_seq_len_output_filename:
        df_seq_len = pd.DataFrame(all_seq_len, columns=['length'])
        df_seq_len.to_csv('all_seq_len.csv', index=False)
    print('There are %d sequences. The length of the shortest sequence is %d, '
          'longest sequence is %d, and the average length is about %d.' % (
              len(all_seq_len), min(all_seq_len),
              max(all_seq_len), int(np.mean(all_seq_len))))

    return return_data


def slice_seq_to_windows(input_data: torch.Tensor, window_length: int,
                         window_overlap_length: int,
                         irregular_window_at_end: bool) -> list:
    """
    Slice a data sequence into windows of fixed size.
    The returned data is a list of windows (torch.tensor)
    input_data: one data sequence in shape [seq_len, n_features]
    for how to set window_length, window_overlap_length, and
    irregular_window_at_end, please refer to
     assets/presentations/Thesis_Figures.pptx page 9
    """
    seq_len = input_data.shape[0]
    effective_window_length = window_length - window_overlap_length
    remainder = seq_len % effective_window_length
    windows_count = seq_len // effective_window_length \
        + int(remainder > window_overlap_length) \
        - window_overlap_length // effective_window_length
    if irregular_window_at_end:
        return_data = [input_data[
            effective_window_length * window_index:
            effective_window_length * window_index + window_length]
                       for window_index in range(windows_count - 1)]
        return_data.append(input_data[-window_length:])
    else:
        first_window = [input_data[:window_length]]
        shift_factor = seq_len - \
            (effective_window_length * (windows_count - 1)
             + window_overlap_length)
        other_windows = [input_data[
            shift_factor + effective_window_length * window_index:
            shift_factor + effective_window_length * window_index
            + window_length]
                         for window_index in range(windows_count - 1)]
        return_data = first_window + other_windows
    return return_data


def windows_to_seq(input_data: list,
                   seq_len: int,
                   window_overlap_length: int,
                   irregular_window_at_end: bool,
                   overlap_ignore_later_window: bool) -> torch.Tensor:
    """
    This function is used when plotting reconstruction data.
    The autoencoder model works on window's basis. The input are windows of
     original data, the output are windows of reconstructed data. In order to
     compare with the original data sequence, the output windows should first
     be put back together as a data sequence.
    This function only handles one data sequence.
    :param input_data: windows of reconstructed data
    :param seq_len: the length that the sequence should be
    :param window_overlap_length: this should be set the same as the value
     used when slicing the original data into windows
    :param irregular_window_at_end: this should be set the same as the value
     used when slicing the original data into windows
    :param overlap_ignore_later_window: this should be set the same as the
     value used when slicing the original data into windows
    :return: the reconstructed data sequence
    """

    input_data = [torch.squeeze(window) for window in input_data]
    window_length = len(input_data[0])
    if irregular_window_at_end and overlap_ignore_later_window:
        return_data = input_data[0]
        for window in input_data[1:-1]:
            return_data = torch.cat((
                return_data, window[window_overlap_length:]))
        remaining_length = seq_len - len(return_data)
        return_data = torch.cat((
            return_data, input_data[-1][-remaining_length:]))

    elif irregular_window_at_end:
        return_data = input_data[0][:-window_overlap_length]
        for window in input_data[1:-2]:
            return_data = torch.cat((
                return_data, window[:-window_overlap_length]))
        length_second_last_window = seq_len - len(return_data) - window_length
        return_data = torch.cat((
            return_data, input_data[-2][:length_second_last_window]))
        return_data = torch.cat((return_data, input_data[-1]))

    elif overlap_ignore_later_window:
        all_windows = torch.cat(input_data)
        all_windows_reversed = torch.flip(all_windows, [0])
        reversed_input_data = torch.split(all_windows_reversed, window_length)
        reversed_return_data = reversed_input_data[0][:-window_overlap_length]
        for window in reversed_input_data[1:-2]:
            reversed_return_data = torch.cat((
                reversed_return_data, window[:-window_overlap_length]))
        length_second_last_window = \
            seq_len - len(reversed_return_data) - window_length
        reversed_return_data = torch.cat((
            reversed_return_data,
            reversed_input_data[-2][:length_second_last_window]))
        reversed_return_data = torch.cat((
            reversed_return_data, reversed_input_data[-1]))
        return_data = torch.flip(reversed_return_data, [0])

    else:  # not irregular_window_at_end and not overlap_ignore_later_window
        all_windows = torch.cat(input_data)
        all_windows_reversed = torch.flip(all_windows, [0])
        reversed_input_data = torch.split(all_windows_reversed, window_length)
        reversed_return_data = reversed_input_data[0]
        for window in reversed_input_data[1:-1]:
            reversed_return_data = torch.cat((
                reversed_return_data, window[window_overlap_length:]))
        remaining_length = seq_len - len(reversed_return_data)
        reversed_return_data = torch.cat((
            reversed_return_data, reversed_input_data[-1][-remaining_length:]))
        return_data = torch.flip(reversed_return_data, [0])
    return return_data


def dataloader_different_seq_len(input_data: list, batch_size: int,
                                 shuffle: bool,
                                 allow_smaller_batch: bool) -> list:
    """
    This function is not used anymore. This function was orially designed to
    handle cases when the dataloader is expected to contain data with different
    size.
    Torch.Tensor doesn't not allow data with different sequence lengths,
    therefore cannot use the DataLoader in PyTorch.
    input_data: [ some data, some data, some data, ...]
    say batch_size is 2, the return_data would be
    [[some data, some data], [some data, some data], ...]
    When allow_smaller_batch is True, the last batch of the return_data
    may have less data sets than the batch_size. Length of the return_data
    equals to len(input_data) devided by batch_size + 1.
    When allow_smaller_batch is False, all batches have the same size and
    the length of return_data equals to len(input_data) devided by batch_size.
    """
    if shuffle:
        random.shuffle(input_data)
    return_data = []
    len_input_data = len(input_data)
    batch_num = len_input_data // batch_size
    last_batch_num = len_input_data % batch_size
    for i in range(batch_num):
        start_index = batch_size * i
        end_index = start_index + batch_size
        return_data.append(input_data[start_index:end_index])
    if allow_smaller_batch:
        return_data.append(input_data[-last_batch_num:])
    return return_data


def add_velocity_to_file(input_filename: str, output_filename: str,
                         position_column: str, velocity_column: str,
                         max_velocity: float, min_velocity: float,
                         last_velocity: float) -> None:
    """
    This function is not used anymore. Originally, this function is designed
    to add the velocity column to dataset that only contains position column

    example:
    position_column = 'x'
    velocity_column = 'velocity'
    max_velocity = 0.0010098153109304175
    min_velocity = -0.0009186215039586054
    last_velocity = 0.47635551077750504
    input_filename = 'dataset_1.csv'
    output_filename = 'dataset_1_withVelocity.csv'
    """
    data = pd.read_csv(input_filename)
    velocity = np.diff(data[position_column])
    normalized_velocity = (velocity - min_velocity) / (
        max_velocity - min_velocity)
    data[velocity_column] = np.append(normalized_velocity, last_velocity)
    data.to_csv(output_filename, index=False)
