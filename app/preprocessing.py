"""
Script for data preprocessing.
Preprocessing of data for training autoencoder models (function
 preprocess_for_training_autoencoder) and preprocessing of data for training
 classification models (function preprocess_for_training_classifiers).
"""

import pandas as pd
import numpy as np
from helper_func.utils import slice_by_pattern, slice_seq_to_windows
from helper_func.config import PreprocessAutoencoderConfig, \
    PreprocessClassificationConfig, AEConfig
from machine_learning.ae import AutoencoderIndividualWindow
import torch


def preprocess_for_training_autoencoder(config):
    """
    Prepare data for training autoencoder.
    This function takes the original data as input data, and slice the data
    into sequences based on given pattern. The output files are data sequences.
    One file is one data sequences.
    """
    data = pd.read_csv(config.input_filename, usecols=config.use_columns)
    data = data.dropna()
    velocity = np.diff(data[config.position_column])
    data[config.velocity_column] = np.append(velocity, config.last_velocity)

    data_normalized = ((data - data.min()) / (data.max() - data.min()))

    # when the output_filename attribute of function slice_by_pattern is not
    # None, function slice_by_pattern will write the slicing results into
    # files, therefore the following line doesn't have a return.
    slice_by_pattern(
        original_data=data_normalized,
        check_pattern_column=config.check_pattern_column,
        pattern_end=config.pattern_end,
        pattern_start=config.pattern_start,
        remove_pattern_column=config.remove_pattern_column,
        output_filename=config.output_filename,
        all_seq_len_output_filename=config.all_seq_len_output_filename)


def preprocess_for_training_classifiers(config, tag=None):
    """
    Prepare dataset for training classifiers.
    This function takes data sequences as input. For each data sequences,
    this function prepares two kinds of dataset: each_window versus mean_seq
        each_window:
            One data sequence is sliced into multiple windows. Each window
            produces one latent datapoint. Each latent datapoint is labelled
            by the leakage type of the original data sequence. In the output
            file of this function (the input file for training leakage
            classifier), one row represent one window.
        mean_seq:
            One data sequence is sliced into multiple windows. Each window
            produces one latent datapoint. The mean of all datapoints from the
            one data sequence is calculated, and is labelled by the leakage
            type of the original data sequence. In the output file of this
            function (the input file for training leakage classifier), one row
            represent one data sequence.

    :param config: please refer to and modify the class
        PreprocessClassificationConfig in app/helper_func/config.py
    :param tag: used to label models, e.g. 'linear_model'
    :return:
    """
    if tag is None:
        tag = '%s_%dfeatures_%dlatents' % (
            config.model_type, config.n_features, config.latent_dim)
    ae_config = AEConfig()
    ae_model = AutoencoderIndividualWindow(ae_config=ae_config)
    state_dict = torch.load(config.saved_model_path)
    ae_model.load_state_dict(state_dict)
    all_latent_mean_dataframe_each_window = pd.DataFrame()
    all_latent_mean_dataframe_mean_seq = pd.DataFrame()
    for seq_id in config.use_seq_id:
        filename = config.data_path / config.filename.replace('*', str(seq_id))
        input_file = pd.read_csv(filename, usecols=config.use_columns)
        label = input_file[config.input_label_column_name][0]
        input_file = input_file.drop(columns=[config.input_label_column_name])
        seq_tensor = torch.tensor(input_file.values)
        seq_windows = slice_seq_to_windows(
            seq_tensor, config.window_length, config.window_overlap_length,
            config.irregular_window_at_end)
        windows_tensor = torch.stack(seq_windows)
        _, latent_mean = ae_model(windows_tensor)

        # treat each windows individually
        latent_mean_dataframe_each_window = \
            pd.DataFrame(latent_mean).astype('float')
        latent_mean_dataframe_each_window[config.output_label_column_name] = \
            [label] * latent_mean.shape[0]
        all_latent_mean_dataframe_each_window = \
            pd.concat([all_latent_mean_dataframe_each_window,
                       latent_mean_dataframe_each_window])

        # each seq results in only one row
        mean_all_windows = torch.mean(latent_mean, dim=0)
        mean_all_windows = torch.unsqueeze(mean_all_windows, dim=0)
        latent_mean_dataframe_mean_seq = \
            pd.DataFrame(mean_all_windows.detach().numpy()).astype('float')
        latent_mean_dataframe_mean_seq[config.output_label_column_name] = \
            [label]
        all_latent_mean_dataframe_mean_seq = \
            pd.concat([all_latent_mean_dataframe_mean_seq,
                       latent_mean_dataframe_mean_seq])

    all_latent_mean_dataframe_each_window.to_csv(
        config.output_filename_each_window.replace('*', tag), index=False)
    all_latent_mean_dataframe_mean_seq.to_csv(
        config.output_filename_mean_seq.replace('*', tag), index=False)


if __name__ == '__main__':
    AE_PREPROCESS_CONFIG = PreprocessAutoencoderConfig()
    preprocess_for_training_autoencoder(AE_PREPROCESS_CONFIG)

    CLASSIFIER_CONFIG = PreprocessClassificationConfig()
    preprocess_for_training_classifiers(CLASSIFIER_CONFIG)
