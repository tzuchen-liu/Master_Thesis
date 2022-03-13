"""
Author: Tzuchen Liu
Supervisor: Faras Brumand
Master Thesis

This file contains functions and class for visualization purpose.
The resulting figures can be found in app/assets/images.
function:
    reconstruct_plot: to plot the original data and the data reconstructed by
     a target autoencoder model on a same figure.
    accuracy_plot: to plot accuracies of different classification models as
     bar plots.
    latent_statistics_bar: to plat statistics of latent values of autoencoder
     models as bar plot
    latent_statistics_line: to plat statistics of latent values of autoencoder
     models as line plot
    recon_loss_comparison_box: to plot reconstruction loss box plots of
     different autoencoder models. This function is used for journal figures.
Class:
    Visualization: this class contains functions to perform visualization task.
        loss: plots training and testing loss of autoencoder models
        seq_recon_loss: calculate and store the reconstruction loss of each
         data sequence versus autoencoder model
        reconstruct_specific_seq: to plot reconstruction data of a specific
         data sequence
        reconstruct_general: to plot reconstruction data of important data
         sequences, to have a better over view of autoencoder model's
         performance
        reconstruct_general_with_error: similar to reconstruct_general, but
         also plot error bars
        accuracy: plot accuracy bar plots of classification models
        recon_loss_comparison: plot reconstruction loss of different
         autoencoder models
        latent_statistics: plot statistics of latent values of autoencoder
         models

The codes under "if __name__ == '__main__':" are examples of using these
 functions and class
"""

from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from app.machine_learning.ae import AutoencoderIndividualWindow
from app.helper_func.config import VisualizationConfig, AEConfig
from app.helper_func.utils import slice_seq_to_windows, windows_to_seq, \
    calculate_mse_loss


def reconstruct_plot(ae_model: AutoencoderIndividualWindow,
                     visual_config: VisualizationConfig,
                     data_full_filename: str,
                     output_full_filename: str,
                     usecols: List[str]) -> None:
    """
    This function is designed to be used by reconstruction plotting functions
    in class Visualization below.
    :param ae_model: the autoencoder model to reconstruct data
    :param visual_config: settings regarding visualization of the
     reconstruction data plots
    :param data_full_filename: the filename including the path of the
     input data
    :param output_full_filename: The '*' symbol will be replaced by each
     column of the usecols.
    :param usecols: the columns involved this analysis
    """

    data = torch.tensor(pd.read_csv(data_full_filename,
                                    usecols=usecols).values)
    original_windows = torch.stack(slice_seq_to_windows(
        data, visual_config.window_length,
        visual_config.window_overlap_length,
        visual_config.irregular_window_at_end))

    with torch.no_grad():
        ae_model.eval()
        reconstructed_windows, _ = ae_model(original_windows)

    reconstructed_seq = windows_to_seq(
        reconstructed_windows,
        len(data),
        visual_config.window_overlap_length,
        visual_config.irregular_window_at_end,
        visual_config.overlap_ignore_later_window)

    for index, column in enumerate(usecols):
        fig_ax = plt.subplot()
        fig_ax.plot(data[:, index].detach().numpy(), '-', color='dimgray',
                    label='Original Data', alpha=0.7, linewidth=3)
        fig_ax.plot(reconstructed_seq[:, index].detach().numpy(),
                    linestyle=(0, (5, 10)), color='lightgray',
                    label='Reconstructed Data', alpha=0.7, linewidth=3)
        if column == 'x':
            plt.ylabel('Normalized Cylinder Position x [-]',
                       fontsize=visual_config.font_size)
        elif column == 'valve_status':
            plt.ylabel('Cylinder Valve Status',
                       fontsize=visual_config.font_size)
        else:
            plt.ylabel('Normalized Cylinder Velocity v [-]',
                       fontsize=visual_config.font_size)
        plt.xlabel('Simulation Time t [ms]', fontsize=visual_config.font_size)

        fig_ax.xaxis.set_tick_params(labelsize=visual_config.axis_tick_size)
        fig_ax.yaxis.set_tick_params(labelsize=visual_config.axis_tick_size)
        plt.legend(prop={'size': visual_config.legend_size})
        plt.grid()
        # plt.show()
        plt.savefig(output_full_filename.replace('*', column),
                    bbox_inches='tight')
        plt.close()


def accuracy_plot(data_left: List[float],
                  label_left: str,
                  data_middle_left: List[float],
                  label_middle_left: str,
                  data_middle_right: List[float],
                  label_middle_right: str,
                  data_right: List[float],
                  label_right: str,
                  x_labels: List[str],
                  visual_config: VisualizationConfig,
                  output_full_filename: str) -> None:
    """
    This function is designed to be used by accuracy plotting functions
    in class Visualization below
    :param data_left: a list of pre-calculated classification accuracy results,
     the value ranges from 0 to 100 (percentage). These values will be plotted
     as the left most bar
    :param label_left: the label of data_left
    :param data_middle_left: a list of pre-calculated classification accuracy
     results, the value ranges from 0 to 100 (percentage). These values will be
     plotted as the second left (the middle left) bar
    :param label_middle_left: the label of data_middle_left
    :param data_middle_right: a list of pre-calculated classification accuracy
     results, the value ranges from 0 to 100 (percentage). These values will be
     plotted as the second right (the middle right) bar
    :param label_middle_right: the label of data_middle_right
    :param data_right: a list of pre-calculated classification accuracy
     results, the value ranges from 0 to 100 (percentage). These values will be
     plotted as the right most bar
    :param label_right: the label of data_right
    :param x_labels: the labels of x axis of the bar plot
    :param visual_config: settings regarding visualization of the bar plots
    :param output_full_filename: The filename of the output figures
    """
    _, fig_ax = plt.subplots()
    x_ticks_location = np.arange(len(x_labels))
    _ = fig_ax.bar(
        x_ticks_location - visual_config.bar_width * 1.5, data_left,
        visual_config.bar_width, label=label_left, hatch='/', color='#00549F')
    _ = fig_ax.bar(
        x_ticks_location - visual_config.bar_width * 0.5, data_middle_left,
        visual_config.bar_width,
        label=label_middle_left, hatch='/', color='gray')
    _ = fig_ax.bar(
        x_ticks_location + visual_config.bar_width * 0.5, data_middle_right,
        visual_config.bar_width,
        label=label_middle_right, color='#00549F')
    _ = fig_ax.bar(
        x_ticks_location + visual_config.bar_width * 1.5, data_right,
        visual_config.bar_width, label=label_right, color='gray')

    fig_ax.set_ylabel('Accuracy [%]', fontsize=visual_config.font_size)
    fig_ax.set_xlabel('Models', fontsize=visual_config.font_size)
    fig_ax.set_xticks(x_ticks_location)
    fig_ax.set_xticklabels(x_labels, fontsize=visual_config.axis_tick_size)
    fig_ax.yaxis.set_tick_params(labelsize=visual_config.axis_tick_size)
    # fig_ax.set_ylim(97.5, 100)
    fig_ax.legend(loc='upper center', prop={'size': visual_config.legend_size},
                  bbox_to_anchor=(1.25, 1.03), ncol=1, fancybox=True)

    plt.grid()
    # plt.show()
    plt.savefig(output_full_filename, bbox_inches='tight')
    plt.close()


def latent_statistics_bar(data_left_max: List[float],
                          label_left_max: str,
                          data_left_min: List[float],
                          label_left_min: str,
                          data_right_max: List[float],
                          label_right_max: str,
                          data_right_min: List[float],
                          label_right_min: str,
                          data_left_mean: List[float],
                          data_left_std: List[float],
                          data_right_mean: List[float],
                          data_right_std: List[float],
                          visual_config: VisualizationConfig,
                          output_full_filename: str,
                          x_ticks=None, shift: float = 0.2,
                          elinewidth: int = 10, capsize: int = 10,
                          capthick: int = 5) -> None:
    """
    This function is designed to be used by function latent_statistics
    in class Visualization below.
    :param data_left_max: the maximum values of the data in the left bar
    :param label_left_max: the label of data_left_max
    :param data_left_min: the minimum values of the data in the left bar
    :param label_left_min:the label of data_left_min
    :param data_right_max:the maximum values of the data in the right bar
    :param label_right_max: the label of data_right_max
    :param data_right_min:the minimum values of the data in the right bar
    :param label_right_min: the label of data_right_min
    :param data_left_mean: the mean values of the data in the left bar
    :param data_left_std: the standard deviation values of the data in the
     left bar
    :param data_right_mean: the mean values of the data in the right bar
    :param data_right_std: the standard deviation values of the data in the
     right bar
    :param visual_config: settings regarding visualization of the plot
    :param output_full_filename: the filename of the output figures
    :param x_ticks: the x labels of each group of bars
    :param shift: this is used to adjust the horizontal distance between bars
    :param elinewidth, capsize, capthick: parameters of the errorbar function
     in matplotlib, please refer to the documentation there.
    """
    if x_ticks is None:
        x_ticks = ['F2_L2', 'F2_L4', 'F2_L64', 'F3_L3', 'F3_L4', 'F3_L64']

    x_ticks_location = np.arange(len(x_ticks))  # the label locations
    _, fig_ax = plt.subplots()
    fig_ax.plot(x_ticks_location - shift, data_left_max, '^',
                label=label_left_max, color='b', markersize=10)
    fig_ax.plot(x_ticks_location - shift, data_left_min, 's',
                label=label_left_min, color='b', markersize=10)
    fig_ax.plot(x_ticks_location + shift, data_right_max, '^',
                label=label_right_max, color='magenta', markersize=10)
    fig_ax.plot(x_ticks_location + shift, data_right_min, 's',
                label=label_right_min, color='magenta', markersize=10)
    fig_ax.errorbar(x_ticks_location - shift, data_left_mean,
                    yerr=data_left_std,
                    fmt=' ', color='b', elinewidth=elinewidth, capsize=capsize,
                    capthick=capthick)
    fig_ax.errorbar(x_ticks_location + shift, data_right_mean,
                    yerr=data_right_std, fmt=' ', color='magenta',
                    elinewidth=elinewidth, capsize=capsize, capthick=capthick)
    fig_ax.set_ylabel('Latent Value', fontsize=visual_config.font_size)
    fig_ax.set_xlabel('Model Name', fontsize=visual_config.font_size)
    fig_ax.set_xticks(x_ticks_location)
    fig_ax.set_xticklabels(x_ticks, fontsize=visual_config.axis_tick_size)
    fig_ax.legend(prop={'size': visual_config.legend_size})
    plt.grid()
    plt.savefig(output_full_filename, bbox_inches='tight')
    plt.close()


def latent_statistics_line(data1: List[float],
                           label1: str,
                           data2: List[float],
                           label2: str,
                           data3: List[float],
                           label3: str,
                           data4: List[float],
                           label4: str,
                           y_label: str,
                           visual_config: VisualizationConfig,
                           output_full_filename: str,
                           x_ticks=None) -> None:
    """
    This function is designed to be used by function latent_statistics
    in class Visualization below.
    :param data1: one of the data to be plotted
    :param label1: the label of data1
    :param data2: one of the data to be plotted
    :param label2: the label of data2
    :param data3: one of the data to be plotted
    :param label3: the label of data3
    :param data4: one of the data to be plotted
    :param label4: the label of data4
    :param y_label: the label of the y axis
    :param visual_config: settings regarding visualization of the plot
    :param output_full_filename: the filename of the output figures
    :param x_ticks: the values/labels to show under the x axis
    :return:
    """
    if x_ticks is None:
        x_ticks = ['F2_L2', 'F2_L4', 'F2_L64', 'F3_L3', 'F3_L4', 'F3_L64']

    x_ticks_location = np.arange(len(x_ticks))  # the label locations
    _, fig_ax = plt.subplots()
    fig_ax.plot(
        x_ticks_location, data1, label=label1, marker='o', color='blue')
    fig_ax.plot(
        x_ticks_location, data2, label=label2, marker='^', color='gray')
    fig_ax.plot(
        x_ticks_location, data3, label=label3, marker='X', color='magenta')
    fig_ax.plot(
        x_ticks_location, data4, label=label4, marker='D', color='orange')
    fig_ax.set_ylabel(y_label, fontsize=visual_config.font_size)
    fig_ax.set_xlabel('Model Name', fontsize=visual_config.font_size)
    fig_ax.set_xticks(x_ticks_location)
    fig_ax.set_xticklabels(x_ticks, visual_config.axis_tick_size)
    fig_ax.legend(prop={'size': visual_config.legend_size})
    plt.grid()
    plt.savefig(output_full_filename, bbox_inches='tight')
    plt.close()


class Visualization:
    """
    This class contains functions to visualize the loss when training
     autoencoder models, the reconstruction loss, the latent values, and the
     accuracy. When initializing this class, many arguments are required, but
     can later be changed according to different needs of different functions.
     For most cases, they wouldn't need to be customize much though.
    """
    def __init__(self, default_input_dir: str, default_output_dir: str,
                 default_model_dir: str, model_list: List[str],
                 font_size: int, legend_size: int, axis_tick_size: int):
        self.default_input_dir = default_input_dir
        self.default_output_dir = default_output_dir
        self.default_model_dir = default_model_dir
        self.model_list = model_list
        self.font_size = font_size
        self.legend_size = legend_size
        self.axis_tick_size = axis_tick_size

    def loss(self, loss_dir: str = None, output_dir: str = None,
             font_size: int = None, legend_size: int = None,
             axis_tick_size: int = None) -> None:
        """
        This function plots the loss and log-scale loss of trained models.
        The files that have 'loss_' in the name and locate in the
        app/assets/images are plotted by this function.

        input files:
        filename of training loss is 'train_loss_' + $model_name
            e.g. train_loss_Linear_F2_L2
        filename of testing loss is 'test_loss_' + $model_name
            e.g. test_loss_Linear_F2_L2
        these two input files are in csv format (but without file extension),
            and has one column 'epoch' stores the epoch numbers, and another
            column 'loss' stores the loss values
        :param loss_dir: the directory of where the above mentioned
         input files are
        :param output_dir: the directory that the output plots will be stored
        :param font_size: font size of the x and y label
        :param legend_size: font size of words in the legend box
        :param axis_tick_size: font size of x and y ticks
        """
        if loss_dir is None:
            loss_dir = self.default_input_dir
        if output_dir is None:
            output_dir = self.default_output_dir
        if font_size is None:
            font_size = self.font_size
        if legend_size is None:
            legend_size = self.legend_size
        if axis_tick_size is None:
            axis_tick_size = self.axis_tick_size

        for model_name in self.model_list:
            print(model_name)
            for ylabel, plot_type in zip(
                    ['Loss', 'Log Loss'], ['loss', 'log_loss']):
                train_loss = pd.read_csv(loss_dir + 'train_loss_' + model_name)
                test_loss = pd.read_csv(loss_dir + 'test_loss_' + model_name)
                fig_ax = plt.subplot()
                fig_ax.plot(train_loss['epoch'], train_loss['loss'], alpha=0.7,
                            label='Training Loss', linewidth=5.0)
                fig_ax.plot(test_loss['epoch'], test_loss['loss'], alpha=0.7,
                            label='Testing Loss', linewidth=5.0)
                plt.xlabel('Number of Epochs', fontsize=font_size)
                plt.ylabel(ylabel, fontsize=font_size)
                fig_ax.xaxis.set_tick_params(labelsize=axis_tick_size)
                fig_ax.yaxis.set_tick_params(labelsize=axis_tick_size)
                plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                if plot_type == 'log_loss':
                    fig_ax.set_yscale('log')
                plt.legend(prop={'size': legend_size})
                plt.grid()
                # plt.show()
                plt.savefig(
                    output_dir + '%s_' % plot_type + model_name + '.pdf',
                    bbox_inches='tight')
                plt.close()

    def seq_recon_loss(self, seq_id_list: List[int],
                       input_filename: str = 'dataset_*.csv',
                       data_dir: str = None,
                       model_dir: str = None,
                       model_list: List[str] = None,
                       output_dir: str = None,
                       output_filename:
                       str = 'recon_loss_all_seq_indiv_feature.csv'):
        """
        This function generates a csv file that stores the reconstruction loss
        of each data sequence versus each autoencoder model.
        Each row in the file represent a data sequence, and each column
        corresponds to an autoencoder model.
        Every value means the reconstruction loss of this data sequence (row)
        being processed by this autoencoder model (column).
        The out file is used as an input to function reconstruct_general and
        reconstruct_general_with_error.
        :param seq_id_list: a python list of target sequence IDs
        :param input_filename: the filename of the target data sequence
                (csv file)
        :param data_dir: the directory that the input_filename locates
        :param model_dir: the directory that the autoencoder models were stored
        :param model_list: a python list of target model names
        :param output_dir: the directory that the output plots will be stored
        :param output_filename: the filename of the output csv file
        """
        if data_dir is None:
            data_dir = self.default_input_dir
        if model_dir is None:
            model_dir = self.default_model_dir
        if model_list is None:
            model_list = self.model_list
        if output_dir is None:
            output_dir = self.default_output_dir

        visual_config = VisualizationConfig()
        ae_config = AEConfig()
        output = {}
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
            recon_loss_all_seq_x = []
            recon_loss_all_seq_valve_status = []
            recon_loss_all_seq_velocity = []
            for seq_id in seq_id_list:
                data = torch.tensor(
                    pd.read_csv(
                        data_dir + input_filename.replace('*', str(seq_id)),
                        usecols=usecols).values)

                original_windows = torch.stack(slice_seq_to_windows(
                    data, visual_config.window_length,
                    visual_config.window_overlap_length,
                    visual_config.irregular_window_at_end))

                with torch.no_grad():
                    ae_model.eval()
                    reconstructed_windows, _ = ae_model(original_windows)

                    reconstructed_window_remove_batch_size = torch.squeeze(
                        reconstructed_windows)

                    mse = torch.nn.MSELoss()
                    recon_loss_all_seq_x.append(
                        mse(original_windows[:, :, 0],
                            reconstructed_window_remove_batch_size[:, :, 0]
                            ).item())
                    recon_loss_all_seq_valve_status.append(
                        mse(original_windows[:, :, 1],
                            reconstructed_window_remove_batch_size[:, :, 1]
                            ).item())
                    if original_windows.shape[2] > 2:
                        recon_loss_all_seq_velocity.append(
                            mse(original_windows[:, :, 2],
                                reconstructed_window_remove_batch_size[:, :, 2]
                                ).item())

                    reconstruction_loss = calculate_mse_loss(
                        original_windows, reconstructed_windows)
                    recon_loss_all_seq.append(reconstruction_loss.item())
            output[model_name] = recon_loss_all_seq
            output[model_name + '_x'] = recon_loss_all_seq_x
            output[
                model_name + '_valve_status'] = recon_loss_all_seq_valve_status
            if recon_loss_all_seq_velocity:
                output[model_name + '_velocity'] = recon_loss_all_seq_velocity
        output['seq_id'] = seq_id_list
        dataframe = pd.DataFrame.from_dict(output)
        dataframe.to_csv(output_dir + output_filename,
                         index=False,
                         header=True, columns=['seq_id'] + model_list)

    def reconstruct_specific_seq(self,
                                 input_filename,
                                 data_dir: str = None,
                                 model_dir: str = None,
                                 model_list: List[str] = None,
                                 output_dir: str = None,
                                 font_size: int = None,
                                 legend_size: int = None,
                                 axis_tick_size: int = None,
                                 figure_tag: str = '') -> None:
        """
        This function takes the data file of one specific data sequence, and
        plots the original data and the reconstructed data (reconstructed via
        models provided in the model_list) on the same plot.
        The files with filename as reconstruct_$modelName_$columnName and
         locate in the app/assets/images are plotted by this function.
        (e.g. reconstruct_LSTM_F2_L2_x.pdf,
            reconstruct_LSTM_F2_L2_valve_status.pdf)
        :param input_filename: the filename of the target data sequence
                (csv file)
        :param data_dir: the directory that the input_filename locates
        :param model_dir: the directory that the autoencoder models were stored
        :param model_list: a python list of target model names
        :param output_dir: the directory that the output plots will be stored
        :param font_size: font size of the x and y label
        :param legend_size: font size of words in the legend box
        :param axis_tick_size: font size of x and y ticks
        """
        if data_dir is None:
            data_dir = self.default_input_dir
        if model_dir is None:
            model_dir = self.default_model_dir
        if model_list is None:
            model_list = self.model_list
        if output_dir is None:
            output_dir = self.default_output_dir

        visual_config = VisualizationConfig()
        visual_config.font_size = self.font_size \
            if font_size is None else font_size
        visual_config.legend_size = self.legend_size \
            if legend_size is None else legend_size
        visual_config.axis_tick_size = self.axis_tick_size \
            if axis_tick_size is None else axis_tick_size

        ae_config = AEConfig()
        for model_name in model_list:
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

            reconstruct_plot(ae_model, visual_config,
                             data_dir + input_filename,
                             output_dir + 'reconstruct_%s_*%s.pdf' % (
                                 model_name, figure_tag),
                             usecols)

    def reconstruct_general(self, input_filename,
                            seq_for_recon_path: str,
                            data_dir: str = None,
                            model_dir: str = None,
                            model_list: List[str] = None,
                            output_dir: str = None,
                            font_size: int = None,
                            legend_size: int = None,
                            axis_tick_size: int = None) -> None:
        """
        This function is similar to function reconstruct_specific_seq, it also
        plots both the original and reconstructed on the same figure. This
        function serves to answer the question: which data sequence to plot?
        For each autoencoder model, this function uses three data sequences
        and produces three reconstruction plots. The three data sequences are
        chosen based on the reconstruction loss, and the data sequences that
        has the minimum, median, maximum reconstruction losses are chosen and
        plotted by this function.
        Files named as reconstruct_$modelName_$columnName_$sequenceTag and
         locate in the app/assets/images are plotted by this function.
        (e.g. reconstruct_LSTM_F2_L2_x_min_loss.pdf,
                reconstruct_LSTM_F2_L2_x_median_loss.pdf,
                reconstruct_LSTM_F2_L2_x_max_loss.pdf)
        :param seq_for_recon_path: Path to a file that contains the sequence
        IDs that cause the corresponding autoencoder models to have the
        minimum, median, maximum reconstruction loss.
        This file can be generated by function seq_for_recon_plot in
        app/helper_func/utils.py
        :param input_filename: the filename of a data sequence. This filename
        includes symbol '*', and this symbol will be replaced by sequence ID.
        For example, input_filename = 'dataset_*.csv'. When this function plots
        the reconstruction figures of sequence ID 100, this function will read
        the data by reading file 'dataset_100.csv'.
        :param data_dir: the directory that the input_filename locates
        :param model_dir: the directory that the autoencoder models were stored
        :param model_list: a python list of target model names
        :param output_dir: the directory that the output plots will be stored
        :param font_size: font size of the x and y label
        :param legend_size: font size of words in the legend box
        :param axis_tick_size: font size of x and y ticks
        """
        if data_dir is None:
            data_dir = self.default_input_dir
        if model_dir is None:
            model_dir = self.default_model_dir
        if model_list is None:
            model_list = self.model_list
        if output_dir is None:
            output_dir = self.default_output_dir

        visual_config = VisualizationConfig()
        visual_config.font_size = self.font_size \
            if font_size is None else font_size
        visual_config.legend_size = self.legend_size \
            if legend_size is None else legend_size
        visual_config.axis_tick_size = self.axis_tick_size \
            if axis_tick_size is None else axis_tick_size

        ae_config = AEConfig()
        for model_name in model_list:
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

            seq_for_recon = pd.read_csv(seq_for_recon_path)
            row = seq_for_recon.loc[seq_for_recon['Model'] == model_name]
            for seq_tag in ['min_loss_seq_id', 'median_loss_seq_id',
                            'max_loss_seq_id']:
                reconstruct_plot(ae_model, visual_config,
                                 data_dir + input_filename.replace(
                                     '*', str(row[seq_tag].item())),
                                 output_dir + 'reconstruct_%s_*%s.pdf' % (
                                     model_name, '_%s' % seq_tag),
                                 usecols)

    def reconstruct_general_with_error(self, input_filename,
                                       seq_for_recon_path: str,
                                       data_dir: str = None,
                                       model_dir: str = None,
                                       model_list: List[str] = None,
                                       output_dir: str = None,
                                       font_size: int = None,
                                       legend_size: int = None,
                                       axis_tick_size: int = None) -> None:
        """
        This function is similar to function reconstruct_specific_seq, it also
        plots both the original and reconstructed on the same figure. This
        function serves to answer the question: which data sequence to plot?
        For each autoencoder model, this function uses three data sequences
        and produces three reconstruction plots. The three data sequences are
        chosen based on the reconstruction loss, and the data sequences that
        has the minimum, median, maximum reconstruction losses are chosen and
        plotted by this function.
        Files named as reconstruct_$modelName_$columnName_$sequenceTag and
         locate in the app/assets/images are plotted by this function.
        (e.g. reconstruct_LSTM_F2_L2_x_min_loss.pdf,
                reconstruct_LSTM_F2_L2_x_median_loss.pdf,
                reconstruct_LSTM_F2_L2_x_max_loss.pdf)
        :param seq_for_recon_path: Path to a file that contains the sequence
        IDs that cause the corresponding autoencoder models to have the
        minimum, median, maximum reconstruction loss.
        This file can be generated by function seq_for_recon_plot in
        app/helper_func/utils.py
        :param input_filename: the filename of a data sequence. This filename
        includes symbol '*', and this symbol will be replaced by sequence ID.
        For example, input_filename = 'dataset_*.csv'. When this function plots
        the reconstruction figures of sequence ID 100, this function will read
        the data by reading file 'dataset_100.csv'.
        :param data_dir: the directory that the input_filename locates
        :param model_dir: the directory that the autoencoder models were stored
        :param model_list: a python list of target model names
        :param output_dir: the directory that the output plots will be stored
        :param font_size: font size of the x and y label
        :param legend_size: font size of words in the legend box
        :param axis_tick_size: font size of x and y ticks
        """
        if data_dir is None:
            data_dir = self.default_input_dir
        if model_dir is None:
            model_dir = self.default_model_dir
        if model_list is None:
            model_list = self.model_list
        if output_dir is None:
            output_dir = self.default_output_dir

        visual_config = VisualizationConfig()
        visual_config.font_size = self.font_size \
            if font_size is None else font_size
        visual_config.legend_size = self.legend_size \
            if legend_size is None else legend_size
        visual_config.axis_tick_size = self.axis_tick_size \
            if axis_tick_size is None else axis_tick_size

        ae_config = AEConfig()
        for model_name in model_list:
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

            seq_for_recon = pd.read_csv(seq_for_recon_path)
            row = seq_for_recon.loc[seq_for_recon['Model'] == model_name]
            # for seq_tag in ['min_loss_seq_id', 'median_loss_seq_id',
            #                 'max_loss_seq_id']:
            for seq_tag in ['median_loss_seq_id']:
                data = torch.tensor(
                    pd.read_csv(data_dir + input_filename.replace(
                        '*', str(row[seq_tag].item())),
                                usecols=usecols).values)

                original_windows = torch.stack(slice_seq_to_windows(
                    data, visual_config.window_length,
                    visual_config.window_overlap_length,
                    visual_config.irregular_window_at_end))

                with torch.no_grad():
                    ae_model.eval()
                    reconstructed_windows, _ = ae_model(original_windows)

                reconstructed_seq = windows_to_seq(
                    reconstructed_windows,
                    len(data),
                    visual_config.window_overlap_length,
                    visual_config.irregular_window_at_end,
                    visual_config.overlap_ignore_later_window)

                for index, column in enumerate(usecols):
                    fig_ax = plt.subplot(211)
                    original_data = data[:, index].detach().numpy()
                    recon_data = reconstructed_seq[:, index].detach().numpy()
                    error = np.absolute(100 * (recon_data - original_data) / (
                        original_data + 0.001))
                    fig_ax.plot(original_data, '-', color='dimgray',
                                label='Original',
                                alpha=0.7, linewidth=3)
                    fig_ax.plot(recon_data, linestyle=(0, (5, 10)),
                                color='lightgray',
                                label='Reconstructed', alpha=0.7,
                                linewidth=3)  # , linestyle=(0, (5, 10))
                    if column == 'x':
                        plt.ylabel('Normalized Cylinder Position x [-]',
                                   fontsize=visual_config.font_size)
                    elif column == 'valve_status':
                        plt.ylabel('Cylinder Valve Status',
                                   fontsize=visual_config.font_size)
                    else:
                        plt.ylabel('Normalized Cylinder Velocity v [-]',
                                   fontsize=visual_config.font_size)
                    # plt.xlabel('Simulation Time t [ms]', fontsize=fontsize)

                    fig_ax.xaxis.set_tick_params(
                        labelsize=visual_config.axis_tick_size)
                    fig_ax.yaxis.set_tick_params(
                        labelsize=visual_config.axis_tick_size)
                    plt.legend(prop={'size': visual_config.legend_size})
                    plt.grid()

                    ax_error = plt.subplot(212)
                    ax_error.plot(error, 'k')
                    plt.xlabel('Simulation Time t [ms]',
                               fontsize=visual_config.font_size)
                    plt.ylabel('Error [%]',
                               fontsize=visual_config.font_size)
                    ax_error.xaxis.set_tick_params(
                        labelsize=visual_config.axis_tick_size)
                    ax_error.yaxis.set_tick_params(
                        labelsize=visual_config.axis_tick_size)
                    ax_error.set_ylim([0.0000000001, 150000000])
                    ax_error.set_yscale('log')

                    plt.grid()
                    # plt.show()
                    plt.savefig(
                        output_dir + 'reconstruct_with_error_%s_%s_%s.pdf' % (
                            model_name, column,
                            seq_tag.replace('_seq_id', '')),
                        bbox_inches='tight')
                    plt.close()

    def accuracy(self, output_dir: str = None,
                 font_size: int = None,
                 legend_size: int = None,
                 axis_tick_size: int = None) -> None:
        """
        This function plots accuracy comparison figures of different
         classification models.
        The files with filename starting as 'accuracy_' and locate in the
         app/assets/images are plotted by this function.
        (accuracy_lstm.pdf, accuracy_linear.pdf, accuracy_per_sequence.pdf)
        :param output_dir: the directory that the output plots will be stored
        :param font_size: font size of the x and y label
        :param legend_size: font size of words in the legend box
        :param axis_tick_size: font size of x and y ticks
        """
        if output_dir is None:
            output_dir = self.default_output_dir

        visual_config = VisualizationConfig()
        visual_config.font_size = self.font_size \
            if font_size is None else font_size
        visual_config.legend_size = self.legend_size \
            if legend_size is None else legend_size
        visual_config.axis_tick_size = self.axis_tick_size \
            if axis_tick_size is None else axis_tick_size

        x_labels = ['F2_L2', 'F2_L4', 'F2_L64', 'F3_L3', 'F3_L4', 'F3_L64']
        linear_rf_sequence = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        linear_mlp_sequence = [100.0, 99.15, 99.86, 100.0, 97.73, 100.0]
        linear_rf_window = [76.0, 92.0, 93.0, 94.0, 94.0, 94.0]
        linear_mlp_window = [78.0, 78.0, 84.0, 90.0, 92.0, 92.0]
        lstm_rf_sequence = [98.58, 100.0, 100.0, 100.0, 100.0, 100.0]
        lstm_mlp_sequence = [97.59, 98.01, 99.86, 100.0, 100.0, 100.0]
        lstm_rf_window = [66.0, 91.0, 93.0, 93.0, 94.0, 94.0]
        lstm_mlp_window = [77.0, 77.0, 91.0, 86.0, 92.0, 92.0]

        # Accuracy of Linear models:
        accuracy_plot(
            linear_rf_sequence, 'RF Sequence', linear_mlp_sequence,
            'MLP Sequence', linear_rf_window, 'RF Window', linear_mlp_window,
            'MLP Window', x_labels, visual_config,
            output_dir + 'accuracy_linear.pdf')

        # Accuracy of LSTM models:
        accuracy_plot(
            lstm_rf_sequence, 'RF Sequence', lstm_mlp_sequence, 'MLP Sequence',
            lstm_rf_window, 'RF Window', lstm_mlp_window, 'MLP Window',
            x_labels, visual_config, output_dir + 'accuracy_linear.pdf')

        # Accuracy of Classifying by Sequences Models:
        accuracy_plot(
            lstm_rf_sequence, 'RF LSTM', lstm_mlp_sequence, 'MLP LSTM',
            linear_rf_sequence, 'RF Linear', linear_mlp_sequence, 'MLP Linear',
            x_labels, visual_config, output_dir + 'accuracy_linear.pdf')

    def recon_loss_comparison(
            self, data_dir: str = None,
            recon_loss_filename: str = 'recon_loss_all_seq.csv',
            output_dir: str = None, font_size: int = None,
            legend_size: int = None, axis_tick_size: int = None) -> None:
        """
        This function plots reconstruction loss comparison between different
         autoencoder models.
        The files 'averaged_recon_loss.pdf' located in the
         app/assets/images are plotted by this function.
        :param data_dir: the directory that the recon_loss_filename locates
        :param recon_loss_filename: the file generated by the function
         self.seq_recon_loss, but doesn't differentiate different columns.
        :param output_dir: the directory that the output plots will be stored
        :param font_size: font size of the x and y label
        :param legend_size: font size of words in the legend box
        :param axis_tick_size: font size of x and y ticks
        """
        if data_dir is None:
            data_dir = self.default_input_dir
        if output_dir is None:
            output_dir = self.default_output_dir
        if font_size is None:
            font_size = self.font_size
        if legend_size is None:
            legend_size = self.legend_size
        if axis_tick_size is None:
            axis_tick_size = self.axis_tick_size

        x_labels = ['F2_L2', 'F2_L4', 'F2_L64', 'F3_L3', 'F3_L4', 'F3_L64']
        width = 0.2  # the width of the bars

        recon_loss = pd.read_csv(data_dir + recon_loss_filename)
        linear_model_names = ['Linear_F2_L2', 'Linear_F2_L4', 'Linear_F2_L64',
                              'Linear_F3_L3', 'Linear_F3_L4', 'Linear_F3_L64']
        lstm_model_names = ['LSTM_F2_L2', 'LSTM_F2_L4', 'LSTM_F2_L64',
                            'LSTM_F3_L3', 'LSTM_F3_L4', 'LSTM_F3_L64']
        linear_loss = [
            np.mean(recon_loss[model]) for model in linear_model_names]
        lstm_loss = [
            np.mean(recon_loss[model]) for model in lstm_model_names]

        x_ticks_location = np.arange(len(x_labels))  # the label locations

        fig, fig_ax = plt.subplots()
        _ = fig_ax.bar(x_ticks_location - width * 0.5, linear_loss, width,
                       label='Linear Models', color='#00549f')
        _ = fig_ax.bar(x_ticks_location + width * 0.5, lstm_loss, width,
                       label='LSTM Models', color='#8ebae5')

        fig_ax.set_ylabel('Averaged Reconstruction Loss', fontsize=font_size)
        fig_ax.set_xlabel('Models', fontsize=font_size)
        fig_ax.set_xticks(x_ticks_location)
        fig_ax.set_xticklabels(x_labels, fontsize=axis_tick_size)
        fig_ax.yaxis.set_tick_params(labelsize=axis_tick_size)
        fig_ax.set_yscale('log')

        ax_compression_rate = fig_ax.twinx()
        compression_rate = [99.5, 99, 84, 99.5, 99.3, 89.3]
        ax_compression_rate.plot(compression_rate, 'ko-',
                                 label='Compression Rate')
        ax_compression_rate.set_ylabel('Compression Rate [%]',
                                       fontsize=font_size)
        ax_compression_rate.set_ylim(70, 100)

        fig.legend(prop={'size': legend_size}, loc='upper center',
                   bbox_to_anchor=(1.1, 0.91), ncol=1, fancybox=True)
        plt.grid()
        # plt.show()
        plt.savefig(output_dir +
                    'averaged_recon_loss_with_compression_rate.pdf',
                    bbox_inches='tight')
        plt.close()

    def latent_statistics(
            self, statistics_mean_of_dims_dir: str = None,
            statistics_mean_of_dims_filename='statistics_mean_of_dims.csv',
            output_dir: str = None, font_size: int = None,
            legend_size: int = None, axis_tick_size: int = None) -> None:
        """
        This function plots statistics of the latent space and show the
         comparison between different autoencoder models.
        The files that have 'latent_statistics' in the name and locate in the
        app/assets/images are plotted by this function.
        (e.g. latent_statistics_each_window.pdf, latent_statistics_mean_seq.pdf
        latent_statistics_linear.pdf, latent_statistics_lstm.pdf,
        latent_statistics_skew.pdf, latent_statistics_kurtosis.pdf)
        :param statistics_mean_of_dims_filename: the file that stores the
         statistics of the latent values
        :param statistics_mean_of_dims_dir: the directory where the
         statistics_mean_of_dims_filename is stored
        :param output_dir: the directory that the output plots will be stored
        :param font_size: font size of the x and y label
        :param legend_size: font size of words in the legend box
        :param axis_tick_size: font size of x and y ticks
        """
        if statistics_mean_of_dims_dir is None:
            statistics_mean_of_dims_dir = self.default_input_dir
        if output_dir is None:
            output_dir = self.default_output_dir

        visual_config = VisualizationConfig()
        visual_config.font_size = self.font_size \
            if font_size is None else font_size
        visual_config.legend_size = self.legend_size \
            if legend_size is None else legend_size
        visual_config.axis_tick_size = self.axis_tick_size \
            if axis_tick_size is None else axis_tick_size

        statistics = pd.read_csv(
            statistics_mean_of_dims_dir + statistics_mean_of_dims_filename,
            index_col=0)
        columns = statistics.columns
        df_linear_each_window = statistics[columns[:6]]
        df_lstm_each_window = statistics[columns[6:12]]
        df_linear_mean_seq = statistics[columns[12:18]]
        df_lstm_mean_seq = statistics[columns[18:]]

        mean_linear_each_window = df_linear_each_window.loc['mean']
        std_linear_each_window = df_linear_each_window.loc['std']
        min_linear_each_window = df_linear_each_window.loc['min']
        max_linear_each_window = df_linear_each_window.loc['max']
        skew_linear_each_window = df_linear_each_window.loc['skew']
        kurtosis_linear_each_window = df_linear_each_window.loc['kurtosis']

        mean_lstm_each_window = df_lstm_each_window.loc['mean']
        std_lstm_each_window = df_lstm_each_window.loc['std']
        min_lstm_each_window = df_lstm_each_window.loc['min']
        max_lstm_each_window = df_lstm_each_window.loc['max']
        skew_lstm_each_window = df_lstm_each_window.loc['skew']
        kurtosis_lstm_each_window = df_lstm_each_window.loc['kurtosis']

        mean_linear_mean_seq = df_linear_mean_seq.loc['mean']
        std_linear_mean_seq = df_linear_mean_seq.loc['std']
        min_linear_mean_seq = df_linear_mean_seq.loc['min']
        max_linear_mean_seq = df_linear_mean_seq.loc['max']
        skew_linear_mean_seq = df_linear_mean_seq.loc['skew']
        kurtosis_linear_mean_seq = df_linear_mean_seq.loc['kurtosis']

        mean_lstm_mean_seq = df_lstm_mean_seq.loc['mean']
        std_lstm_mean_seq = df_lstm_mean_seq.loc['std']
        min_lstm_mean_seq = df_lstm_mean_seq.loc['min']
        max_lstm_mean_seq = df_lstm_mean_seq.loc['max']
        skew_lstm_mean_seq = df_lstm_mean_seq.loc['skew']
        kurtosis_lstm_mean_seq = df_lstm_mean_seq.loc['kurtosis']

        # each window: mean, std, min, max, linear vs lstm
        latent_statistics_bar(max_linear_each_window, 'Linear. Max.',
                              min_linear_each_window, 'Linear. Min.',
                              max_lstm_each_window, 'LSTM. Max.',
                              min_lstm_each_window, 'LSTM. Min.',
                              mean_linear_each_window, std_linear_each_window,
                              mean_lstm_each_window, std_lstm_each_window,
                              visual_config,
                              output_dir + 'latent_statistics_each_window.pdf')

        # mean seq: mean, std, min, max, linear vs lstm
        latent_statistics_bar(max_linear_mean_seq, 'Linear. Max.',
                              min_linear_mean_seq, 'Linear. Min.',
                              max_lstm_mean_seq, 'LSTM. Max.',
                              min_lstm_mean_seq, 'LSTM. Min.',
                              mean_linear_mean_seq, std_linear_mean_seq,
                              mean_lstm_mean_seq, std_lstm_mean_seq,
                              visual_config,
                              output_dir + 'latent_statistics_mean_seq.pdf')

        # linear: mean, std, min, max, each window vs mean seq
        latent_statistics_bar(max_linear_each_window, 'W. Max.',
                              min_linear_each_window, 'W. Min.',
                              max_linear_mean_seq, 'S. Max.',
                              min_linear_mean_seq, 'S. Min.',
                              mean_linear_each_window, std_linear_each_window,
                              mean_linear_mean_seq, std_linear_mean_seq,
                              visual_config,
                              output_dir + 'latent_statistics_linear.pdf')

        # lstm: mean, std, min, max, each window vs mean seq
        latent_statistics_bar(max_lstm_each_window, 'W. Max.',
                              min_lstm_each_window, 'W. Min.',
                              max_lstm_mean_seq, 'S. Max.',
                              min_lstm_mean_seq, 'S. Min.',
                              mean_lstm_each_window, std_lstm_each_window,
                              mean_lstm_mean_seq, std_lstm_mean_seq,
                              visual_config,
                              output_dir + 'latent_statistics_lstm.pdf')

        # skew: (each window, mean seq) x (linear, lstm)
        latent_statistics_line(skew_lstm_mean_seq, 'LSTM. S',
                               skew_lstm_each_window, 'LSTM. W',
                               skew_linear_mean_seq, 'Linear. S',
                               skew_linear_each_window, 'Linear. W',
                               'Skew Value', visual_config,
                               output_dir + 'latent_statistics_skew.pdf')

        # kurtosis: (each window, mean seq) x (linear, lstm)
        latent_statistics_line(kurtosis_lstm_mean_seq, 'LSTM. S',
                               kurtosis_lstm_each_window, 'LSTM. W',
                               kurtosis_linear_mean_seq, 'Linear. S',
                               kurtosis_linear_each_window, 'Linear. W',
                               'Kurtosis Value', visual_config,
                               output_dir + 'latent_statistics_kurtosis.pdf')


def recon_loss_comparison_box(
        output_dir: str,
        recon_loss_path: str = 'recon_loss_all_seq_indiv_feature.csv') -> None:
    """
    This function is completely customized for generating figures for the
     journal, so the figure settings (e.g. y scale limits) are all
     customized based on the data.
    :param output_dir: the directory that the output plots will be stored
    :param recon_loss_path: the file generated by the function seq_recon_loss
     in class Visualization
    """
    font_size = 18
    x_labels = ['F2_L2', 'F2_L4', 'F2_L64', 'F3_L3', 'F3_L4', 'F3_L64']

    recon_loss = pd.read_csv(recon_loss_path)
    linear_model_names = ['Linear_F2_L2', 'Linear_F2_L4', 'Linear_F2_L64',
                          'Linear_F3_L3', 'Linear_F3_L4', 'Linear_F3_L64']
    lstm_model_names = ['LSTM_F2_L2', 'LSTM_F2_L4', 'LSTM_F2_L64',
                        'LSTM_F3_L3', 'LSTM_F3_L4', 'LSTM_F3_L64']

    x_ticks_location = np.arange(1, len(x_labels) + 1)  # the label locations

    # linear models
    for feature in ['', '_x', '_valve_status']:
        data_boxplot = [recon_loss[model + feature] for model in
                        linear_model_names]
        # plt.hist(recon_loss['Linear_F3_L3'])
        _, fig_ax = plt.subplots(figsize=(8, 6))
        box_plot = plt.boxplot(data_boxplot, showfliers=False)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians',
                        'caps']:
            plt.setp(box_plot[element], color='k')
        fig_ax.set_ylabel('Reconstruction Loss [-]', fontsize=font_size)
        fig_ax.set_xlabel('Models', fontsize=font_size)
        fig_ax.set_xticks(x_ticks_location)
        fig_ax.set_xticklabels(x_labels, fontsize=font_size)
        fig_ax.yaxis.set_tick_params(labelsize=font_size)
        plt.yscale('log')
        plt.grid()
        # plt.show()
        plt.savefig(
            output_dir + 'recon_loss_boxplot_linear%s.pdf' % feature,
            bbox_inches='tight')
        plt.close()

    # lstm models
    for feature in ['', '_x', '_valve_status']:
        data_boxplot = [recon_loss[model + feature] for model in
                        lstm_model_names]
        _, fig_ax = plt.subplots(figsize=(8, 6))
        box_plot = plt.boxplot(data_boxplot, showfliers=False)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians',
                        'caps']:
            plt.setp(box_plot[element], color='k')
        fig_ax.set_ylabel('Reconstruction Loss [-]', fontsize=font_size)
        fig_ax.set_xlabel('Models', fontsize=font_size)
        fig_ax.set_xticks(x_ticks_location)
        fig_ax.set_xticklabels(x_labels, fontsize=font_size)
        fig_ax.yaxis.set_tick_params(labelsize=font_size)
        fig_ax.set_ylim([0.00001, 0.001])
        plt.yscale('log')
        plt.grid()
        # plt.show()
        plt.savefig(output_dir + 'recon_loss_boxplot_lstm%s.pdf' % feature,
                    bbox_inches='tight')
        plt.close()

    x_ticks = ['F3_L3', 'F3_L4', 'F3_L64']
    x_ticks_location = np.arange(1, len(x_ticks) + 1)  # the label locations
    # linear model, feature = velocity (only 3-feature model have velocity)
    data_boxplot = [recon_loss[model + feature] for model in
                    ['Linear_F3_L3', 'Linear_F3_L4', 'Linear_F3_L64']]
    _, fig_ax = plt.subplots(figsize=(8, 6))
    box_plot = plt.boxplot(data_boxplot, showfliers=False)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians',
                    'caps']:
        plt.setp(box_plot[element], color='k')
    fig_ax.set_ylabel('Reconstruction Loss [-]', fontsize=font_size)
    fig_ax.set_xlabel('Models', fontsize=font_size)
    fig_ax.set_xticks(x_ticks_location)
    fig_ax.set_xticklabels(x_ticks, fontsize=font_size)
    fig_ax.yaxis.set_tick_params(labelsize=font_size)
    plt.yscale('log')
    plt.grid()
    # plt.show()
    plt.savefig(output_dir + 'recon_loss_boxplot_linear_velocity.pdf',
                bbox_inches='tight')
    plt.close()

    # lstm, feature = velocity
    data_boxplot = [recon_loss[model + feature] for model in
                    ['LSTM_F3_L3', 'LSTM_F3_L4', 'LSTM_F3_L64']]
    _, fig_ax = plt.subplots(figsize=(8, 6))
    box_plot = plt.boxplot(data_boxplot, showfliers=False)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians',
                    'caps']:
        plt.setp(box_plot[element], color='k')
    fig_ax.set_ylabel('Reconstruction Loss [-]', fontsize=font_size)
    fig_ax.set_xlabel('Models', fontsize=font_size)
    fig_ax.set_xticks(x_ticks_location)
    fig_ax.set_xticklabels(x_ticks, fontsize=font_size)
    fig_ax.yaxis.set_tick_params(labelsize=font_size)
    fig_ax.set_ylim([0.00001, 0.001])
    plt.yscale('log')
    plt.grid()
    # plt.show()
    plt.savefig(output_dir + 'recon_loss_boxplot_lstm_velocity.pdf',
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Usage examples:
    MODEL_LIST = ['Linear_F2_L2', 'Linear_F2_L4', 'Linear_F2_L64',
                  'Linear_F3_L3', 'Linear_F3_L4', 'Linear_F3_L64',
                  'LSTM_F2_L2', 'LSTM_F2_L4', 'LSTM_F2_L64',
                  'LSTM_F3_L3', 'LSTM_F3_L4', 'LSTM_F3_L64']
    LOSS_DIR = '/home/tzuchen/Master_Thesis/Data/AE/'
    MODEL_DIR = '/home/tzuchen/Master_Thesis/Data/Models/'
    DATA_DIR = \
        '/home/tzuchen/Master_Thesis/Data/dataset_sequences/'
    DATA_FILENAME = 'dataset_5000.csv'
    DATA_FILENAME_GENERAL = 'dataset_*.csv'
    OUTPUT_DIR = '/home/tzuchen/Master_Thesis/Data/Plots/'
    LATENT_DIR = '/home/tzuchen/Master_Thesis/Data/dataset_classification/'
    CLASSIFIER_DIR = '/home/tzuchen/Master_Thesis/Data/Models/Classifiers/RN/'
    STATISTICS_DIR = '/home/tzuchen/Master_Thesis/Data/Statistics/'
    STATISTICS_MEAN_OF_DIMS_FILENAME = 'statistics_mean_of_dims.csv'
    SEQ_FOR_RECON_PATH = \
        '/home/tzuchen/Master_Thesis/Data/seq_for_recon_plot.csv'
    RECON_LOSS_DIR = '/home/tzuchen/Master_Thesis/Data/'
    RECON_LOSS_FILENAME = 'recon_loss_all_seq.csv'
    RECON_LOSS_DETAILS_FILENAME = 'recon_loss_all_seq_indiv_feature.csv'

    VISUAL_OBJECT = Visualization(DATA_DIR, OUTPUT_DIR, MODEL_DIR, MODEL_LIST,
                                  16, 16, 16)

    VISUAL_OBJECT.loss(loss_dir=LOSS_DIR)
    VISUAL_OBJECT.reconstruct_specific_seq(DATA_FILENAME)
    VISUAL_OBJECT.reconstruct_general(
        DATA_FILENAME_GENERAL, SEQ_FOR_RECON_PATH)
    VISUAL_OBJECT.reconstruct_general_with_error(
        DATA_FILENAME_GENERAL, SEQ_FOR_RECON_PATH)
    VISUAL_OBJECT.accuracy()
    VISUAL_OBJECT.latent_statistics(
        statistics_mean_of_dims_dir=STATISTICS_DIR,
        statistics_mean_of_dims_filename=STATISTICS_MEAN_OF_DIMS_FILENAME)
    VISUAL_OBJECT.seq_recon_loss(seq_id_list=list(range(3395, 6415)))
    VISUAL_OBJECT.recon_loss_comparison(
        data_dir=RECON_LOSS_DIR, recon_loss_filename=RECON_LOSS_FILENAME)

    recon_loss_comparison_box(
        OUTPUT_DIR, RECON_LOSS_DIR + RECON_LOSS_DETAILS_FILENAME)
