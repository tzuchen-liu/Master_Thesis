"""
It's the file to store all configuration parameters.
TODO
"""
from pathlib import Path
import torch


class AEConfig:
    """ This class stores configurations for Autoencoder. """
    def __init__(self):
        # configuration in general
        self.n_features = 2
        self.dim_batch_size = 0
        self.window_length = 200
        self.latent_dim = 2
        self.model_type = 'linear'  # 'linear' or 'lstm'
        self.update()

    def update(self) -> None:
        """
        The details of the encoder and decoder layers are defined by the above
        parameter settings. When the values of the above parameters are
        changed, this function can be called and well re-calculate the layers.
        """
        # Encoder layer one: LSTM
        self.encoder_layer_one_input_size = self.n_features
        self.encoder_layer_one_hidden_size = self.latent_dim
        self.encoder_layer_one_num_layers = 1
        self.encoder_layer_one_batch_first = True
        self.encoder_layer_one_initial_hidden_state = torch.zeros(
            self.encoder_layer_one_num_layers, 1,
            self.encoder_layer_one_hidden_size)
        self.encoder_layer_one_initial_cell_state = torch.zeros(
            self.encoder_layer_one_num_layers, 1,
            self.encoder_layer_one_hidden_size)

        # Encoder layer two: Linear
        self.encoder_layer_two_in_features = self.window_length
        self.encoder_layer_two_out_features = self.window_length // 2

        # Encoder layer three: Linear
        self.encoder_layer_three_in_features = \
            self.encoder_layer_two_out_features
        self.encoder_layer_three_out_features = 1

        # Encoder layer mean: Linear
        self.encoder_layer_mean_in_features = self.latent_dim
        self.encoder_layer_mean_out_features = self.latent_dim

        # Encoder layer log_variance: Linear
        self.encoder_layer_var_in_features = self.latent_dim
        self.encoder_layer_var_out_features = self.latent_dim

        # Decoder layer one: Linear
        self.decoder_layer_one_in_features = \
            self.encoder_layer_three_out_features
        self.decoder_layer_one_out_features = \
            self.encoder_layer_three_in_features

        # Decoder layer two: Linear
        self.decoder_layer_two_in_features = \
            self.encoder_layer_two_out_features
        self.decoder_layer_two_out_features = \
            self.encoder_layer_two_in_features

        # Decoder layer three: LSTM
        self.decoder_layer_three_input_size = \
            self.encoder_layer_one_hidden_size
        self.decoder_layer_three_hidden_size = \
            self.encoder_layer_one_input_size
        self.decoder_layer_three_num_layers = \
            self.encoder_layer_one_num_layers
        self.decoder_layer_three_batch_first = \
            self.encoder_layer_one_batch_first
        self.decoder_layer_three_initial_hidden_state = torch.zeros(
            self.encoder_layer_one_num_layers, 1,
            self.decoder_layer_three_hidden_size)
        self.decoder_layer_three_initial_cell_state = torch.zeros(
            self.encoder_layer_one_num_layers, 1,
            self.decoder_layer_three_hidden_size)

        # Decoder layer four: LSTM
        self.decoder_layer_four_input_size = \
            self.n_features
        self.decoder_layer_four_hidden_size = \
            self.n_features
        self.decoder_layer_four_num_layers = \
            self.encoder_layer_one_num_layers
        self.decoder_layer_four_batch_first = \
            self.encoder_layer_one_batch_first
        self.decoder_layer_four_initial_hidden_state = torch.zeros(
            self.encoder_layer_one_num_layers, 1,
            self.decoder_layer_four_hidden_size)
        self.decoder_layer_four_initial_cell_state = torch.zeros(
            self.encoder_layer_one_num_layers, 1,
            self.decoder_layer_four_hidden_size)

        # Decoder layer four: LSTM, self-consistent
        self.decoder_layer_four_input_size_self = \
            self.n_features * 2
        self.decoder_layer_four_hidden_size_self = \
            self.n_features
        self.decoder_layer_four_num_layers_self = \
            self.encoder_layer_one_num_layers
        self.decoder_layer_four_batch_first_self = \
            self.encoder_layer_one_batch_first
        self.decoder_layer_four_initial_hidden_state_self = torch.zeros(
            self.encoder_layer_one_num_layers, 1,
            self.decoder_layer_four_hidden_size_self)
        self.decoder_layer_four_initial_cell_state_self = torch.zeros(
            self.encoder_layer_one_num_layers, 1,
            self.decoder_layer_four_hidden_size_self)


class TrainConfig:
    """
    This class stores configurations for the autoencoder's training process.
    Please refer to app/train.py for the training process.
    """
    def __init__(self):
        self.data_path = Path.cwd() / 'Dataset' / 'Dataset_Split'
        self.filename = 'dataset_*.csv'
        self.use_seq_id = list(range(2, 5))
        self.print_on_screen = True

        # config model
        self.load_saved_model = False
        self.saved_model_path = '/home/tzuchen/Master_Thesis/liu_ma/app/Log/' \
                                '2021-08-20_12-24-45/' \
                                'VAE_2021-08-20_12-24-45_epoch-150.pth'

        # data property
        self.columns = [
            'ID', 'cycle', 'x', 'leakage', 'lab', 'sim', 'valve_status',
            'velocity']
        self.use_columns = ['x', 'valve_status']

        # creating windows
        self.window_length = 200
        self.window_overlap_ratio = 0.1
        # the first {window_overlap_length} datapoints of the next sequence
        # is consist of the last sequence
        self.window_overlap_length = int(
            self.window_length * self.window_overlap_ratio)
        self.irregular_window_at_end = True  # when False means at beginning

        # prepare data for training
        self.test_ratio = 0.2  # between 0 and 1
        self.batch_size = 25
        self.shuffle = True
        self.allow_smaller_batch = False

        # training process
        self.starting_loss = 0
        self.learning_rate = 1e-3
        self.epochs = 150

        # log model and training loss
        self.backup_file_list = [
            Path.cwd() / 'train.py',
            Path.cwd() / 'machine_learning' / 'ae.py',
            Path.cwd() / 'helper_func' / 'config.py']
        self.model_name = 'AE'
        self.decoder_model_name = 'Decoder'
        self.train_loss_filename = 'train_loss'
        self.train_recon_loss_filename = 'train_recon_loss'
        self.train_kl_loss_filename = 'train_kl_loss'
        self.test_loss_filename = 'test_loss'
        self.test_recon_loss_filename = 'test_recon_loss'
        self.test_kl_loss_filename = 'test_kl_loss'
        self.log_dir = Path.cwd() / 'Log'
        self.latent_mean_variable_name = 'latent_mean'
        self.latent_log_var_variable_name = 'latent_log_var'
        self.save_model_interval = 1
        self.save_loss_interval = 1
        self.save_latent_values_interval = 1
        self.print_when_saving_model = True
        self.print_when_saving_loss = False
        self.print_when_saving_latent_variables = False


class VisualizationConfig:
    """
    This class stores configurations for the visualization methods.
    Please refer to app/visualization.py for the visualization methods.
    """
    def __init__(self):
        self.data_seq_id = list(range(2, 5))
        self.model_name = '/home/tzuchen/Master_Thesis/liu_ma/app/Log/' \
                          '2021-08-21_13-47-24/' \
                          'AE_2021-08-21_13-47-24_epoch-48.pth'
        self.data_path = Path.cwd() / 'Dataset' / 'Dataset_Split_Velocity'
        # self.data_path = Path.cwd().parent.parent / 'Data'
        # self.data_path = Path.cwd().parent.parent / 'Data' / \
        #                  'dataset_sequences'  # on cluster
        # self.data_path = Path.cwd().parent.parent / 'Data' / \
        #                  'dataset_sequences_velocity'  # on cluster
        self.filename = 'dataset_*.csv'
        self.window_length = 200
        self.window_overlap_ratio = 0.1
        self.window_overlap_length = int(
            self.window_length * self.window_overlap_ratio)
        self.irregular_window_at_end = True  # when False means at beginning

        # data property
        self.columns = ['x', 'valve_status']
        self.overlap_ignore_later_window = True

        # plotting settings
        self.font_size = 16
        self.legend_size = 16
        self.axis_tick_size = 16
        self.bar_width = 0.2


class PreprocessClassificationConfig:
    """
    This class stores configurations for the preprocessing data before
     the training classification models.
    Please refer to app/preprocessing.py for the data preprocessing details.
    """
    def __init__(self):
        self.n_features = 2
        self.latent_dim = 64
        self.data_path = Path.cwd().parent / 'Dataset' / 'Dataset_Split'
        self.filename = 'dataset_*.csv'
        self.use_seq_id = [0]  # [seq_id for seq_id in range(2, 5)]
        self.saved_model_path = ''
        self.model_type = 'linear'  # 'linear' or 'lstm'
        self.window_length = 200
        self.window_overlap_ratio = 0.1
        self.window_overlap_length = int(
            self.window_length * self.window_overlap_ratio)
        self.irregular_window_at_end = True  # when False means at beginning
        self.use_columns = ['x', 'valve_status', 'leakage']
        self.input_label_column_name = 'leakage'
        self.output_label_column_name = 'label'
        self.output_filename_mean_seq = 'mean_seq_*.csv'
        self.output_filename_each_window = 'each_window_*.csv'


class ClassificationConfig:
    """
    The configuration for training random forest classifier and multi-layer
    perceptron (neural network) classifier.
    The training script is app/classification.py
    """
    def __init__(self):
        self.dataset_path = Path.cwd().parent.parent / 'Data' / \
                            'dataset_classification'
        self.test_ratio = 0.2
        self.log_dir = Path.cwd() / 'Log'
        self.backup_file_list = [Path.cwd() / 'classification.py',
                                 Path.cwd() / 'helper_func' / 'config.py']
        self.print_on_screen = True

        # data precessing:
        # if category_cut_bins = [1, 2, 3] and category_cut_labels = ['A', 'B']
        # data between 1 and 2 will be labelled as A, and data between 2 and 3
        # labelled as B
        self.category_cut_bins = [-1, 0.25, 0.5, 0.75, 1]
        self.category_cut_labels = ['1', '2', '3', '4']

        # about logging models
        self.compress = 3  # please refer to joblib.dump documentation:
        # https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html
        self.input_file_extension = '.csv'

        # random forest classifier config
        self.n_estimators = 100
        self.random_forest_classifier_name = \
            'random_forest_classifier_*.joblib'

        # neural network classifier config
        self.random_state = 1
        self.max_iter = 300
        self.neural_network_classifier_name = \
            'neural_network_classifier_*.joblib'


class PreprocessAutoencoderConfig:
    """
    This class stores configurations for the preprocessing data before
     the training the autoencoder models.
    Please refer to app/preprocessing.py for the data preprocessing details.
    """
    def __init__(self):
        self.input_filename = \
            '/home/tzuchen/Master_Thesis/Data/sim_batch_003_clean.csv'
        self.position_column = 'x'
        self.velocity_column = 'velocity'
        self.use_columns = ['x', 'valve_status', 'leakage']
        self.last_velocity = 0  # see below for explanation
        # velocity is calculated by the difference of position, which means
        # the length of velocity array is the length of position array minus 1
        # In order to make velocity array the same length as the position
        # array, the value of last_velocity is appended to the velocity array

        # configuration for using function slice_by_pattern
        self.check_pattern_column = 'valve_status'
        self.pattern_end = 0
        self.pattern_start = 1
        self.remove_pattern_column = False
        self.output_filename = \
            '/home/tzuchen/Master_Thesis/Data/test/dataset_*.csv'
        # The '*' will replaced by a number. If 2 output files are generated,
        # and if output_filename is set to 'example_*.csv', the 2 output files
        # are 'example_1.csv' and 'example_2.csv'
        self.all_seq_len_output_filename = 'all_seq_len.csv'
