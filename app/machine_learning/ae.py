"""
Autoencoder Model.
An autoencoder (AE) model consist of an encoder and a decoder.
In this file, there are two encoders:
 one has only linear neural network layers(EncoderLinear),
 the other contains one LSTM layer(EncoderLSTM).
There are two decoders as well, the DecoderLinear has only linear layers,
 and the DecoderLSTM contains one LSTM layer.

The autoencoder model can be set to use linear or LSTM encoder or decoder.
The setting can be changed by modifying the "model_type" variable in the
 AEConfig in app/helper_func/config
"""
import torch
import torch.nn as nn
from app.helper_func.config import AEConfig


class EncoderLSTM(nn.Module):
    """
    The encoder of AE.
    Please set the architecture by modifying the AEConfig in
    app/helper_func/config.
    The first layer of this encoder is torch.nn.LSTM,
     other layers use torch.nn.Linear
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = AEConfig() if config is None else config

        # construct layer one
        self.layer_one = nn.LSTM(
            input_size=self.config.encoder_layer_one_input_size,
            hidden_size=self.config.encoder_layer_one_hidden_size,
            num_layers=self.config.encoder_layer_one_num_layers,
            batch_first=self.config.encoder_layer_one_batch_first
        )

        # construct layer two
        self.layer_two = nn.Linear(self.config.encoder_layer_two_in_features,
                                   self.config.encoder_layer_two_out_features)

        # construct layer three
        self.layer_three = nn.Linear(
            self.config.encoder_layer_three_in_features,
            self.config.encoder_layer_three_out_features)

        # construct layer for latent mean
        self.layer_mean = nn.Linear(self.config.encoder_layer_mean_in_features,
                                    self.config.encoder_layer_mean_out_features
                                    )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        To encode the input data to latent data.
        input_data dimension: batch_size * window_length * n_features
        return dimension: batch_size * n_features
        """
        input_layer_one = input_data.float()
        out_layer_one, (_, _) = self.layer_one(input_layer_one)
        out_layer_one = torch.tanh(out_layer_one)

        # transpose every batch
        dim_seq_len = 1
        dim_n_features = 2
        out_layer_trans = torch.transpose(out_layer_one, dim_seq_len,
                                          dim_n_features)

        out_layer_two = self.layer_two(out_layer_trans)
        out_layer_two = torch.tanh(out_layer_two)

        out_layer_three = self.layer_three(out_layer_two)

        out_squeeze = torch.squeeze(out_layer_three)
        latent_mean = self.layer_mean(out_squeeze)

        return latent_mean


class DecoderLSTM(nn.Module):
    """
    The decoder of AE.
    Please set the architecture by modifying the AEConfig in
    app/helper_func/config.
    The last layer of this decoder is torch.nn.LSTM,
     other layers use torch.nn.Linear
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = AEConfig() if config is None else config

        # construct layer one
        self.layer_one = nn.Linear(self.config.decoder_layer_one_in_features,
                                   self.config.decoder_layer_one_out_features)

        # construct layer two
        self.layer_two = nn.Linear(self.config.decoder_layer_two_in_features,
                                   self.config.decoder_layer_two_out_features)

        # construct layer three
        self.layer_three = nn.LSTM(
            input_size=self.config.decoder_layer_three_input_size,
            hidden_size=self.config.decoder_layer_three_hidden_size,
            num_layers=self.config.decoder_layer_three_num_layers,
            batch_first=self.config.decoder_layer_three_batch_first
        )

    def forward(self, out_encoder: torch.Tensor) -> torch.Tensor:
        """
        To decode the input latent data to original data.
        out_encoder dimension: batch_size * n_features
        return dimension: batch_size * window_length * n_features
        """
        # torch.unsqueeze is ud to change the dimension of out_encoder
        # from [batch_size * n_features] to [batch_size, n_features, 1]
        in_decoder = torch.unsqueeze(out_encoder, dim=-1)
        out_layer_one = self.layer_one(in_decoder.float())
        out_layer_one = torch.tanh(out_layer_one)
        out_layer_two = self.layer_two(out_layer_one)
        out_layer_two = torch.tanh(out_layer_two)

        # transpose every batch
        dim_seq_len = 2
        dim_n_features = 1
        out_layer_trans = torch.transpose(out_layer_two, dim_seq_len,
                                          dim_n_features)

        out_layer_three, (_, _) = self.layer_three(out_layer_trans)

        return out_layer_three


class EncoderLinear(nn.Module):
    """
    The encoder of AE.
    Please set the architecture by modifying the AEConfig in
    app/helper_func/config.
    All layers of this encoder use torch.nn.Linear
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = AEConfig() if config is None else config

        # construct layer one
        self.layer_one = nn.Linear(self.config.encoder_layer_one_input_size,
                                   self.config.encoder_layer_one_hidden_size)

        # construct layer two
        self.layer_two = nn.Linear(self.config.encoder_layer_two_in_features,
                                   self.config.encoder_layer_two_out_features)

        # construct layer three
        self.layer_three = nn.Linear(
            self.config.encoder_layer_three_in_features,
            self.config.encoder_layer_three_out_features)

        # construct layer for latent mean
        self.layer_mean = nn.Linear(self.config.encoder_layer_mean_in_features,
                                    self.config.encoder_layer_mean_out_features
                                    )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        To encode the input data to latent data.
        input_data dimension: batch_size * window_length * n_features
        return dimension: batch_size * n_features
        """
        input_layer_one = input_data.float()
        out_layer_one = self.layer_one(input_layer_one)
        out_layer_one = torch.tanh(out_layer_one)

        # transpose every batch
        dim_seq_len = 1
        dim_n_features = 2
        out_layer_trans = torch.transpose(out_layer_one, dim_seq_len,
                                          dim_n_features)

        out_layer_two = self.layer_two(out_layer_trans)
        out_layer_two = torch.tanh(out_layer_two)

        out_layer_three = self.layer_three(out_layer_two)

        out_squeeze = torch.squeeze(out_layer_three)
        latent_mean = self.layer_mean(out_squeeze)

        return latent_mean


class DecoderLinear(nn.Module):
    """
    The decoder of AE.
    Please set the architecture by modifying the AEConfig in
    app/helper_func/config.
    All layers of this decoder use torch.nn.Linear
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = AEConfig() if config is None else config

        # construct layer one
        self.layer_one = nn.Linear(self.config.decoder_layer_one_in_features,
                                   self.config.decoder_layer_one_out_features)

        # construct layer two
        self.layer_two = nn.Linear(self.config.decoder_layer_two_in_features,
                                   self.config.decoder_layer_two_out_features)

        # construct layer three
        self.layer_three = nn.Linear(
            self.config.decoder_layer_three_input_size,
            self.config.decoder_layer_three_hidden_size)

    def forward(self, out_encoder: torch.Tensor) -> torch.Tensor:
        """
        To decode the input latent data to original data.
        out_encoder dimension: batch_size * n_features
        return dimension: batch_size * window_length * n_features
        """
        # torch.unsqueeze is ud to change the dimension of out_encoder
        # from [batch_size * n_features] to [batch_size, n_features, 1]
        in_decoder = torch.unsqueeze(out_encoder, dim=-1)
        out_layer_one = self.layer_one(in_decoder.float())
        out_layer_one = torch.tanh(out_layer_one)
        out_layer_two = self.layer_two(out_layer_one)
        out_layer_two = torch.tanh(out_layer_two)

        # transpose every batch
        dim_seq_len = 2
        dim_n_features = 1
        out_layer_trans = torch.transpose(out_layer_two, dim_seq_len,
                                          dim_n_features)

        out_layer_three = self.layer_three(out_layer_trans)

        return out_layer_three


class AutoencoderIndividualWindow(nn.Module):
    """
    Consists of an encoder and a decoder.
    Returns latent variables of the encoder,
    and reconstructed/generated data of the decoder.
    """
    def __init__(self, ae_config=None):
        super().__init__()
        self.ae_config = AEConfig() if ae_config is None else ae_config
        if self.ae_config.model_type == 'lstm':
            self.encoder = EncoderLSTM(ae_config)
            self.decoder = DecoderLSTM(ae_config)
        elif self.ae_config.model_type == 'linear':
            self.encoder = EncoderLinear(ae_config)
            self.decoder = DecoderLinear(ae_config)

    def forward(self, input_data: torch.Tensor) -> tuple:
        """
        Propagate input data through encoder and decoder.
        """
        latent_mean = \
            self.encoder.forward(input_data)
        reconstructed_data = self.decoder.forward(
            latent_mean)

        return reconstructed_data, latent_mean
