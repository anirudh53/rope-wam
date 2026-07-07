"""
This file contains the model classes necessary for a convolutional orthogonal autoencoder (COAE).
The encoder and decoder are set up to define the entire COAE with an array of hyperparameters and 
architectures. The loss function for the COAE is a combination of mean square error (MSE) and 
ortogonal loss.
"""
import yaml
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union


class SqueezeAndExcitation(nn.Module):
    """Squeeze and excitation module"""

    def __init__(self, channels: int, ratio: int):
        """initializes squeeze and excitation module

        :param channels: input channels to the module
        :type channels: int
        :param ratio: ratio to squeeze in the linear portion of the module
        :type ratio: int
        """
        super(SqueezeAndExcitation, self).__init__()
        self._gap = torch.nn.AdaptiveAvgPool3d(1)
        self._fc1 = torch.nn.Linear(channels, channels // ratio, bias=False)
        self._fc2 = torch.nn.Linear(channels // ratio, channels, bias=False)
        self._relu = nn.ReLU(inplace=True)
        self._sigmoid = nn.Sigmoid()

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """defines the forward pass through the model

        :param x_in: input to the module
        :type x_in: torch.Tensor
        :return: output of the module
        :rtype: torch.Tensor
        """
        b, c, _, _, _ = x_in.shape
        x = self._gap(x_in)
        x = x.view(b, c)
        x = self._relu(self._fc1(x))
        x = self._sigmoid(self._fc2(x))
        x = x.view(b, c, 1, 1, 1)
        x = x_in * x
        return x


class Encoder(nn.Module):
    """Encoder section of the COAE"""

    def __init__(
        self,
        bottleneck_size: int,
        num_encoder_downsamples: int,
        num_layers_E_0: int,
        num_filters_E_0: int,
        activation_E_0: str,
        num_layers_E_1: int,
        num_filters_E_1: int,
        activation_E_1: str,
        num_layers_E_2: int,
        num_filters_E_2: int,
        activation_E_2: str,
        num_layers_E_3: int,
        num_filters_E_3: int,
        activation_E_3: str,
    ):
        """initializes Encoder object
        :param bottleneck_size: output size of the encoder model
        :type bottleneck_size: int
        :param num_encoder_downsamples: number of times the dimensions are downsampled in the encoder
        :type num_encoder_downsamples: int
        :param num_layers_E_0: number of layers before the first downsampling
        :type num_layers_E_0: int
        :param num_filters_E_0: number of filters in the first set of layers in the encoder
        :type num_filters_E_0: int
        :param activation_E_0: activation for first set of encoder layers
        :type activation_E_0: str
        :param num_layers_E_1: number of layers before the second downsampling
        :type num_layers_E_1: int
        :param num_filters_E_1: number of filters in the second set of layers in the encoder
        :type num_filters_E_1: int
        :param activation_E_1: activation for second set of encoder layers
        :type activation_E_1: str
        :param num_layers_E_2: number of layers before the third downsampling
        :type num_layers_E_2: int
        :param num_filters_E_2: number of filters in the third set of layers in the encoder
        :type num_filters_E_2: int
        :param activation_E_2: activation for third set of encoder layers
        :type activation_E_2: str
        :param num_layers_E_3: number of layers before the fourth downsampling
        :type num_layers_E_3: int
        :param num_filters_E_3: number of filters in the fourth set of layers in the encoder
        :type num_filters_E_2: int
        :param activation_E_3: activation for fourth set of encoder layers
        :type activation_E_3: str
        """
        super(Encoder, self).__init__()
        self._bottleneck_size = bottleneck_size
        # prevent data overflow from large tensors in first set of layers
        num_filters_E_0 = min(num_filters_E_0, 80)
        num_filters_E_1 = min(num_filters_E_0, 128)
        num_layers = [num_layers_E_0, num_layers_E_1, num_layers_E_2, num_layers_E_3]
        num_filters = [1, num_filters_E_0, num_filters_E_1, num_filters_E_2, num_filters_E_3][
            : num_encoder_downsamples + 1
        ]
        activations = [activation_E_0, activation_E_1, activation_E_2, activation_E_3]
        self._setup_activation_dict()
        self._build_layers(num_encoder_downsamples, num_layers, num_filters, activations)

    def _setup_activation_dict(self) -> None:
        """creates a dictionary that converts activation function names to the functional object"""
        self._activation_dict = {
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(0.1),
            "softplus": nn.Softplus(),
        }

    def _setup_pooling(self, num_encoder_downsamples: int) -> List[Tuple[int]]:
        """determines pooling depending on number of times downsampling happens in the encoder

        :param num_encoder_downsamples: number of times the dimensions are downsamples in the encoder
        :type num_encoder_downsamples: int
        :raises ValueError: if the value of num_encoder_downsamples is not 1, 2, 3, or 4
        :return: pooling information
        :rtype: List[Tuple[int]]
        """
        pools = []
        if num_encoder_downsamples == 1:
            pools.append((36, 36, 45))
        elif num_encoder_downsamples == 2:
            pools.append((9, 6, 9))
            pools.append((4, 6, 5))
        elif num_encoder_downsamples == 3:
            pools.append((4, 6, 5))
            pools.append((3, 3, 3))
            pools.append((3, 2, 3))
        elif num_encoder_downsamples == 4:
            pools.append((3, 3, 1))
            pools.append((3, 3, 1))
            pools.append((2, 2, 1))
            pools.append((2, 2, 1))
        else:
            raise ValueError(
                f"Only valid choices for `num_encoder_downsamples` are 1, 2, 3, or 4]. You input: {num_encoder_downsamples}"
            )
        return pools

    def _build_layers(
        self,
        num_encoder_downsamples: int,
        num_layers: List[int],
        num_filters: List[int],
        activations: List[str],
    ) -> None:
        """builds all layers depending on the configuration

        :param num_encoder_downsamples: number of times the dimensions are downsamples in the encoder
        :type num_encoder_downsamples: int
        :param num_layers: number of layers in each encoder section
        :type num_layers: List[int]
        :param num_filters: number of filters in each encoder section
        :type num_filters: List[int]
        :param activations: activation function for each encoder section
        :type activations: List[str]
        """
        self._layers = nn.ModuleList()
        pools = self._setup_pooling(num_encoder_downsamples)
        for i in range(num_encoder_downsamples):
            for j in range(num_layers[i]):
                self._layers.append(
                    nn.Conv3d(
                        in_channels=num_filters[i] if j == 0 else num_filters[i + 1],
                        out_channels=num_filters[i + 1],
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding="same",
                    )
                )
                #self._layers.append(self._activation_dict[activations[i].lower()])
                #self._layers.append(nn.Dropout3d(p=0.2)) #ADDED
                #self._layers.append(SqueezeAndExcitation(num_filters[i + 1], 8))
            #self._layers.append(nn.BatchNorm3d(num_filters[i + 1]))
            #self._layers.append(nn.MaxPool3d(kernel_size=pools[i]))
        #self._output_filters = num_filters[i + 1]
        self._layers.append(nn.Flatten())
        self._layers.append(nn.Linear(in_features=2 * num_filters[i + 1], out_features=self._bottleneck_size))
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """passes input through the model

        :param x: input tensor
        :type x: torch.Tensor
        :return: latent output mean and log variance
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        for layer in self._layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """Decoder section of the COAE"""

    def __init__(
        self,
        bottleneck_size: int,
        num_decoder_upsamples: int,
        num_layers_D_0: int,
        num_filters_D_0: int,
        activation_D_0: str,
        num_layers_D_1: int,
        num_filters_D_1: int,
        activation_D_1: str,
        num_layers_D_2: int,
        num_filters_D_2: int,
        activation_D_2: str,
        num_layers_D_3: int,
        num_filters_D_3: int,
        activation_D_3: str,
    ):
        """initializes Decoder object
        :param bottleneck_size: input size of the decoder model
        :type bottleneck_size: int
        :param num_decoder_upsamples: number of times the dimensions are upsampled in the decoder
        :type num_decoder_upsamples: int
        :param num_layers_D_0: number of layers before the first upsampling
        :type num_layers_D_0: int
        :param num_filters_D_0: number of filters in the first set of layers in the decoder
        :type num_filters_D_0: int
        :param activation_D_0: activation for first set of decoder layers
        :type activation_D_0: str
        :param num_layers_D_1: number of layers before the second upsampling
        :type num_layers_D_1: int
        :param num_filters_D_1: number of filters in the second set of layers in the decoder
        :type num_filters_D_1: int
        :param activation_D_1: activation for second set of decoder layers
        :type activation_D_1: str
        :param num_layers_D_2: number of layers before the third upsampling
        :type num_layers_D_2: int
        :param num_filters_D_2: number of filters in the third set of layers in the decoder
        :type num_filters_D_2: int
        :param activation_D_2: activation for third set of decoder layers
        :type activation_D_2: str
        :param num_layers_D_3: number of layers before the fourth upsampling
        :type num_layers_D_3: int
        :param num_filters_D_3: number of filters in the fourth set of layers in the decoder
        :type num_filters_D_3: int
        :param activation_D_3: activation for fourth set of decoder layers
        :type activation_D_3: str
        """
        super(Decoder, self).__init__()
        self._bottleneck_size = bottleneck_size
        num_layers = [num_layers_D_0, num_layers_D_1, num_layers_D_2, num_layers_D_3]
        num_filters = self._process_filters(
            num_decoder_upsamples,
            num_filters_D_0,
            num_filters_D_1,
            num_filters_D_2,
            num_filters_D_3,
        )
        activations = [activation_D_0, activation_D_1, activation_D_2, activation_D_3]
        self._setup_activation_dict()
        self._build_layers(num_decoder_upsamples, num_layers, num_filters, activations)

    def _process_filters(
        self,
        num_decoder_upsamples: int,
        num_filters_D_0: int,
        num_filters_D_1: int,
        num_filters_D_2: int,
        num_filters_D_3: int,
    ) -> List[int]:
        """compiles list of filters for each set of layers to make building model easy

        :param num_decoder_upsamples: number of times the dimensions are upsampled in the decoder
        :type num_decoder_upsamples: int
        :param num_filters_D_0: number of filters in the first set of layers in the decoder
        :type num_filters_D_0: int
        :param num_filters_D_1: number of filters in the second set of layers in the decoder
        :type num_filters_D_1: int
        :param num_filters_D_2: number of filters in the third set of layers in the decoder
        :type num_filters_D_2: int
        :param num_filters_D_3: number of filters in the fourth set of layers in the decoder
        :type num_filters_D_3: int
        :return: list of number of filters in each set of decoder layers with a 1 appended to the end (output)
        :rtype: List[int]
        """
        # what the num_filters list would look like depending on the number of sets of layers
        filters_dict = {
            1: [num_filters_D_0, 1],
            2: [num_filters_D_0, min(num_filters_D_1, 80), 1],
            3: [num_filters_D_0, min(num_filters_D_1, 128), min(num_filters_D_2, 80), 1],
            4: [num_filters_D_0, num_filters_D_1, min(num_filters_D_2, 128), min(num_filters_D_3, 80), 1],
        }
        # get num_filters list corresponding to config
        num_filters = filters_dict[num_decoder_upsamples]
        return num_filters

    def _setup_activation_dict(self) -> None:
        """creates a dictionary that converts activation function names to the functional object"""
        self._activation_dict = {
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(0.1),
            "softplus": nn.Softplus(),
        }

    def _setup_upsampling(self, num_decoder_upsamples: int) -> List[Tuple[int]]:
        """determines upsampling depending on number of times upsampling happens in the decoder

        :param num_encoder_downsamples: number of times the dimensions are downsampled in the encoder
        :type num_encoder_downsamples: int
        :raises ValueError: if the value of num_encoder_downsamples is not 1, 2, 3, or 4
        :return: pooling information
        :rtype: List[Tuple[int]]
        """
        unpools = []
        if num_decoder_upsamples == 1:
            unpools.append((36, 36, 45))
        elif num_decoder_upsamples == 2:
            unpools.append((4, 6, 5))
            unpools.append((9, 6, 9))
        elif num_decoder_upsamples == 3:
            unpools.append((3, 2, 3))
            unpools.append((3, 3, 3))
            unpools.append((4, 6, 5))
        elif num_decoder_upsamples == 4:
            unpools.append((2, 2, 1))
            unpools.append((2, 2, 1))
            unpools.append((3, 3, 1))
            unpools.append((3, 3, 1))
        else:
            raise ValueError(
                f"Only valid choices for `num_decoder_upsamples` are 1, 2, 3, or 4]. You input: {num_decoder_upsamples}"
            )
        return unpools

    def _build_layers(
        self,
        num_decoder_upsamples: int,
        num_layers: List[int],
        num_filters: List[int],
        activations: List[str],
    ) -> None:
        """builds all layers depending on the configuration

        :param num_decoder_upsamples: number of times the dimensions are downsampled in the encoder
        :type num_decoder_upsamples: int
        :param num_layers: number of layers in each decoder section
        :type num_layers: List[int]
        :param num_filters: number of filters in each decoder section
        :type num_filters: List[int]
        :param activations: activation function for each decoder section
        :type activations: List[str]
        """
        self._initial_layers = nn.ModuleList(
            [
                nn.Linear(in_features=self._bottleneck_size, out_features=2 * num_filters[0]),
                #self._activation_dict["leakyrelu"],
            ]
        )
        self._first_filter = num_filters[0]
        self._layers = nn.ModuleList()
        unpools = self._setup_upsampling(num_decoder_upsamples)
        for i in range(num_decoder_upsamples):
            self._layers.append(nn.Upsample(scale_factor=unpools[i]))
            if (i + 1) != num_decoder_upsamples:
                for j in range(num_layers[i]):
                    self._layers.append(
                        nn.Conv3d(
                            in_channels=num_filters[i] if j == 0 else num_filters[i + 1],
                            out_channels=num_filters[i + 1],
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding="same",
                        )
                    )
                    #self._layers.append(self._activation_dict[activations[i].lower()])
                    #self._layers.append(SqueezeAndExcitation(num_filters[i + 1], 4))
                self._layers.append(nn.BatchNorm3d(num_filters[i + 1]))
        self._layers.append(
            nn.Conv3d(
                in_channels=num_filters[-2],
                out_channels=num_filters[-1],
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding="same",
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """passes input through the model
        :param x: input tensor
        :type x: torch.Tensor
        :return: output tensor
        :rtype: torch.Tensor
        """
        for layer in self._initial_layers:
            x = layer(x)
        # x = x.view(x.size(0), self._first_filter, 2, 1, 1)
        x = x.view(-1, self._first_filter, 2, 1, 1)

        for layer in self._layers:
            x = layer(x)
        return x
        

class COAE2(nn.Module):
    """Convolutional Orthogonal Autoencoder that builds itself based on hyperparameters from a config file"""

    def __init__(self, config: dict):
        """initializes COAE object

        :param config: config for the COAE hyperparameters
        :type config: dict
        """
        super(COAE, self).__init__()
        # save necessary attributes from function inputs
        self._bottleneck_size = config.get("bottleneck_size", 10)
        self._alpha = config.get("alpha", 1.0)
        # build encoder
        self.encoder = Encoder(
            bottleneck_size=self._bottleneck_size,
            num_encoder_downsamples=config.get("num_encoder_downsamples", 4),
            num_layers_E_0=config.get("num_layers_E_0", 1),
            num_layers_E_1=config.get("num_layers_E_1", 1),
            num_layers_E_2=config.get("num_layers_E_2", 1),
            num_layers_E_3=config.get("num_layers_E_3", 1),
            num_filters_E_0=config.get("num_filters_E_0", 16),
            num_filters_E_1=config.get("num_filters_E_1", 32),
            num_filters_E_2=config.get("num_filters_E_2", 64),
            num_filters_E_3=config.get("num_filters_E_3", 128),
            activation_E_0=config.get("activation_E_0", "relu"),
            activation_E_1=config.get("activation_E_1", "relu"),
            activation_E_2=config.get("activation_E_2", "relu"),
            activation_E_3=config.get("activation_E_3", "relu"),
        )
        # build decoder
        self.decoder = Decoder(
            bottleneck_size=self._bottleneck_size,
            num_decoder_upsamples=config.get("num_decoder_upsamples", 4),
            num_layers_D_0=config.get("num_layers_D_0", 1),
            num_layers_D_1=config.get("num_layers_D_1", 1),
            num_layers_D_2=config.get("num_layers_D_2", 1),
            num_layers_D_3=config.get("num_layers_D_3", 1),
            num_filters_D_0=config.get("num_filters_D_0", 128),
            num_filters_D_1=config.get("num_filters_D_1", 64),
            num_filters_D_2=config.get("num_filters_D_2", 32),
            num_filters_D_3=config.get("num_filters_D_3", 16),
            activation_D_0=config.get("activation_D_0", "relu"),
            activation_D_1=config.get("activation_D_1", "relu"),
            activation_D_2=config.get("activation_D_2", "relu"),
            activation_D_3=config.get("activation_D_3", "relu"),
        )
        # self.encoder = torch.nn.parallel.DataParallel(self.encoder)
        # self.decoder = torch.nn.parallel.DataParallel(self.decoder)

    def _load_config(self, filename: str) -> Dict[str, Union[str, int, float]]:
        """loads config file and returns dictionary

        :param filename: path to config file
        :type filename: str
        :return: configuration dictionary
        :rtype: Dict[str, Union[str, int, float]]
        """
        with open(filename, "r") as f:
            config = yaml.safe_load(f)
        return config.get("model")

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """uses the encoder to predict the latent distribution

        :param x: model / encoder input
        :type x: torch.Tensor
        :return: latent space
        :rtype: Tuple[torch.Tensor]
        """
        return self.encoder(x)

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        """uses the decoder to predict the full-state from a reparameterized latent distribution

        :param x: decoder input
        :type x: torch.Tensor
        :return: prediction of the full-space
        :rtype: torch.Tensor
        """
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """runs an input through the model

        :param x: input
        :type x: torch.Tensor
        :return: prediction, latent mean, and latent log-variance
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        z = self._encode(x)
        xhat = self._decode(z)
        return xhat, z
import numpy as np
class COAEE(nn.Module):
    def __init__(self, config:dict, input_shape=(1, 72, 36,45), latent_dim=10):
        """
        Strictly linear convolutional autoencoder (no BN, no activations, no pooling).
        input_shape: (C, H, W, D)
        latent_dim: size of bottleneck
        """
        super(COAE, self).__init__()
        self.input_shape = input_shape
        C, H, W, D = input_shape

        # ----- Encoder (convs only, no BN/activation/pooling) -----
        self.encoder_convs = nn.Sequential(
            nn.Conv3d(C, 16, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm3d(16),
            nn.LeakyReLU(0.01, inplace=True),
            #ResBlock3D(16),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm3d(32),
            nn.LeakyReLU(0.01, inplace=True),
            #ResBlock3D(32),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm3d(64),
            nn.LeakyReLU(0.01, inplace=True),
            #ResBlock3D(64),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm3d(128),
            nn.LeakyReLU(0.01, inplace=True),
            #ResBlock3D(128),
        )

        # compute flattened size after encoder convs
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            enc_out = self.encoder_convs(dummy)
            self.flat_dim = int(np.prod(enc_out.shape[1:]))

        # bottleneck fully-connected (linear projection)
        self.fc_enc = nn.Linear(self.flat_dim, latent_dim, bias=True)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim, bias=True)

        # ----- Decoder (mirror convs) -----
        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm3d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm3d(32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm3d(16),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose3d(16, C, kernel_size=3, stride=1, padding=1, bias=True),
        )
        #self._mirror_decoder_weights()

    def _load_config(self, filename: str) -> Dict[str, Union[str, int, float]]:
        """loads config file and returns dictionary

        :param filename: path to config file
        :type filename: str
        :return: configuration dictionary
        :rtype: Dict[str, Union[str, int, float]]
        """
        with open(filename, "r") as f:
            config = yaml.safe_load(f)
        return config.get("model")

    def forward(self, x):
        B = x.size(0)

        # encoder convs
        enc = self.encoder_convs(x)

        # flatten and bottleneck
        enc_flat = enc.view(B, -1)              # (B, flat_dim)
        z = self.fc_enc(enc_flat)               # (B, latent_dim)
        dec_flat = self.fc_dec(z)               # (B, flat_dim)
        dec_unflat = dec_flat.view(enc.shape)   # reshape back to conv feature map

        # decoder convs
        xrec = self.decoder_convs(dec_unflat)

        return xrec, z

    def _mirror_decoder_weights(self):
        """
        Initialize decoder ConvTranspose3d weights as transposed (mirrored)
        versions of encoder Conv3d weights.
        """
        enc_layers = [m for m in self.encoder_convs if isinstance(m, nn.Conv3d)]
        dec_layers = [m for m in self.decoder_convs if isinstance(m, nn.ConvTranspose3d)]

        # Match reversed encoder order to decoder order
        for enc, dec in zip(reversed(enc_layers), dec_layers):
            try:
                # Direct copy since shapes match
                dec.weight.data.copy_(enc.weight.data)
                dec.bias.data.zero_()
            except Exception as e:
                print(f"⚠️ Could not mirror layer: {e}")
class SpatialAttention3D(nn.Module):
    """
    Generic 3D Spatial Attention.
    Works with any number of channels.
    Computes attention map from avg and max pooled feature maps (along channel axis).
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W, D)
        avg_out = torch.mean(x, dim=1, keepdim=True)           # (B, 1, H, W, D)
        max_out, _ = torch.max(x, dim=1, keepdim=True)         # (B, 1, H, W, D)
        attn = torch.cat([avg_out, max_out], dim=1)            # (B, 2, H, W, D)
        attn = self.sigmoid(self.conv(attn))                   # (B, 1, H, W, D)
        return x * attn   
class ResBlock3D(nn.Module):
    def __init__(self, channels, slope=0.01):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=True)
        self.act = nn.LeakyReLU(slope, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.act(out + x)  # residual add

class Encoder3D(nn.Module):
    def __init__(self, in_channels, latent_dim, input_shape):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            enc = self.convs(dummy)
            self.flat_dim = int(np.prod(enc.shape[1:]))

        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.convs(x)
        h = h.view(x.size(0), -1)
        z = self.fc(h)
        return z

class EncoderLowSimple(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,3,2), padding=(1,1,0)),
            nn.LeakyReLU(0.01),

            nn.Conv3d(32, 64, kernel_size=(3,3,1), padding=(1,1,0)),
            nn.LeakyReLU(0.01),

            nn.Conv3d(64, 128, kernel_size=1),
            nn.LeakyReLU(0.01),
        )

        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = self.convs(x)
        h = self.pool(h).view(x.size(0), -1)
        return self.fc(h)
        
class Decoder3D(nn.Module):
    def __init__(self, latent_dim, enc_shape, out_channels=1):
        super().__init__()

        self.fc = nn.Linear(latent_dim, int(np.prod(enc_shape)))
        self.enc_shape = enc_shape

        self.convs = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose3d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose3d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose3d(16, out_channels, 3, padding=1),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), *self.enc_shape)
        return self.convs(h)

class COAE(nn.Module):
    def __init__(self, config:dict, input_shape=(1, 72, 36, 45), latent_dim=10):
        super().__init__()

        C, H, W, D = input_shape

        # ----- Encoder A (lower altitudes) -----
        self.encoder_low = Encoder3D(
            in_channels=C,
            latent_dim=latent_dim,
            input_shape=(C, H, W, 2)
        )

        # ----- Encoder B (higher altitudes) -----
        self.encoder_high = Encoder3D(
            in_channels=C,
            latent_dim=latent_dim,
            input_shape=(C, H, W, D - 2)
        )

        # ----- Fusion -----
        self.fuse = nn.Linear(2 * latent_dim, latent_dim)

        # ----- Decoder -----
        # use shape from one encoder conv output
        #with torch.no_grad():
        #    dummy = torch.zeros(1, C, H, W, 45)
        #    enc_dummy = self.encoder_low.convs(dummy)
        #    enc_shape = enc_dummy.shape[1:]
        enc_shape = (128, 72, 36, 45)


        self.decoder = Decoder3D(
            latent_dim=latent_dim,
            enc_shape=enc_shape,
            out_channels=C
        )

    def forward(self, x):
        x_low = x[..., :2]
        x_high = x[..., 2:]

        z_low = self.encoder_low(x_low)     # (B,10)
        z_high = self.encoder_high(x_high)  # (B,10)

        z_cat = torch.cat([z_low, z_high], dim=1)
        z = self.fuse(z_cat)                # (B,10)

        xrec = self.decoder(z)

        return xrec, z#, z_low, z_high