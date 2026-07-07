from ray.train import Checkpoint, session
from torch.utils.data import DataLoader
from typing import Dict, Union, List, Tuple
from string import ascii_lowercase
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import h5py
import json
import yaml
import time
import copy
import os


LABELS = [f"{letter})" for letter in ascii_lowercase]


class Dataset(torch.utils.data.Dataset):
    """object to handle the dataset"""

    def __init__(self, data):
        self.x = torch.Tensor(data[0]) if not isinstance(data[0], torch.Tensor) else data[0]
        self.y = torch.Tensor(data[1]) if not isinstance(data[1], torch.Tensor) else data[1]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = torch.Tensor(self.x[idx])
        y = torch.Tensor(self.y[idx])
        return x, y


def series_to_supervised(
    data: np.ndarray, n_in: int = 1, n_out: int = 1, dropnan: bool = True
) -> pd.DataFrame:
    """converts raw data to supervised time series format; takes time x (data + inputs)

    :param data: input data (data/coefficients + inputs)
    :type data: np.ndarray
    :param n_in: number of input time steps, defaults to 1
    :type n_in: int, optional
    :param n_out: number of output time steps, defaults to 1
    :type n_out: int, optional
    :param dropnan: drop columns with nan values, defaults to True
    :type dropnan: bool, optional
    :return: structured time series
    :rtype: pd.DataFrame
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # shift input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_training_statistics(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """takes the training data and computes the standard normalization statistics

    :param data: training dataset
    :type data: torch.Tensor
    :return: mean and standard deviation about the first axis (should be samples or time)
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    mean = data.mean(0)
    std = data.std(0)
    return mean, std


def normalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """normalizes the input data using normalization statistics

    :param data: data to be normalized
    :type data: torch.Tensor
    :param mean: mean spatial grid
    :type mean: torch.Tensor
    :param std: spatial grid standard deviation
    :type std: torch.Tensor
    :return: normalized data
    :rtype: torch.Tensor
    """
    return (data - mean) / std


def denormalize(
    data: Union[np.ndarray, torch.Tensor], stats: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """denormalizes some input data based on standard normalization statistics

    :param data: input data
    :type data: Union[np.ndarray, torch.Tensor]
    :param stats: standard normalization statistics
    :type stats: Dict[str: torch.Tensor]
    :return: de-normalized data
    :rtype: torch.Tensor
    """
    if isinstance(data, np.ndarray):
        data = torch.Tensor(data)
    return (data * stats["sigma"] + stats["mu"]).numpy()


def load_h5(filename: str, use_f10_avg: bool = False, use_ap: bool = True) -> Dict[str, np.ndarray]:
    """loads a single hdf file and returns the variables

    :param filename: name of TIEGCM coefficient file
    :type filename: str
    :param use_f10_avg: whether to use F10_41day_avg field if available, defaults to False
    :type use_f10_avg: bool, optional
    :param use_ap: whether to load Ap index instead of Kp, defaults to True
    :type use_ap: bool, optional
    :return: density array
    :rtype: Dict[str, np.ndarray]
    """
    geo_key = "Ap" if use_ap else "Kp"
    keys = ["coeff", "F10", geo_key, "doy", "utc"]
    output = {}

    with h5py.File(filename, "r") as hf:
        if use_f10_avg and "F10_41day_avg" in hf.keys():
            keys_to_load = ["coeff", "F10", "F10_41day_avg", geo_key, "doy", "utc"]
        else:
            keys_to_load = keys

        for key in keys_to_load:
            if key in hf.keys():
                output[key] = np.array(hf.get(key))

    return output


def normalize_time(array: np.ndarray, tau: int) -> Tuple[np.ndarray, np.ndarray]:
    """take an array of time information and create sinusoidal versions using a time constant (tau)

    :param array: input time array
    :type array: np.ndarray
    :param tau: time constant
    :type tau: int
    :return: sinusoidal transformations of the input array
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    output1 = np.sin(2 * np.pi * array / tau)
    output2 = np.cos(2 * np.pi * array / tau)
    return output1, output2


def combine_coefficients_and_inputs(
    coeffs: np.ndarray,
    F10: np.ndarray,
    geo_index: np.ndarray,
    doy: np.ndarray,
    utc: np.ndarray,
    F10_41day_avg: np.ndarray = None,
) -> np.ndarray:
    """take the coefficients and all inputs and combine them into a single array

    :param coeffs: latent space coefficients
    :type coeffs: np.ndarray
    :param F10: F_10.7 solar proxy
    :type F10: np.ndarray
    :param geo_index: geomagnetic activity index (Ap or Kp)
    :type geo_index: np.ndarray
    :param doy: day of year
    :type doy: np.ndarray
    :param utc: universal time (hours)
    :type utc: np.ndarray
    :param F10_41day_avg: 41-day trailing average of F10, defaults to None
    :type F10_41day_avg: np.ndarray, optional
    :return: combined coefficient and input array (n_time x (n_coeff + n_inputs))
    :rtype: np.ndarray
    """
    doy1, doy2 = normalize_time(doy, 365.25)
    utc1, utc2 = normalize_time(utc, 24)

    concat_list = [
        coeffs,
        F10.reshape(len(F10), 1),
        geo_index.reshape(len(geo_index), 1),
        utc1.reshape(len(utc1), 1),
        utc2.reshape(len(utc2), 1),
        doy1.reshape(len(doy1), 1),
        doy2.reshape(len(doy2), 1),
    ]

    if F10_41day_avg is not None:
        concat_list.insert(2, F10_41day_avg.reshape(len(F10_41day_avg), 1))

    output = np.concatenate(concat_list, axis=1)
    return output


def load_data(
    data_config: Dict[str, Union[str, List[int]]],
    test: bool,
    return_doy: bool = False,
    use_f10_avg: bool = False,
    use_ap: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """loads data from all years for training and validation

    :param data_config: data section of the config
    :type data_config: Dict[str, Union[str, List[int]]]
    :param test: whether or not you are using the test set
    :type test: bool
    :param return_doy: do you want to return the day of year as well, defaults to False
    :type return_doy: bool, optional
    :param use_f10_avg: whether to use F10_41day_avg if available, defaults to False
    :type use_f10_avg: bool, optional
    :param use_ap: whether to use Ap index instead of Kp, defaults to True
    :type use_ap: bool, optional
    :return: training and validation data
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    directory = data_config.get("directory")
    suffix = data_config.get("suffix")
    training_years = data_config.get("training_years")

    if not test:
        validation_years = data_config.get("validation_years")
    else:
        validation_years = data_config.get("test_years")

    geo_key = "Ap" if use_ap else "Kp"

    # Load training data
    for year in training_years:
        if year == training_years[0]:
            loadedT = load_h5(f"{directory}coae_coeffs_{year}{suffix}", use_f10_avg=use_f10_avg, use_ap=use_ap)
        else:
            loaded0 = load_h5(f"{directory}coae_coeffs_{year}{suffix}", use_f10_avg=use_f10_avg, use_ap=use_ap)
            for key, value in loaded0.items():
                loadedT[key] = np.concatenate((loadedT[key], loaded0[key]), axis=0)

    # Load validation data
    for year in validation_years:
        if year == validation_years[0]:
            loadedV = load_h5(f"{directory}coae_coeffs_{year}{suffix}", use_f10_avg=use_f10_avg, use_ap=use_ap)
        else:
            loaded0 = load_h5(f"{directory}coae_coeffs_{year}{suffix}", use_f10_avg=use_f10_avg, use_ap=use_ap)
            for key, value in loaded0.items():
                loadedV[key] = np.concatenate((loadedV[key], loaded0[key]), axis=0)

    if return_doy:
        return loadedT["doy"], loadedV["doy"]

    loadedT['doy'] = loadedT['doy'] + loadedT['utc'] / 24
    loadedV['doy'] = loadedV['doy'] + loadedV['utc'] / 24

    F10_avg_T = loadedT.get("F10_41day_avg", None)
    F10_avg_V = loadedV.get("F10_41day_avg", None)

    loadedT = combine_coefficients_and_inputs(
        loadedT["coeff"],
        loadedT["F10"],
        loadedT[geo_key],
        loadedT["doy"],
        loadedT["utc"],
        F10_avg_T,
    )
    loadedV = combine_coefficients_and_inputs(
        loadedV["coeff"],
        loadedV["F10"],
        loadedV[geo_key],
        loadedV["doy"],
        loadedV["utc"],
        F10_avg_V,
    )
    return loadedT, loadedV


def load_and_process_data(
    data_config: Dict[str, Union[str, List[int]]],
    test: bool = False,
    use_f10_avg: bool = False,
    use_ap: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """loads TIE-GCM data, converts to log10 space, splits, and normalizes the data

    :param data_config: data section of the config
    :type data_config: Dict[str, Union[str, List[int]]]
    :param test: whether or not you are using the test set, defaults to False
    :type test: bool, optional
    :param use_f10_avg: whether to use F10_41day_avg if available, defaults to False
    :type use_f10_avg: bool, optional
    :param use_ap: whether to use Ap index instead of Kp, defaults to True
    :type use_ap: bool, optional
    :return: normalized training and validation data
    :rtype: Dict[str, torch.Tensor]
    """
    outputT, outputV = load_data(data_config, test=test, use_f10_avg=use_f10_avg, use_ap=use_ap)
    mean, std = get_training_statistics(outputT)
    mean, std = np.array(mean), np.array(std)
    scaledT = normalize(outputT, mean, std)
    scaledV = normalize(outputV, mean, std)

    # frame as supervised learning
    reframedT = series_to_supervised(scaledT, data_config["notS"], 1)
    reframedV = series_to_supervised(scaledV, data_config["notS"], 1)

    # Adjust noI based on whether F10_41day_avg is included
    # Base: 6 inputs (F10, Kp/Ap, utc1, utc2, doy1, doy2)
    # With F10_41day_avg: 7 inputs
    noI = data_config["noI"]
    if use_f10_avg:
        noI += 1

    # drop columns we do not want to predict (last noI are the inputs for what we predict)
    reframedT.drop(reframedT.columns[-noI:], axis=1, inplace=True)
    reframedV.drop(reframedV.columns[-noI:], axis=1, inplace=True)

    # split into train and test sets
    valuesT = reframedT.values
    valuesV = reframedV.values
    n_train_hoursT = data_config["batch_size"] * int(
        np.round(valuesT.shape[0] / data_config["batch_size"])
    )
    n_train_hoursV = data_config["batch_size"] * int(
        np.round(valuesV.shape[0] / data_config["batch_size"])
    )
    train0 = valuesT[:n_train_hoursT, :]
    test0 = valuesV[:n_train_hoursV, :]

    # split train into input and outputs
    train_X, train_y = train0[:, : -data_config["nm"]], train0[:, -data_config["nm"] :]
    test_X, test_y = test0[:, : -data_config["nm"]], test0[:, -data_config["nm"] :]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape(
        int(train_X.shape[0]),
        data_config["notS"],
        int(train_X.shape[1] / float(data_config["notS"])),
    )
    test_X = test_X.reshape(
        (
            int(test_X.shape[0]),
            data_config["notS"],
            int(test_X.shape[1] / float(data_config["notS"])),
        )
    )

    # convert to tensor
    xT, xV = torch.Tensor(train_X), torch.Tensor(test_X)
    yT, yV = torch.Tensor(train_y), torch.Tensor(test_y)

    # get normalization statistics
    stats = {"mu": torch.Tensor(mean), "sigma": torch.Tensor(std)}
    if test:
        data = {"test": (xV, yV)}
    else:
        data = {"train": (xT, yT), "valid": (xV, yV)}
    output = (data, stats)
    return output


def stack_for_training(
    x: torch.Tensor, y: torch.Tensor, n: int = 125
) -> Tuple[torch.Tensor, torch.Tensor]:
    """stack training examples, so there are 125-step chunks

    :param x: input
    :type x: torch.Tensor
    :param y: output
    :type y: torch.Tensor
    :return: stacked input / output
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    num_examples = x.shape[0]
    x1 = torch.zeros([num_examples // n, n, 3, 16], dtype=torch.float32)
    for i in range(num_examples // n):
        x1[i, :, :, :] = x[i * n : (i + 1) * n, :, :]
    y1 = torch.zeros([num_examples // n, n, 10], dtype=torch.float32)
    for i in range(num_examples // n):
        y1[i, :, :] = y[i * n : (i + 1) * n, :]
    return x1, y1