import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import utils
import pandas as pd

# SIREN code reference link: https://github.com/vsitzmann/siren


class SineLayer(torch.nn.Module):

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=True,
        first_omega_0=3000,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations[
                    "_".join((str(layer.__class__), "%d" % activation_count))
                ] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


# Source code for our newly designed temporal encoding technique


class Timedata(torch.utils.data.Dataset):
    def __init__(self, data, encoded_input, train_scale=None):
        self.data = data
        self.scale = train_scale
        self.timepoints = encoded_input
        self.data_ready = self.data
        if self.scale is None:
            self.scale = np.max(np.abs(self.data_ready))
        self.data_ready = self.data_ready / self.scale
        self.data_ready = torch.Tensor(self.data_ready).view(
            -1, self.data.shape[1]
        )  # dimension change

    def get_num_samples(self):
        return self.timepoints.shape[0]

    def __len__(self):
        return self.timepoints.shape[0]

    def __getitem__(self, idx):
        return self.timepoints[idx, :], self.data_ready[idx, :]


def timestamp_maker(total_length, start=None, unit="1min"):
    if start == None:
        start = "1/1/2021"
    timestamp = pd.date_range(start, periods=total_length, freq=unit)
    return timestamp


def mapping(x, a, b):
    # x = array, list
    # a,b : list
    new_x = []
    for i in x:
        j = b[a.index(i)]
        new_x.append(j)
    return torch.Tensor(new_x)


def temporal_encoding(timestamp, overyear_prediction=None):

    df = pd.DataFrame(index=timestamp)

    df["y"] = timestamp.year
    df["m"] = timestamp.month
    df["d"] = timestamp.day
    # df['w'] = timestamp.weekday
    df["h"] = timestamp.hour
    df["min"] = timestamp.minute
    df["s"] = timestamp.second

    unit_needed = ["y", "m", "d", "h", "min", "s"]
    # dic = df.nunique(dropna=True)
    # unit_needed = [i for i in dic.keys() if dic[i] > 1]

    # discrete values for basic time scale with 5 kinds (year has no discrete limit.)
    m_num = 12
    d_num = 31
    h_num = 24
    min_num = 60
    s_num = 60

    if overyear_prediction is None:
        overyear_prediction = 1

    actual_values = {}
    normalized_value = {}

    for i in unit_needed:
        if i == "y":
            value = list(
                range(
                    min(df[i].values), max(df[i].values) + overyear_prediction
                )  # first_year ~ first_year + overyear_prediction
            )
            stamp = torch.linspace(
                -1, 1, steps=len(df["y"].value_counts().keys()) + overyear_prediction
            )
        elif i == "m":
            value = list(range(1, 1 + m_num))  # 1~12
            stamp = torch.linspace(-1, 1, steps=m_num)
        elif i == "d":
            value = list(range(1, 1 + d_num))  # 1~31
            stamp = torch.linspace(-1, 1, steps=d_num)
        elif i == "h":
            value = list(range(h_num))  # 0~23
            stamp = torch.linspace(-1, 1, steps=h_num)
        elif i == "min":
            value = list(range(min_num))  # 0~59
            stamp = torch.linspace(-1, 1, steps=min_num)
        elif i == "s":
            value = list(range(s_num))  # 0~59
            stamp = torch.linspace(-1, 1, steps=s_num)
        normalized_value[i] = stamp
        actual_values[i] = value
    encoded_time_input = []
    for i in unit_needed:
        encoded_time_input.append(
            mapping(df[i].values, actual_values[i], normalized_value[i])
        )
    encoded_time_input = torch.stack(encoded_time_input, dim=1)

    return encoded_time_input
