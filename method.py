from torch._C import device
import modules
import utils
import eval_methods
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import copy

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")


class INRAD:
    def __init__(self, dataset, variant=True, evaluate=True, verbose=True):
        self.dataset = dataset
        self.variant = variant  # If True, INRAD-c is chosen.
        self.evaluate = evaluate
        self.verbose = verbose
        if self.variant:
            self.train_shape = None  # train set information
            self.train_representation_time = None
            self.train_representation_epoch = None
            self.train_time_per_epoch = None
        else:
            self.train_representation_time = []
            self.train_representation_epoch = []
            self.train_shape = []
            self.train_time_per_epoch = []
        self.current_process = []

        self.test_shape = []  # test set information
        self.test_representation_time = []
        self.test_representation_epoch = []

        self.performance = []  # performance information
        self.threshold = []  # performance information

    def Representation_Learning(
        self,
        hidden_dim=256,
        batch_size=131072,
        epochs=10000,
        earlystopping_patience=30,
        first_omega_0=3000,
    ):

        if self.variant == False:
            for data in self.dataset:
                torch.cuda.empty_cache()
                self.current_process.append(data.entity_name)
                if self.verbose:
                    print("current_process: ", data.entity_name)
                # on training set #######################################################################
                start = time.time()
                self.train_shape.append(data.x_train.shape)

                # if train_timestamps are unknown:
                if data.train_timestamp is None:
                    train_timestamps = modules.timestamp_maker(
                        len(data.x_train) + 1,
                        start=data.train_start,
                        unit=data.timeunit,
                    )

                # temporal encoding
                train_encoded_input = modules.temporal_encoding(train_timestamps[:-1])

                # Implicit Nerual Representation Learning on train set.
                data_train = modules.Timedata(data.x_train, train_encoded_input)
                train_dataloader = DataLoader(
                    data_train,
                    shuffle=True,
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=0,
                )

                model = modules.Siren(
                    in_features=data_train.timepoints.shape[1],
                    out_features=data.x_dim,
                    hidden_features=hidden_dim,
                    hidden_layers=3,
                    first_omega_0=first_omega_0,
                    outermost_linear=True,
                )
                model.to(device)

                optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
                early_stopping = utils.EarlyStopping(
                    patience=earlystopping_patience, verbose=False
                )

                epoch_time = []
                for step in range(epochs):
                    epoch_start = time.time()
                    model_loss = 0
                    for batch_model_input, batch_ground_truth in train_dataloader:
                        batch_model_input = batch_model_input.to(device)
                        batch_ground_truth = batch_ground_truth.to(device)

                        batch_model_output, _ = model(batch_model_input)
                        loss = F.mse_loss(batch_model_output, batch_ground_truth)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        model_loss += loss.item()
                        batch_model_input = batch_model_input.detach().cpu()
                        batch_ground_truth = batch_ground_truth.detach().cpu()
                    epoch_time.append(time.time() - epoch_start)
                    early_stopping(model_loss)
                    if early_stopping.early_stop:
                        break
                self.train_time_per_epoch.append(np.mean(epoch_time))

                time_required = copy.deepcopy(time.time() - start)

                self.train_representation_epoch.append(step)
                self.train_representation_time.append(time_required)

                # on test set (re_training) ###########################################################################

                test_start_time = time.time()
                torch.cuda.empty_cache()
                self.test_shape.append(data.x_test.shape)

                # if test_timestamps are unknown:

                if data.test_timestamp is None:
                    if data.test_start is None:
                        data.test_start = train_timestamps[-1]

                    test_timestamps = modules.timestamp_maker(
                        len(data.x_test), start=data.test_start, unit=data.timeunit
                    )
                # temporal encoding
                test_encoded_input = modules.temporal_encoding(test_timestamps)

                # Implicit Nerual Representation Learning on test set (retraining).

                data_test = modules.Timedata(data.x_test, test_encoded_input)
                test_dataloader = DataLoader(
                    data_test,
                    shuffle=True,
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=0,
                )

                early_stopping = utils.EarlyStopping(
                    patience=earlystopping_patience, verbose=False
                )

                for step in range(epochs):
                    model_loss = 0
                    for batch_model_input, batch_ground_truth in test_dataloader:
                        batch_model_input = batch_model_input.to(device)
                        batch_ground_truth = batch_ground_truth.to(device)

                        batch_model_output, _ = model(batch_model_input)
                        loss = F.mse_loss(batch_model_output, batch_ground_truth)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        model_loss += loss.item()
                        batch_model_input = batch_model_input.detach().cpu()
                        batch_ground_truth = batch_ground_truth.detach().cpu()
                    early_stopping(model_loss)
                    if early_stopping.early_stop:
                        break

                self.test_representation_epoch.append(step)

                # anomaly score calculation

                total_input = data_test.timepoints
                model = model.cpu()
                total_ground_truth = data_test.data_ready
                total_model_output, _ = model(total_input)

                anomaly_score = np.mean(
                    np.abs(
                        np.squeeze(
                            total_ground_truth.numpy()
                            - total_model_output.detach().cpu().numpy()
                        )
                    ),
                    axis=1,
                )

                time_required = copy.deepcopy(time.time() - test_start_time)
                self.test_representation_time.append(time_required)

                if self.evaluate:
                    if self.verbose:
                        print("Start evaluation in terms of Best-f1 score")

                    start = time.time()
                    m, m_t = eval_methods.bf_search(
                        anomaly_score, data.y_test
                    )  # m = performance, m_t = threshold
                    if self.verbose:
                        print("time for best_f1 score finding: ", time.time() - start)

                    self.performance.append(m)
                    self.threshold.append(m_t)

        else:
            if self.verbose:
                print("INRAD-c (using only test set")
            for data in self.dataset:
                if self.verbose:
                    print("current_process: ", data.entity_name)
                torch.cuda.empty_cache()
                self.current_process.append(data.entity_name)

                test_start_time = time.time()
                # if test_timestamps are unknown:
                if data.test_timestamp is None:
                    test_timestamps = modules.timestamp_maker(
                        len(data.x_test), start=data.test_start, unit=data.timeunit
                    )
                # temporal encoding
                test_encoded_input = modules.temporal_encoding(test_timestamps)

                data_test = modules.Timedata(data.x_test, test_encoded_input)
                test_dataloader = DataLoader(
                    data_test,
                    shuffle=True,
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=0,
                )

                model = modules.Siren(
                    in_features=data_test.timepoints.shape[1],
                    out_features=data.x_dim,
                    hidden_features=hidden_dim,
                    hidden_layers=3,
                    first_omega_0=first_omega_0,
                    outermost_linear=True,
                )
                model.to(device)

                optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
                early_stopping = utils.EarlyStopping(
                    patience=earlystopping_patience, verbose=False
                )

                for step in range(epochs):
                    model_loss = 0
                    for batch_model_input, batch_ground_truth in test_dataloader:
                        batch_model_input = batch_model_input.to(device)
                        batch_ground_truth = batch_ground_truth.to(device)

                        batch_model_output, _ = model(batch_model_input)
                        loss = F.mse_loss(batch_model_output, batch_ground_truth)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        model_loss += loss.item()
                        batch_model_input = batch_model_input.detach().cpu()
                        batch_ground_truth = batch_ground_truth.detach().cpu()
                    early_stopping(model_loss)
                    if early_stopping.early_stop:
                        break

                self.test_representation_epoch.append(step)

                # anomaly score calculation

                total_input = data_test.timepoints
                model = model.cpu()
                total_ground_truth = data_test.data_ready
                total_model_output, _ = model(total_input)

                anomaly_score = np.mean(
                    np.abs(
                        np.squeeze(
                            total_ground_truth.numpy()
                            - total_model_output.detach().cpu().numpy()
                        )
                    ),
                    axis=1,
                )

                time_required = copy.deepcopy(time.time() - test_start_time)
                self.test_representation_time.append(time_required)

                if self.evaluate:

                    if self.verbose:
                        print("Start evaluation in terms of Best-f1 score")
                    start = time.time()

                    m, m_t = eval_methods.bf_search_v2(
                        anomaly_score, data.y_test
                    )  # m = performance, m_t = threshold
                    if self.verbose:
                        print("time for best_f1 score finding: ", time.time() - start)

                    self.performance.append(m)
                    self.threshold.append(m_t)

    def evaluation(self, verbose=True):

        average_p = np.mean([val[1] for val in self.performance])
        average_r = np.mean([val[2] for val in self.performance])
        average_f1 = (average_p * average_r) * 2 / (average_p + average_r)

        if verbose:
            print("average precision: ", average_p)
            print("average recall: ", average_r)
            print("best f1: ", average_f1)
        return average_p, average_r, average_f1
