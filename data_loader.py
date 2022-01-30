import utils
import csv
import os
import ast
import numpy as np
import pandas as pd


def SMAP_MSL_processor(dataset, path):
    # output_folder = 'processed'+'/SMAP'
    dataset_folder = path + "SMAP_MSL/"
    # print(dataset_folder)
    with open(os.path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        data_info = [row for row in res if row[1] == dataset]
    return data_info


class output:
    def __init__(
        self,
        entity_name,
        x_train,
        x_test,
        y_test,
        train_start=None,
        test_start=None,
        unit=None,
        train_timestamp=None,
        test_timestamp=None,
    ):
        self.entity_name = entity_name
        self.x_train = x_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_dim = x_train.shape[1]
        self.train_start = train_start
        self.test_start = test_start
        self.timeunit = unit
        self.train_timestamp = train_timestamp
        self.test_timestamp = test_timestamp


def dataset_choice(name, path):
    if name not in ["SMD", "MSL", "SMAP", "SWaT", "WADI"]:
        raise KeyError("you need own your function")

    if name == "SMD":
        print("This is multi-entity dataset.")

        filepath = path + "SMD/train/"
        filename = os.listdir(filepath)
        filename = [i[:-4] for i in filename]
        for dataset in filename:
            # dataset configuration
            # x_dim = utils.get_data_dim(dataset)

            # prepare the data

            prefix = path + "SMD/processed"
            (x_train, _), (x_test, y_test) = utils.get_data(
                prefix,
                dataset,
                None,
                None,
                train_start=0,
                test_start=0,
            )
            out = output(dataset, x_train, x_test, y_test, unit="1min")
            yield out

    elif name in ["MSL", "SMAP"]:
        print("This is multi-entity dataset.")
        data_info = SMAP_MSL_processor(name, path)
        dataset_folder = "SMAP_MSL"

        for row in data_info:
            # row example = ['T-3', 'SMAP', '[[2098, 2180], [5200, 5300]]', '[point, point]', '8579']
            # dataset configuration

            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros(length)
            for anomaly in anomalies:
                label[anomaly[0] : anomaly[1] + 1] = True

            x_train_name = path + dataset_folder + "/" + "train/" + row[0] + ".npy"
            x_train = np.load(x_train_name)
            x_test_name = path + dataset_folder + "/" + "test/" + row[0] + ".npy"
            x_test = np.load(x_test_name)
            y_test = label

            out = output(row[0], x_train, x_test, y_test, unit="1min")
            yield out

        return

    elif name == "SWaT":

        for name in ["SWaT"]:
            dataset_folder = "SWAT/"
            x_train = pd.read_csv(path + dataset_folder + "SWaT_Dataset_Normal_v0.csv")
            x_test = pd.read_csv(path + dataset_folder + "SWaT_Dataset_Attack_v0.csv")
            y_test = x_test[x_test.columns[-1]].values
            y_test = np.array([0 if a == "Normal" else 1 for a in list(y_test)])

            columns = list(x_train.columns[1:-1])
            x_train = x_train[columns]
            x_test = x_test[columns]

            x_train = x_train.values
            x_test = x_test.values

            out = output(
                name,
                x_train,
                x_test,
                y_test,
                train_start="2015-12-22 16:00:00",
                test_start="2015-12-28 10:00:00",
                unit="1s",
            )
            yield out

    elif name == "WADI":
        for name in ["WADI"]:
            dataset_folder = "WADI/"
            x_train = pd.read_csv(path + dataset_folder + "WADI_14days.csv")
            x_test = pd.read_csv(path + dataset_folder + "WADI_attackdata.csv")
            y_test = pd.read_csv(path + dataset_folder + "WADI_attackdataLABLE.csv")
            y_test = y_test[y_test.columns[-1]].values
            y_test = ((y_test * -1) + 1) / 2

            x_train = x_train.fillna(method="ffill")
            x_train = x_train.dropna(axis=1)
            columns = list(x_train.columns[3:])

            x_train = x_train[columns]
            x_test = x_test[columns]

            x_train = x_train.values
            x_test = x_test.values

            out = output(
                name,
                x_train,
                x_test,
                y_test,
                train_start="2017-9-25 00:00:00",
                test_start="2017-10-09 18:00:00",
                unit="1s",
            )

            yield out
