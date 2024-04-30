import os
import random
import torch
import h5py
import scipy.io
import numpy as np
from typing import List, Union
from sklearn import preprocessing
import pickle

"""
Functions list

load_matlab_data:   Load the matlab data from the .mat file.

init_model_hdf5:   Initialize the hdf5 file for the model.

init_noise_model_hdf5:  Initialize the hdf5 file for the noise model.

load_data_from_hdf5:    Load all datas for the chosen part of the training from hdf5 file.
                        Training parts are: "configs", "batch_stats", "train_stats", "test_stats"

load_data_noise_from_hdf5:  Load all datas for the chosen part of the robustess training
                            from the hdf5 files.
                            Robustness training parts are: "configs", "batch_stats", "train_stats", "test_stats"


get_seeds:  Generate a list of random seeds.

retrieve_raw_data:  Retrieve the raw data from the predictions and ground truths.
                    Inverse scaling is applied to the predictions and ground truths.

l2_similarity:  Compute the L2 distance between the predictions and the ground truths.
                The similarity is computed by comparing the L2 distance to a threshold.

batch_metric_average:   Compute the average of the metric for the batch.
                        The metric is applied to the similarity.

accuracy:   Compute the accuracy of the predictions.

"""

# ---Functions---#


def load_matlab_data(
    path_to_file: str, feature_name: str = "measures"
) -> Union[torch.Tensor, callable]:
    dataset = scipy.io.loadmat(path_to_file)
    tensor = torch.from_numpy(dataset[feature_name].T)
    scaler = preprocessing.StandardScaler().fit(tensor)
    scaler.mean_[-3:]
    scaler.scale_[-3:]
    data = torch.from_numpy(scaler.transform(tensor))
    return data, scaler


def make_splitset(invivo_data: torch.Tensor, phantom_data: torch.Tensor) -> None:
    splits = {}
    for i in range(50):
        x_start_idx = 599 + i
        x_end_idx = 699 + i
        y_start_idx = 600 + i
        y_end_idx = 700 + i

        x_train_in_vivo, x_test_in_vivo = (
            invivo_data[:599, :],
            invivo_data[x_start_idx:x_end_idx, :],
        )
        y_train_in_vivo, y_test_in_vivo = (
            invivo_data[1:600, -3:],
            invivo_data[y_start_idx:y_end_idx, -3:],
        )

        x_train_phantom, x_test_phantom = (
            phantom_data[:599, :],
            phantom_data[x_start_idx:x_end_idx:],
        )
        y_train_phantom, y_test_phantom = (
            phantom_data[1:600, -3:],
            phantom_data[y_start_idx:y_end_idx, -3:],
        )

        splits[f"{i}"] = {
            "in_vivo": {
                "x_train": x_train_in_vivo,
                "y_train": y_train_in_vivo,
                "x_test": x_test_in_vivo,
                "y_test": y_test_in_vivo,
            },
            "phantom": {
                "x_train": x_train_phantom,
                "y_train": y_train_phantom,
                "x_test": x_test_phantom,
                "y_test": y_test_phantom,
            },
        }

    with open(f"datasets_split.pickle", "wb") as handle:
        pickle.dump(splits, handle)


def init_model_hdf5(
    model_name: str,
    seeds_test: List[int],
    splitsets: dict,
    training_seed: int = 42,
) -> None:
    with h5py.File(model_name, "w") as file:
        if "configs" not in file:
            file.create_group("configs")
        if "batch_stats" not in file:
            file.create_group("batch_stats")
        if "epoch_stats" not in file:
            file.create_group("epoch_stats")
        if "test_stats" not in file:
            file.create_group("test_stats")

        train_group = file["configs"]
        train_group.create_group(str(training_seed))
        train_group[str(training_seed)].attrs["seed"] = training_seed
        train_group[str(training_seed)].create_group("in_vivo")
        train_group[str(training_seed)].create_group("phantom")

        train_group = file["batch_stats"]
        train_group.create_group(str(training_seed))
        train_group[str(training_seed)].attrs["seed"] = training_seed
        train_group[str(training_seed)].create_group("in_vivo")
        train_group[str(training_seed)].create_group("phantom")

        train_group = file["epoch_stats"]
        train_group.create_group(str(training_seed))
        train_group[str(training_seed)].attrs["seed"] = training_seed
        train_group[str(training_seed)].create_group("in_vivo")
        train_group[str(training_seed)].create_group("phantom")

        test_group = file["test_stats"]
        for SEED in seeds_test:
            test_group.create_group(str(SEED))
            test_group[str(SEED)].attrs["seed"] = SEED
            test_group[str(SEED)].create_group("in_vivo")
            test_group[str(SEED)].create_group("phantom")
            for start_idx in splitsets.keys():
                test_group[str(SEED)]["in_vivo"].create_group(start_idx)
                test_group[str(SEED)]["phantom"].create_group(start_idx)


def init_noise_model_hdf5(
    model_name: str,
    seeds_test: List[int],
    splitsets: dict,
    noises: List[float] = [0.0, 0.05, 0.2, 0.5, 1.0],
    training_seed: int = 42,
) -> None:
    with h5py.File(model_name, "w") as file:
        if "configs" not in file:
            file.create_group("configs")
        if "batch_stats" not in file:
            file.create_group("batch_stats")
        if "epoch_stats" not in file:
            file.create_group("epoch_stats")
        if "test_stats" not in file:
            file.create_group("test_stats")

        train_group = file["configs"]
        train_group.create_group(str(training_seed))
        train_group[str(training_seed)].attrs["seed"] = training_seed
        train_group[str(training_seed)].create_group("in_vivo")
        train_group[str(training_seed)].create_group("phantom")

        train_group = file["batch_stats"]
        train_group.create_group(str(training_seed))
        train_group[str(training_seed)].attrs["seed"] = training_seed
        train_group[str(training_seed)].create_group("in_vivo")
        train_group[str(training_seed)].create_group("phantom")

        train_group = file["epoch_stats"]
        train_group.create_group(str(training_seed))
        train_group[str(training_seed)].attrs["seed"] = training_seed
        train_group[str(training_seed)].create_group("in_vivo")
        train_group[str(training_seed)].create_group("phantom")

        test_group = file["test_stats"]
        for SEED in seeds_test:
            test_group.create_group(str(SEED))
            test_group[str(SEED)].attrs["seed"] = SEED
            test_group[str(SEED)].create_group("in_vivo")
            test_group[str(SEED)].create_group("phantom")
            for noise in noises:
                test_group[str(SEED)]["in_vivo"].create_group(str(noise))
                test_group[str(SEED)]["phantom"].create_group(str(noise))
                for start_idx in splitsets.keys():
                    test_group[str(SEED)]["in_vivo"][str(noise)].create_group(start_idx)
                    test_group[str(SEED)]["phantom"][str(noise)].create_group(start_idx)


def load_data_from_hdf5(hdf5_path: List[str], dataset_key: str = "test_stats") -> dict:
    data = {}
    if dataset_key == "test_stats":
        with h5py.File(hdf5_path, "r") as file:
            test_stats_group = file[dataset_key]
            for seed in test_stats_group.keys():
                data[seed] = {}
                for dataset_type in test_stats_group[seed].keys():
                    data[seed][dataset_type] = {}
                    for step in test_stats_group[seed][dataset_type].keys():
                        group = test_stats_group[seed][dataset_type][step]
                        data[seed][dataset_type][step] = {}
                        for key in group.keys():
                            data[seed][dataset_type][step][f"{key}"] = group[key][()]
    else:
        with h5py.File(hdf5_path, "r") as file:
            data["configs"] = {}
            for seed in file[dataset_key].keys():
                data["configs"][seed] = {}
                for dataset in file[dataset_key][seed].keys():
                    data["configs"][seed][dataset] = file[dataset_key][seed][dataset][
                        ()
                    ]
    return data


def load_data_noise_from_hdf5(
    hdf5_path: List[str], dataset_key: str = "test_stats"
) -> dict:
    data = {}
    if dataset_key == "test_stats":
        with h5py.File(hdf5_path, "r") as file:
            test_stats_group = file[dataset_key]
            for seed in test_stats_group.keys():
                data[seed] = {}
                for dataset_type in test_stats_group[seed].keys():
                    data[seed][dataset_type] = {}
                    for noise in test_stats_group[seed][dataset_type].keys():
                        data[seed][dataset_type][noise] = {}
                        for step in test_stats_group[seed][dataset_type][noise].keys():
                            group = test_stats_group[seed][dataset_type][noise][step]
                            data[seed][dataset_type][noise][step] = {}
                            for key in group.keys():
                                data[seed][dataset_type][noise][step][f"{key}"] = group[
                                    key
                                ][()]
    else:
        with h5py.File(hdf5_path, "r") as file:
            data["configs"] = {}
            for seed in file[dataset_key].keys():
                data["configs"][seed] = {}
                for dataset in file[dataset_key][seed].keys():
                    data["configs"][seed][dataset] = file[dataset_key][seed][dataset][
                        ()
                    ]

    return data


def get_seeds(nb_seeds: int = 20) -> List[str]:
    seeds_init = []
    for i in range(nb_seeds):
        seeds_init.append(random.randint(0, 100000))
    return seeds_init


def retrieve_raw_data(
    preds: torch.Tensor, ground_truths: torch.Tensor, scaler: callable
) -> List[np.array]:
    preds = preds.cpu()
    preds = preds.detach().numpy()
    ground_truths = ground_truths.cpu()
    ground_truths = ground_truths.detach().numpy()

    placeholder_full_dim = np.zeros((preds.shape[0], 12))
    placeholder_full_dim[:, -3:] = preds.reshape(-1, 3)
    preds = scaler.inverse_transform(placeholder_full_dim)
    preds = preds[:, -3:]

    placeholder_full_dim = np.zeros((ground_truths.shape[0], 12))
    placeholder_full_dim[:, -3:] = ground_truths.reshape(-1, 3)
    ground_truths = scaler.inverse_transform(placeholder_full_dim)
    ground_truths = ground_truths[:, -3:]

    return preds, ground_truths


def l2_similarity(
    preds: np.array, ground_truths: np.array, threshold: float = 0.02
) -> List[bool]:
    l2_distance = np.linalg.norm(preds - ground_truths, axis=1)
    similarity = l2_distance < threshold
    return similarity.tolist()


def batch_metric_average(
    metric: callable,
    l2_similarity_batch: List[bool],
) -> float:
    avg_score = 0.0
    batch_size = len(l2_similarity_batch)
    for step in l2_similarity_batch:
        avg_score += metric(step)
    return avg_score / batch_size


def accuracy(l2_similarity: List[bool]) -> float:
    return np.sum(l2_similarity) / len(l2_similarity)


def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i : i + sequence_length]
        sequences.append(sequence)
    return torch.stack(sequences)


"""
Classes list

training_info:  Store the training information.

save_models:    Save the models weights.

load_model:     Load the models with the weights.
"""
# ---classes---#


class training_info:
    def __init__(self):
        self.dict_info = {
            "epoch": [],
            "epoch_time": [],
            "total_time": [],
            "MSE_loss": [],
            "RMSE_loss": [],
        }

    def _get_times(
        self, current_time: float, previous_time: float, start_time: float
    ) -> None:
        epoch_time = current_time - previous_time
        total_time = current_time - start_time
        self.dict_info["epoch_time"].append(epoch_time)
        self.dict_info["total_time"].append(total_time)

    def __call__(self, epoch, start_time, current_time, previous_time, loss):
        self._get_times(current_time, previous_time, start_time)
        self.dict_info["epoch"].append(epoch)
        self.dict_info["MSE_loss"].append(loss)
        self.dict_info["RMSE_loss"].append(np.sqrt(loss))


class save_models:
    def __init__(self, models: List[torch.nn.Module], info: dict):
        self.models = models
        self.info = {
            "version": info["version"],
            "file_name": info["file_name"],
            "seeds": info["seeds"],
        }
        self.path = "./" + self.info["file_name"] + "/" + self.info["version"] + "/"

    def _mkdir(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _save_files(self):
        self._mkdir()
        for i, model in enumerate(self.models):
            torch.save(
                model.state_dict(), f"{self.path}model_{self.info['seeds'][i]}.pt"
            )

    def __call__(self):
        self._save_files()
        print("Models saved")


class load_model:
    def __init__(
        self,
        info: dict,
        model_type: torch.nn.Module,
        model_config: dict,
        cpu: bool = True,
    ):
        """
        info.keys() = ["version", "file_name", "model_name"]
        Exemple:
            ST-GRU ~ model_config.keys() = ["input_dim", "HIDDEN_SIZE", "NB_LAYERS",
                                            "DROPOUT_RATE", "BATCH_FIRST",]
        """
        self.path = (
            "./" + info["file_name"] + "/" + info["version"] + "/" + info["model_name"]
        )
        self.model = model_type(**model_config)
        self.cpu = cpu

    def _load_model(self):
        if self.cpu:
            self.model.load_state_dict(
                torch.load(self.path, map_location=torch.device("cpu"))
            )
        else:
            self.model.load_state_dict(torch.load(self.path))
        self.model.eval()

    def __call__(self) -> torch.nn.Module:
        print(self.path)
        self._load_model()
        return self.model
