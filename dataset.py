from os.path import exists, join
import numpy as np
from typing import Union, List
import scipy
import tensorflow as tf


def convert_mat_to_dict(mat_file_path: str) -> dict:
    """
    Convert matlab .mat format to dictionary.

    Returns:
    A dictionary with format:
    {
        "data": np.ndarray([samples, time, 1, channel]),
        "us": np.ndarray([samples, 1]),
    }
    """
    print(f"Read mat file at {mat_file_path}.")

    # Read .mat file.
    data_mat = scipy.io.loadmat(mat_file_path)

    # Create empty dictionary to store data and label.
    data_dict = {}

    # The mat format is:
    # data_mat["op"][0][0][0] is real part data.
    # data_mat["op"][0][0][1] is imaginary part data.
    # data_mat["op"][0][0][2] is simulation result.
    # data_mat["op"][0][0][3] is relative permeability.
    # data_mat["op"][0][0][4] is lift-off.

    # Read input data.
    data_dict["data"] = np.concatenate(
        [
            data_mat["op"][0][0][0][:, :, None],
            data_mat["op"][0][0][1][:, :, None],
        ],
        axis=-1,
    )[
        :,
        :,
        None,
        :,
    ]

    # Read features.
    data_dict["features"] = data_mat["op"][0][0][2]

    # Read output label.
    data_dict["us"] = data_mat["op"][0][0][3].reshape(-1)
    data_dict["liftoff"] = data_mat["op"][0][0][4].reshape(-1)

    # Convert to one-hot.
    # us_labels = list(set(data_dict["us"].reshape(-1)))
    # us_labels.sort()
    # data_dict["usLabel"] = np.zeros([data_dict["data"].shape[0], len(us_labels)])
    # for i, us in enumerate(data_dict["us"]):
    #     data_dict["usLabel"][i] = np.eye(len(us_labels))[np.where(us_labels == us)]

    return data_dict


def dataPipeline(
    data_root: str,
    data_code: str,
    batch_size: int,
    z_norm_flag: bool = True,
) -> List[tf.data.Dataset]:
    """
    return [TrainDataPipeline, TestDataPipeline]
    """
    # Read dataset as dictionary.
    train_mat_path = join(data_root, data_code, "train.mat")
    train_dict = convert_mat_to_dict(train_mat_path)
    test_mat_path = join(data_root, data_code, "test.mat")
    test_dict = convert_mat_to_dict(test_mat_path)

    # Check if z-normed.
    if z_norm_flag:
        dataset_info = {}
        # Calculate mean and std.
        dataset_info["data"] = {
            "mean": np.mean(train_dict["data"], axis=(0, 1, 2)),
            "std": np.std(train_dict["data"], axis=(0, 1, 2)),
        }
        dataset_info["features"] = {
            "mean": np.mean(train_dict["features"], axis=(0,)),
            "std": np.std(train_dict["features"], axis=(0,)),
        }
        dataset_info["us"] = {
            "mean": np.mean(train_dict["us"], axis=(0,)),
            "std": np.mean(train_dict["us"], axis=(0,)),
        }
        dataset_info["liftoff"] = {
            "mean": np.mean(train_dict["liftoff"], axis=(0,)),
            "std": np.mean(train_dict["liftoff"], axis=(0,)),
        }

        print(f"Impedance is z-normed with {dataset_info['data']}")
        print(f"Features are z-normed with {dataset_info['features']}")
        print(f"us is z-normed with {dataset_info['us']}")
        print(f"liftoff is z-normed with {dataset_info['liftoff']}")

        for label_name in ["data", "features", "us", "liftoff"]:
            train_dict[label_name] = (
                train_dict[label_name] - dataset_info[label_name]["mean"]
            ) / dataset_info[label_name]["std"]
            test_dict[label_name] = (
                test_dict[label_name] - dataset_info[label_name]["mean"]
            ) / dataset_info[label_name]["std"]
    else:
        dataset_info = None
        print(f"Dataset is not z-normed.")

    train_index = np.random.choice(len(train_dict["data"]), 80000, replace=False)
    val_index = np.array(list(set(np.arange(90000)) - set(train_index)))

    for val_i in val_index:
        assert np.where(train_index == val_i)[0].shape == (0,)

    val_dict = {}
    for dict_key in train_dict.keys():
        val_dict[dict_key] = train_dict[dict_key][val_index]

    for dict_key in train_dict.keys():
        train_dict[dict_key] = train_dict[dict_key][train_index]

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_dict)
        .shuffle(len(train_dict["data"]), reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices(val_dict)
        .batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices(test_dict)
        .batch(1, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, val_dataset, test_dataset, dataset_info


if __name__ == "__main__":

    def load_data_02112022():
        data_root = join("datasets", "Samples_plate")
        data_code = "data_02112022"

        # mat_file_path = join(data_root, data_code, "train.mat")
        # convert_mat_to_dict(mat_file_path)

        train_dataset, test_dataset = dataPipeline(
            data_root,
            data_code,
            batch_size=32,
            z_norm_flag=True,
        )

        for data in train_dataset.as_numpy_iterator():
            print(data["data"].shape, data["data"].dtype)
            print(data["features"].shape, data["features"].dtype)
            print(data["us"].shape, data["us"].dtype, "\n")
            break

        for data in test_dataset.as_numpy_iterator():
            print(data["data"].shape, data["data"].dtype)
            print(data["features"].shape, data["features"].dtype)
            print(data["us"].shape, data["us"].dtype, "\n")
            break

    def load_data_23022023():
        data_root = join("datasets", "Samples_plate")
        data_code = "data_23022023"

        # mat_file_path = join(data_root, data_code, "train.mat")
        # convert_mat_to_dict(mat_file_path)

        train_dataset, test_dataset, dataset_info = dataPipeline(
            data_root,
            data_code,
            batch_size=32,
            z_norm_flag=True,
        )

        for data in train_dataset.as_numpy_iterator():
            print(data["data"].shape, data["data"].dtype)
            print(data["features"].shape, data["features"].dtype)
            print(data["us"].shape, data["us"].dtype, "\n")
            break

        for data in test_dataset.as_numpy_iterator():
            print(data["data"].shape, data["data"].dtype)
            print(data["features"].shape, data["features"].dtype)
            print(data["us"].shape, data["us"].dtype, "\n")
            break

    def load_data_21032023():
        data_root = join("datasets", "Samples_plate")
        data_code = "data_21032023"

        # mat_file_path = join(data_root, data_code, "train.mat")
        # convert_mat_to_dict(mat_file_path)

        train_dataset, val_dataset, test_dataset, dataset_info = dataPipeline(
            data_root,
            data_code,
            batch_size=32,
            z_norm_flag=True,
        )

        for data in train_dataset.as_numpy_iterator():
            print(data["data"].shape, data["data"].dtype)
            print(data["features"].shape, data["features"].dtype)
            print(data["us"].shape, data["us"].dtype, "\n")
            break

        for data in val_dataset.as_numpy_iterator():
            print(data["data"].shape, data["data"].dtype)
            print(data["features"].shape, data["features"].dtype)
            print(data["us"].shape, data["us"].dtype, "\n")
            break

        for data in test_dataset.as_numpy_iterator():
            print(data["data"].shape, data["data"].dtype)
            print(data["features"].shape, data["features"].dtype)
            print(data["us"].shape, data["us"].dtype, "\n")
            break

    load_data_21032023()
