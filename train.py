import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import os
from os.path import exists, join
from typing import Union, List, NamedTuple, Callable, Tuple
import numpy as np
import haiku as hk
import multiprocessing
import mlflow
from urllib.parse import unquote, urlparse
import time
import glob
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import graphviz
import json
import copy


import dataset
from simpleCNN import SimpleCNN2D
import utils
import modified_resnet


class expState(NamedTuple):
    """
    diff_model_state = {params: hk.Params}
    non_diff_state = {state: hk.State
                      optState: optax.OptState}
    """

    diff: dict
    non_diff: dict


def define_forward(hparams: dict, experiment: int) -> Callable:
    multi_head_instruction = hparams["multi_head_instruction"][experiment]
    n_last_second_logit = hparams["n_last_second_logit"][experiment]
    model_name = hparams["model_name"][experiment]

    model_book = {
        "resnet18": modified_resnet.ResNet18,
        "resnet34": modified_resnet.ResNet34,
        "resnet50": modified_resnet.ResNet50,
        "resnet101": modified_resnet.ResNet101,
        "simpleCNN": SimpleCNN2D,
        "mlp": hk.nets.MLP,
    }

    def one_head(
        multi_head_outputs: dict, x: np.ndarray, n_logit: int, head_type: str
    ) -> None:
        classifier = hk.Linear(n_logit, name=head_type.replace("-", ""))
        multi_head_outputs[head_type + "Pred"] = classifier(x)

    def multi_head(x_in: np.ndarray) -> dict:
        x = jax.nn.relu(x_in)
        multi_head_outputs = {}
        for head_key in multi_head_instruction.keys():
            one_head(multi_head_outputs, x, multi_head_instruction[head_key], head_key)
        return multi_head_outputs

    def _forward(x: np.ndarray, features: np.ndarray, is_training: bool) -> jnp.ndarray:
        # Define forward-pass.
        if "resnet" in model_name:
            module = model_book[model_name](
                num_classes=n_last_second_logit,
                resnet_v2=hparams["resnet_v2"][experiment],
            )
            x = module(x, is_training)
            x = jnp.concatenate([x, features], axis=-1)
            x = multi_head(x)
            return x
        elif "simpleCNN" in model_name:
            module = model_book[model_name](
                output_size=n_last_second_logit,
                output_channels_list=hparams["simpleCNN_list"][experiment],
                kernel_shape=(3, 1),
                stride=hparams["simpleCNN_stride"][experiment],
                bn_decay_rate=0.9,
                activation_fn=jax.nn.relu,
                bn_flag=True,
                dropoutRate=None,
            )
            x = module(x, is_training)
            x = jnp.concatenate([x, features], axis=-1)
            x = multi_head(x)
            return x
        elif "mlp" in model_name:
            flat = hk.Flatten()
            module = model_book[model_name](
                hparams["mlp_list"][experiment] + [n_last_second_logit]
            )
            x = flat(x)
            x = module(x)
            x = jnp.concatenate([x, features], axis=-1)
            x = multi_head(x)
            return x
        else:
            print(f"model_name should be in {model_book.keys()}, but get {model_name}")
            assert False

    return _forward


def define_loss_fn(
    forward: hk.Transformed,
    is_training: bool,
    optax_loss: Callable,
    hparams: dict,
    experiment: int,
) -> Callable:
    @jax.jit
    def loss_fn(
        params: hk.Params, state: hk.State, data_dict: dict
    ) -> Tuple[jnp.ndarray, Tuple[hk.State, jnp.ndarray]]:
        # Forward-pass.
        if is_training:
            # Update state.
            y_pred, state = forward.apply(
                params, state, data_dict["data"], data_dict["features"], is_training
            )
        else:
            # Do not update state.
            y_pred, _ = forward.apply(
                params, state, data_dict["data"], data_dict["features"], is_training
            )

        # Calculate loss. loss = target_loss + a * weight_decay.
        loss = 0.0
        n_head = len(hparams["multi_head_instruction"][experiment])
        for head_name in hparams["multi_head_instruction"][experiment].keys():
            pred_head_key = head_name + "Pred"
            # Calculate mean loss.
            if "regression" in hparams["problem"][experiment]:
                # If regression, true label is y_pred[pred_head_key].
                y_pred[pred_head_key] = y_pred[pred_head_key].reshape(-1)
                loss += optax_loss(y_pred[pred_head_key], data_dict[head_name]).mean()
            elif "classification" in hparams["problem"][experiment]:
                # If classification, true label is data_dict[head_name + "Label"]
                loss += optax_loss(
                    y_pred[pred_head_key], data_dict[head_name + "Label"]
                ).mean()

        # Average multi-heads loss.
        loss = loss / n_head

        # Add weight decay.
        if hparams["weight_decay"][experiment] is not None:
            decayLoss = hparams["weight_decay"][experiment] * utils.weightDecay(params)
            loss += decayLoss
        return loss, (state, y_pred)

    return loss_fn


def define_train_step(
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    optimizer_schedule: Callable,
) -> Callable:
    @jax.jit
    def train_step(
        train_exp_state: expState, data_dict: dict
    ) -> Tuple[expState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Forward-pass and backward-pass.
        (
            (loss, (train_exp_state.non_diff["state"], y_pred)),
            grads_dict,
        ) = jax.value_and_grad(loss_fn, has_aux=True)(
            train_exp_state.diff["params"], train_exp_state.non_diff["state"], data_dict
        )

        # Record inner learning rate.
        record_lr = optimizer_schedule(train_exp_state.non_diff["opt_state"][0].count)

        # Calculate gradient_update and update opt_state.
        updates, train_exp_state.non_diff["opt_state"] = optimizer.update(
            grads_dict, train_exp_state.non_diff["opt_state"]
        )

        # update params.
        train_exp_state.diff["params"] = optax.apply_updates(
            train_exp_state.diff["params"], updates
        )
        return train_exp_state, loss, y_pred, record_lr, grads_dict

    return train_step


def discard_str_keys(record, data_dict):
    """
    Discard str keys in data_dict and save fileName as identifier.
    """
    for data_key in list(data_dict.keys()):
        if data_dict[data_key].dtype == object:
            if data_key == "fileName":
                if data_key not in record.keys():
                    record[data_key] = []
                temp_data = data_dict[data_key].copy().tolist()
                record[data_key].append([d.decode() for d in temp_data])
            del data_dict[data_key]


def define_train(train_step: Callable, hparams: dict, experiment: int) -> Callable:
    def train(
        train_exp_state: expState,
        dataset: tf.data.Dataset,
    ) -> Tuple[expState, dict]:
        record = {"loss": [], "lr": [], "grads_norm": [], "y_pred": []}
        for head_name in hparams["multi_head_instruction"][experiment].keys():
            if "classification" == hparams["problem"][experiment]:
                record_name = head_name + "Label"
            elif "regression" == hparams["problem"][experiment]:
                record_name = head_name
            record[record_name] = []

        for data_dict in dataset.as_numpy_iterator():
            discard_str_keys(record, data_dict)
            # Update train_exp_state.
            train_exp_state, loss, y_pred, lr, grads_dict = train_step(
                train_exp_state, data_dict
            )
            record["loss"].append(loss.tolist())
            record["y_pred"].append({k: y_pred[k].tolist() for k in y_pred.keys()})
            record["lr"].append(lr.tolist())
            record["grads_norm"].append(utils.calculate_norm(grads_dict).tolist())
            for head_name in hparams["multi_head_instruction"][experiment].keys():
                if "classification" == hparams["problem"][experiment]:
                    record_name = head_name + "Label"
                elif "regression" == hparams["problem"][experiment]:
                    record_name = head_name
                record[record_name].append(data_dict[record_name].tolist())

        record["loss"] = jnp.mean(jnp.array(record["loss"])).tolist()
        return train_exp_state, record

    return train


def define_test(lossFn: Callable, hparams: dict, experiment: int) -> Callable:
    def test(text_exp_state: expState, dataset: tf.data.Dataset) -> dict:
        record = {"loss": [], "y_pred": []}
        for head_name in hparams["multi_head_instruction"][experiment].keys():
            if "classification" == hparams["problem"][experiment]:
                record_name = head_name + "Label"
            elif "regression" == hparams["problem"][experiment]:
                record_name = head_name
            record[record_name] = []

        for data_dict in dataset.as_numpy_iterator():
            discard_str_keys(record, data_dict)
            # Do not update state.
            loss, (_, y_pred) = lossFn(
                text_exp_state.diff["params"],
                text_exp_state.non_diff["state"],
                data_dict,
            )
            record["loss"].append(loss.tolist())
            record["y_pred"].append({k: y_pred[k].tolist() for k in y_pred.keys()})
            for head_name in hparams["multi_head_instruction"][experiment].keys():
                if "classification" == hparams["problem"][experiment]:
                    record_name = head_name + "Label"
                elif "regression" == hparams["problem"][experiment]:
                    record_name = head_name
                record[record_name].append(data_dict[record_name].tolist())

        record["loss"] = jnp.mean(jnp.array(record["loss"])).tolist()
        return record

    return test


def define_forward_and_optimizer(hparams: dict, experiment: int, DATA_SIZE: Tuple):
    # Define _forward.
    _forward = define_forward(hparams, experiment)

    nData = 80000

    # Define optimizer and learning rate schedule.
    optimizerSchedule = utils.lr_schedule(
        hparams["lr"][experiment],
        hparams["lr_schedule_flag"][experiment],
        int(nData * hparams["epoch"][experiment] / hparams["batch_size"][experiment]),
    )
    optimizer = utils.optimizerSelector(hparams["optimizer"][experiment])(
        learning_rate=optimizerSchedule
    )

    def summary_model():
        """
        Summary model.
        """

        def temp_forward(x, features):
            _forward(x, features, True)

        # Summary model.
        dummy_x = np.random.uniform(
            size=(
                hparams["batch_size"][experiment],
                DATA_SIZE[0],
                DATA_SIZE[1],
                DATA_SIZE[2],
            )
        ).astype(np.float32)

        dummy_features = np.random.uniform(
            size=(hparams["batch_size"][experiment], 3)
        ).astype(np.float32)

        summary_message = (
            f"{hk.experimental.tabulate(temp_forward)(dummy_x, dummy_features)}"
        )
        return summary_message

    summary_message = summary_model()

    return (_forward, optimizer, optimizerSchedule, summary_message)


def initialize_train_exp_state(DATA_SIZE, forward, optimizer, mlflow_artifact_path):
    # Initialize the parameters and states of the network and return them.
    dummy_x = np.random.uniform(
        size=(1, DATA_SIZE[0], DATA_SIZE[1], DATA_SIZE[2])
    ).astype(np.float32)

    dummy_features = np.random.uniform(size=(1, 3)).astype(np.float32)

    params, state = forward.init(
        rng=jax.random.PRNGKey(42), x=dummy_x, features=dummy_features, is_training=True
    )

    # Visualize model.
    dot = hk.experimental.to_dot(forward.apply)(
        params, state, dummy_x, dummy_features, True
    )
    dot_plot = graphviz.Source(dot)
    dot_plot.source.replace("rankdir = TD", "rankdir = TB")
    dot_plot_save_path = join(mlflow_artifact_path, "summary")
    dot_plot.render(filename="model_plot", directory=dot_plot_save_path)

    # Initialize model and optimiser.
    opt_state = optimizer.init(params)

    # Initialize train state.
    train_exp_state = expState(
        {"params": params}, {"state": state, "opt_state": opt_state}
    )
    return train_exp_state


def save_exp_state(exp_state, epoch, mlflow_artifact_path):
    saving_history = {
        int(os.path.normpath(p).split(os.sep)[-1][5:]): p
        for p in glob.glob(join(mlflow_artifact_path, "Epoch*"))
    }
    # Only keep the latest N models.
    if len(saving_history) >= 5:
        shutil.rmtree(saving_history[min(saving_history.keys())])
        print("Delete save", min(saving_history.keys()))

    save_ckpt_dir = join(mlflow_artifact_path, "Epoch" + str(epoch))
    utils.save_data(save_ckpt_dir, exp_state.diff["params"], "params")
    utils.save_data(save_ckpt_dir, exp_state.non_diff["state"], "state")
    utils.save_data(save_ckpt_dir, exp_state.non_diff["opt_state"], "opt_state")


def restore_exp_state(starting_epoch, mlflow_artifact_path):
    restore_ckpt_dir = join(mlflow_artifact_path, "Epoch" + str(starting_epoch))
    print(f"Restore from {restore_ckpt_dir}")

    params = utils.restore(restore_ckpt_dir, "params")
    state = utils.restore(restore_ckpt_dir, "state")
    opt_state = utils.restore(restore_ckpt_dir, "opt_state")

    exp_state = expState({"params": params}, {"state": state, "opt_state": opt_state})
    return exp_state


def plot_confusion_matrix(
    train_result: dict,
    val_result: dict,
    test_result: dict,
    train_dataset_dict: dict,
    hparams: dict,
    experiment: int,
    starting_epoch: int,
    mlflowArtifactPath: str,
    transparent: bool = False,
):
    def total_mean_acc(confusionMatrix):
        cmShape = confusionMatrix.shape
        cnt = 0
        for ii in range(cmShape[0]):
            cnt += confusionMatrix[ii][ii]
        return cnt / np.sum(confusionMatrix)

    labelCodeBook = {}
    for head_name in hparams["multi_head_instruction"][experiment].keys():
        temp_list = list(
            set(
                train_dataset_dict[data_key]["metaDataAndLabel"][head_name]
                for data_key in train_dataset_dict.keys()
            )
        )
        temp_list.sort()
        labelCodeBook[head_name] = temp_list

    fig = plt.figure(
        figsize=(4 + 4 * len(hparams["multi_head_instruction"][experiment]), 12)
    )
    gs = plt.GridSpec(
        3, 1 + len(hparams["multi_head_instruction"][experiment]), figure=fig
    )

    for nR, (name, result) in enumerate(
        zip(["Train", "validation", "Test"], [train_result, val_result, test_result])
    ):
        text_info = f"Total Mean Acc: \n"
        for nH, head_name in enumerate(
            hparams["multi_head_instruction"][experiment].keys()
        ):
            dataTrue = np.argmax(np.concatenate(result[head_name + "Label"]), axis=1)
            dataPred = np.argmax(
                np.concatenate([re[head_name + "Pred"] for re in result["y_pred"]]),
                axis=1,
            )
            dataCm = tf.math.confusion_matrix(
                dataTrue,
                dataPred,
                num_classes=np.max(dataTrue) + 1,
            ).numpy()
            # Plot train confusion matrix.
            dataAxes = fig.add_subplot(gs[nR, nH])
            sns.heatmap(dataCm, annot=True, fmt="d", ax=dataAxes, cmap="YlGnBu")

            dataAxes.set_title(name + " " + head_name, wrap=True)
            dataAxes.set_xlabel("Pred Labels")
            dataAxes.set_ylabel("True Labels")
            dataAxes.set_xticklabels(
                labelCodeBook[head_name], fontsize=6.5, rotation=np.pi / 4
            )
            dataAxes.set_yticklabels(
                labelCodeBook[head_name], fontsize=6.5, rotation=np.pi / 4
            )

            text_info += f"{head_name}: {total_mean_acc(dataCm):.3f}\n"

        textAxes = fig.add_subplot(gs[nR, nH + 1])
        textAxes.axis("off")
        textAxes.text(0, 0.3, text_info, wrap=True)

    # Set fig property.
    plt.tight_layout()

    figSavePath = join(
        mlflowArtifactPath, "confusion_matrix", "Epoch" + str(starting_epoch)
    )
    # Save fig.
    if not exists(figSavePath):
        os.makedirs(figSavePath)
        print(f"Create {figSavePath} to store image.png")
    fig.savefig(join(figSavePath, "confusion matrix.png"), transparent=transparent)

    # Close fig to release memory.
    # RuntimeWarning: More than 20 figures have been opened.
    # Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained
    # until explicitly closed and may consume too much memory.
    # (To control this warning, see the rcParam `figure.max_open_warning`).
    plt.close(fig)


def plot_residual_fig(
    trainResult: dict,
    val_result: dict,
    testResult: dict,
    hparams: dict,
    experiment: int,
    starting_epoch: int,
    mlflowArtifactPath: str,
    transparent=False,
):
    def createTable(result, headType):
        headClass = sorted(
            list(set(np.concatenate(result[headType]).astype(np.int32).tolist()))
        )
        classStatistic = {hC: {"Name": str(hC) + " mm"} for hC in headClass}
        for classKey in classStatistic.keys():
            dataTrue = np.concatenate(result[headType])
            dataPred = np.concatenate(
                [re[headType + "Pred"] for re in result["y_pred"]]
            )
            dataIndex = np.where(dataTrue == classKey)
            classStatistic[classKey]["True"] = dataTrue[dataIndex]
            classStatistic[classKey]["Pred"] = dataPred[dataIndex]
            classStatistic[classKey]["Abs Error Mean"] = np.mean(
                np.abs(dataTrue[dataIndex] - dataPred[dataIndex])
            )
            classStatistic[classKey]["Abs Error Std"] = np.std(
                np.abs(dataTrue[dataIndex] - dataPred[dataIndex])
            )
            classStatistic[classKey]["Amount"] = len(dataTrue[dataIndex])
        tableReturn = {}
        tableReturn["colLabels"] = [
            classStatistic[k]["Name"] for k in classStatistic.keys()
        ]
        tableReturn["rowLabels"] = ["Amount", "Abs Error Mean", "Abs Error Std"]
        tableReturn["cellText"] = [
            ["%.3f" % classStatistic[k][row] for k in classStatistic.keys()]
            for row in ["Amount", "Abs Error Mean", "Abs Error Std"]
        ]
        return tableReturn, classStatistic

    fig = plt.figure(
        figsize=(5 + 5 * len(hparams["multi_head_instruction"][experiment]), 12)
    )
    gs = plt.GridSpec(
        3, 3 * len(hparams["multi_head_instruction"][experiment]), figure=fig
    )

    for nR, (name, result) in enumerate(
        zip(["Train", "validation", "Test"], [trainResult, val_result, testResult])
    ):
        for nH, headType in enumerate(
            hparams["multi_head_instruction"][experiment].keys()
        ):
            dataTrue = np.concatenate(result[headType])
            dataPred = np.concatenate(
                [re[headType + "Pred"] for re in result["y_pred"]]
            )
            # tableReturn, classStatistic = createTable(result, headType)

            # Residual plots.
            predictedAxes = fig.add_subplot(gs[nR, 3 * nH])
            # residualAxes = fig.add_subplot(gs[nR, 3 * nH + 1])
            # tableAxes = fig.add_subplot(gs[nR, 3 * nH + 2])

            predictedAxes.plot(dataPred, dataTrue, "o", markersize=0.5)
            predictedAxes.set_title(
                name + " " + headType + " Regression Plot", wrap=True
            )
            predictedAxes.set_xlabel("Pred " + headType + " (mm)")
            predictedAxes.set_ylabel("True " + headType + " (mm)")

            # for k in list(classStatistic.keys()):
            #     if len(classStatistic[k]["Pred"]) == 0:
            #         print(f"Delete {k} because it does not have data.")
            #         del classStatistic[k]

            # residualAxes.plot(thicknessTrue,
            #                   thicknessPred - thicknessTrue, 'o')
            # residualAxes.violinplot(
            #     [classStatistic[k]["Pred"] for k in classStatistic.keys()],
            #     list(classStatistic.keys()),
            #     widths=(list(classStatistic.keys())[1] - list(classStatistic.keys())[0])
            #     / 2,
            #     showmeans=False,
            #     showmedians=False,
            #     showextrema=False,
            # )
            # residualAxes.set_title(name + " " + headType + " Residual Plot", wrap=True)
            # residualAxes.set_xlabel("True " + headType + " (mm)")
            # residualAxes.set_ylabel("Residual (mm)")

            # tableAxes.table(
            #     cellText=tableReturn["cellText"],
            #     rowLabels=tableReturn["rowLabels"],
            #     colLabels=tableReturn["colLabels"],
            #     loc="center",
            # )
            # tableAxes.axis("off")
            # tableAxes.set_title(name + " " + headType + " Summary", wrap=True)

    figSavePath = join(
        mlflowArtifactPath, "residual_plot", "Epoch" + str(starting_epoch)
    )
    # Set fig property.
    plt.tight_layout()

    # Save fig.
    if not exists(figSavePath):
        os.makedirs(figSavePath)
        print(f"Create {figSavePath} to store image.png")
    fig.savefig(join(figSavePath, "residual plot.png"), transparent=transparent)

    # Close fig to release memory.
    # RuntimeWarning: More than 20 figures have been opened.
    # Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained
    # until explicitly closed and may consume too much memory.
    # (To control this warning, see the rcParam `figure.max_open_warning`).
    plt.close(fig)


def main(training_flag, DEVICE, data_root):
    # Get number of avaliable CPU cores.
    local_cpu_count = multiprocessing.cpu_count()

    # Get number of avaliable GPU.
    local_gpu_count = jax.local_device_count()
    print(f"-----Avaliable CPU cores: {local_cpu_count}, GPU: {local_gpu_count}-----")

    # Define log epoch.
    LOG_EPOCH = 10

    # Set mlflow parameters.
    mlflow.set_registry_uri(join(".", DEVICE))
    mlflow.set_tracking_uri(join(".", DEVICE))

    # Set data size.
    DATA_SIZE = (100, 1, 2)

    if "Samples_plate_Runs" in DEVICE:
        hparams = {
            # -----------Experiment setting-----------
            "number_of_experiment": 1,
            "description": ["This model is trained on data_21032023."],
            "run_name": ["Samples plate DL project"],
            # -----------Model setting-----------
            "model_name": ["resnet18"],
            "n_last_second_logit": [4],
            "multi_head_instruction": [{"us": 1}],  # {"us": 1, "liftoff": 1}
            "resnet_v2": [True],
            "simpleCNN_list": [[64, 64, 64]],
            "simpleCNN_stride": [2],
            "mlp_list": [[1024, 1164, 2048]],
            # -----------Dataset setting-----------
            "dataset_name": ["data_21032023"],
            "batch_size": [64],
            "z_norm_flag": [True],
            # -----------optimizer setting-----------
            "optimizer": ["adam"],
            "lr": [0.001],
            "lr_schedule_flag": [True],
            # -----------Training setting-----------
            "epoch": [1500],
            # None, 0.0001
            "weight_decay": [0.000001],
            # huber_loss, l2_loss, softmax_cross_entropy
            "loss_name": ["l2_loss"],
            # regression, classification
            "problem": ["regression"],
        }

    # Define PRNGKey.
    rng = jax.random.PRNGKey(666)

    if training_flag:
        # Loop for hyperparameters.
        for experiment in range(hparams["number_of_experiment"]):
            # Check if the experiment has already run.
            already_ran_flag, previous_run_id, starting_epoch = utils._already_ran(
                {
                    k: hparams[k][experiment]
                    for k in hparams.keys()
                    if k != "number_of_experiment"
                }
            )
            # If already ran, skip this experiment.
            if already_ran_flag:
                print(f"Experiment is skiped.")
                continue
            # Run experiment.
            with mlflow.start_run(
                run_id=previous_run_id,
                run_name=hparams["run_name"][experiment],
                description=hparams["description"][experiment],
            ) as active_run:
                # log hyper parameters.
                utils.mf_loghyperparams(hparams, experiment)

                # Get mlflow artifact saving path.
                mlflow_artifact_path = unquote(
                    urlparse(active_run.info.artifact_uri).path
                )
                if "t50851tm" in mlflow_artifact_path:
                    mlflow_artifact_path = os.path.relpath(
                        mlflow_artifact_path, "/net/scratch2/t50851tm/momaml_jax"
                    )

                # Load dataset.
                (
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    dataset_info,
                ) = dataset.dataPipeline(
                    data_root,
                    data_code=hparams["dataset_name"][experiment],
                    batch_size=hparams["batch_size"][experiment],
                    z_norm_flag=hparams["z_norm_flag"][experiment],
                )

                # Define _forward and optimizer.
                (
                    _forward,
                    optimizer,
                    optimizerSchedule,
                    summary_message,
                ) = define_forward_and_optimizer(hparams, experiment, DATA_SIZE)

                # Log model architecture.
                mlflow.log_text(
                    summary_message, join("summary", "model_architecture.txt")
                )

                # Transform forward-pass into pure functions.
                forward = hk.without_apply_rng(hk.transform_with_state(_forward))

                # Define training loss function.
                loss_fn = define_loss_fn(
                    forward,
                    is_training=True,
                    optax_loss=utils.lossSelector(hparams["loss_name"][experiment]),
                    hparams=hparams,
                    experiment=experiment,
                )

                # Define train_step.
                train_step = define_train_step(loss_fn, optimizer, optimizerSchedule)

                # Define train.
                train = define_train(train_step, hparams, experiment)

                # Define test loss function.
                loss_fn_test = define_loss_fn(
                    forward,
                    is_training=False,
                    optax_loss=utils.lossSelector(hparams["loss_name"][experiment]),
                    hparams=hparams,
                    experiment=experiment,
                )

                # Define test.
                test = define_test(loss_fn_test, hparams, experiment)

                # Initialize train_exp_state.
                train_exp_state = initialize_train_exp_state(
                    DATA_SIZE, forward, optimizer, mlflow_artifact_path
                )

                # Check if restore checkpoint is needed.
                if starting_epoch != 0 and starting_epoch is not None:
                    train_exp_state = restore_exp_state(
                        starting_epoch, mlflow_artifact_path
                    )
                    if "best_test_loss" in active_run.data.metrics.keys():
                        best_test_loss = active_run.data.metrics["best_test_loss"]
                    else:
                        best_test_loss = 99999.9
                    print("Restored from Epoch", starting_epoch)
                else:
                    best_test_loss = 99999.9

                # Training loop.
                for epoch in range(starting_epoch, hparams["epoch"][experiment]):
                    start_time = time.time()
                    # Update trainState.
                    train_exp_state, train_result = train(
                        train_exp_state, train_dataset
                    )
                    val_result = test(train_exp_state, val_dataset)
                    test_result = test(train_exp_state, test_dataset)

                    if ((epoch % LOG_EPOCH == 0) and (epoch != 0)) or (
                        epoch == hparams["epoch"][experiment] - 1
                    ):
                        plot_residual_fig(
                            train_result,
                            val_result,
                            test_result,
                            hparams,
                            experiment,
                            epoch,
                            mlflow_artifact_path,
                        )

                        jsonSavePath = join(mlflow_artifact_path, "result")
                        if not exists(jsonSavePath):
                            os.makedirs(jsonSavePath)
                        for name, results in zip(
                            [
                                "trainResult_" + "epoch" + str(epoch) + ".json",
                                "validationResult_" + "epoch" + str(epoch) + ".json",
                                "testResult_" + "epoch" + str(epoch) + ".json",
                            ],
                            [train_result, val_result, test_result],
                        ):
                            with open(
                                join(
                                    jsonSavePath,
                                    name,
                                ),
                                "w",
                            ) as f:
                                json.dump(results, f)

                    # Save model if train query loss is lower.
                    if (test_result["loss"] < best_test_loss) or (
                        epoch == hparams["epoch"][experiment] - 1
                    ):
                        save_exp_state(train_exp_state, epoch, mlflow_artifact_path)

                    best_test_loss = min(test_result["loss"], best_test_loss)

                    # Log metric.
                    mlflow.log_metric("Epoch", epoch, step=epoch)
                    mlflow.log_metric("train_loss", train_result["loss"], step=epoch)
                    mlflow.log_metric("validation_loss", val_result["loss"], step=epoch)
                    mlflow.log_metric("test_loss", test_result["loss"], step=epoch)
                    mlflow.log_metric(
                        "learning_rate", train_result["lr"][-1], step=epoch
                    )
                    mlflow.log_metric("best_test_loss", best_test_loss, step=epoch)
                    mlflow.log_metric(
                        "grads_norm",
                        jnp.mean(jnp.array(train_result["grads_norm"])),
                        step=epoch,
                    )

                    print_message = f" \
                    Run id: {active_run.info.run_id} \n \
                    Epoch: {epoch} \n \
                    time: {time.time() - start_time} s \n \
                    training loss: {train_result['loss']} \n \
                    validation loss: {val_result['loss']} \n \
                    test loss: {test_result['loss']} \n \
                    test us true: {np.array(test_result['us']).reshape(-1) * dataset_info['us']['std'] + dataset_info['us']['mean']} \n \
                    test us pred: {np.array([p['usPred'] for p in test_result['y_pred']]).reshape(-1) * dataset_info['us']['std'] + dataset_info['us']['mean']} \n \
                    test liftoff true: {np.array(test_result['liftoff']).reshape(-1) * dataset_info['liftoff']['std'] + dataset_info['liftoff']['mean']} \n \
                    test liftoff pred: {np.array([p['liftoffPred'] for p in test_result['y_pred']]).reshape(-1) * dataset_info['liftoff']['std'] + dataset_info['liftoff']['mean']} \n"

                    # print_message = f" \
                    # Run id: {active_run.info.run_id} \n \
                    # Epoch: {epoch} \n \
                    # time: {time.time() - start_time} s \n \
                    # training loss: {train_result['loss']} \n \
                    # validation loss: {val_result['loss']} \n \
                    # test loss: {test_result['loss']} \n \
                    # test us true: {np.array(test_result['us']).reshape(-1)} \n \
                    # test us pred: {np.array([p['usPred'] for p in test_result['y_pred']]).reshape(-1)} \n \
                    # test liftoff true: {np.array(test_result['liftoff']).reshape(-1)} \n \
                    # test liftoff pred: {np.array([p['liftoffPred'] for p in test_result['y_pred']]).reshape(-1)} \n"

                    # print_message = f" \
                    # Run id: {active_run.info.run_id} \n \
                    # Epoch: {epoch} \n \
                    # time: {time.time() - start_time} s \n \
                    # training loss: {train_result['loss']} \n \
                    # validation loss: {val_result['loss']} \n \
                    # test loss: {test_result['loss']} \n \
                    # test us true: {np.array(test_result['us']).reshape(-1) * dataset_info['us']['std'] + dataset_info['us']['mean']} \n \
                    # test us pred: {np.array([p['usPred'] for p in test_result['y_pred']]).reshape(-1) * dataset_info['us']['std'] + dataset_info['us']['mean']} \n"

                    mlflow.log_text(print_message, f"Message/Training_Epoch{epoch}.txt")
                    print(print_message)


if __name__ == "__main__":
    # Ensure TF does not see GPU and grab all GPU memory.
    tf.config.set_visible_devices([], device_type="GPU")
    DEVICE = "Samples_plate_Runs"
    DATA_ROOT = join("datasets", "Samples_plate")

    main(
        training_flag=True,
        DEVICE=DEVICE,
        data_root=DATA_ROOT,
    )
