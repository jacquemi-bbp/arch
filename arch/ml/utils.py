"""Utilities for train_and_predict.py script."""

import os
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib_scalebar.scalebar import ScaleBar

# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn.cluster import DBSCAN
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler


def get_image_files(directory_path: Path | str, glob_pattern: str) -> list[str]:
    """Fetch relevant files in provided directory."""
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    files = directory_path.glob(glob_pattern)
    return [f.name for f in files]


def image_to_df(
    image_names: list[str],
    data_path: str,
    classes: list[str] | None = None,
    features: list[str] | None = None,
    filter: bool = True,  # noqa: A002
    distinguish_second_layer: bool = False,
    extension: str = "csv",
    gt_column: str = "Expert_layer",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and filter the dataframe associated to images."""
    image_paths = [
        os.path.join(f"{data_path}", image_name) for image_name in image_names
    ]

    li = []
    df_image_names = []
    for image_name, path in zip(image_names, image_paths):
        if extension == "csv":
            df = pd.read_csv(path, engine="python")
        elif extension == "txt":
            df = pd.read_csv(path, sep="   |\t", engine="python")
        else:
            raise ValueError(
                "Invalid extension: {}. Has to be 'csv' or 'txt'.".format(extension)
            )
        df_image_names.extend([image_name] * len(df))
        li.append(df)

    detection_dataframe = pd.concat(li, axis=0, ignore_index=True)
    detection_dataframe["image_id"] = df_image_names

    # Remove distinction of Layer 2 and Layer 3 if they are not distinguishable.
    if not distinguish_second_layer:
        detection_dataframe.loc[
            detection_dataframe[gt_column] == "Layer 2", gt_column
        ] = "Layer 2/3"
        detection_dataframe.loc[
            detection_dataframe[gt_column] == "Layer 3", gt_column
        ] = "Layer 2/3"
    else:
        # Drop cells with label 'Layer 2/3'.
        detection_dataframe = detection_dataframe[
            detection_dataframe[gt_column] != "Layer 2/3"
        ]

    if filter:
        if features and classes:
            detection_dataframe_filtered = detection_dataframe[
                features + [gt_column]
            ].dropna()
            detection_dataframe = detection_dataframe.loc[
                detection_dataframe_filtered.index
            ]
            if len(set(detection_dataframe_filtered[gt_column].unique())) != 1:
                detection_dataframe_filtered = detection_dataframe_filtered[
                    detection_dataframe_filtered[gt_column].isin(classes)
                ]
            x = detection_dataframe_filtered[features]
            y = detection_dataframe_filtered[gt_column]
        else:
            raise ValueError("The features cannot be None if applying filtering.")
    else:
        detection_dataframe = detection_dataframe.dropna()
        x = detection_dataframe
        y = detection_dataframe[gt_column]
    return x, y, detection_dataframe


def plot_cell_by_predicted_layers(
    _cells_dataframe: pd.DataFrame,
    _predict_layers: pd.DataFrame,
    name: str,
    img_path: Path | str | None = None,
    _truth_layers: pd.DataFrame | None = None,
    distinguishable_layers: bool = False,
    show_fig: bool = False,
    sub_ax: np.ndarray[Any, Any] | None = None,
) -> int:
    """Plot results of predictions or ground truth."""
    X = _cells_dataframe["Centroid X µm"].to_numpy(dtype=np.float64)
    Y = _cells_dataframe["Centroid Y µm"].to_numpy(dtype=np.float64)
    layers_dict = defaultdict(list)
    wrong_cells_dict = defaultdict(list)
    for index, value in enumerate(_predict_layers):
        layers_dict[value].append([X[index], Y[index]])
        if _truth_layers:
            if _truth_layers[index] is not _predict_layers[index]:
                wrong_cells_dict[_truth_layers[index]].append([X[index], Y[index]])
    fig = plt.figure(figsize=(6, 6))
    if sub_ax is None:
        ax = plt.gca()
    else:
        ax = sub_ax
    ax.invert_yaxis()

    if distinguishable_layers:
        layer_color = {
            "Layer 1": (1, 0, 0),
            "Layer 2": (1, 0, 153 / 255),
            "Layer 3": (204 / 255, 0, 1),
            "Layer 4": (51 / 255, 0, 1),
            "Layer 5": (0, 102 / 255, 1),
            "Layer 6 a": (0, 1, 1),
            "Layer 6 b": (0, 1, 102 / 255),
        }
    else:
        layer_color = {
            "Layer 1": (1, 0, 0),
            "Layer 2/3": (117 / 255, 20 / 255, 2 / 255),
            "Layer 4": (51 / 255, 0, 1),
            "Layer 5": (0, 102 / 255, 1),
            "Layer 6 a": (0, 1, 1),
            "Layer 6 b": (0, 1, 102 / 255),
        }
    for layer_name, coor_list in layers_dict.items():
        coor = np.array(coor_list)
        color = layer_color[layer_name]
        ax.scatter(
            coor[:, 0], coor[:, 1], s=20, alpha=0.7, color=color, label=layer_name
        )

    if _truth_layers:
        for layer_name, coor_list in wrong_cells_dict.items():
            coor = np.array(coor_list)
            color = layer_color[layer_name]
            ax.scatter(
                coor[:, 0], coor[:, 1], s=1, alpha=1.0, color=color, label=layer_name
            )

    # bar = AnchoredSizeBar(ax.transData, 250, "", loc="lower left", frameon=False)
    # ax.add_artist(bar)
    # plt.gca().add_artist(scalebar)
    ax.set_title(name, fontweight="bold", loc="left")
    leg = plt.legend(layer_color.keys())
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for handle, color in zip(leg.legend_handles, list(layer_color.values())):
        handle.set_color(color)
    if sub_ax is None:
        if img_path:
            fig.savefig(
                os.path.join(img_path, Path(name).stem + ".svg"),
                bbox_inches="tight",
                dpi=150,
            )
            fig.savefig(
                os.path.join(img_path, Path(name).stem + ".png"),
                bbox_inches="tight",
                dpi=150,
            )
        else:
            warnings.warn(
                "The img_path variable is not set. The images cannot be saved.",
                stacklevel=2,
            )
    if sub_ax is None and show_fig:
        plt.show()
    plt.close()
    return 0


def clean_predictions(
    df: pd.DataFrame,
    pred: np.ndarray[Any, Any],
    eps: float = 0.05,
    min_samples: int = 3,
    gt_column: str = "Expert_layer",
) -> pd.Series:
    """Clean prediction based on obvious discrepencies.

    Parameters
    ----------
    df
        Dataframe containing the features.
    pred
        Numpy array containing the layer prediction for each sample.
    eps
        Max distance at which we considered a point as being.
    min_samples
        Minimum number of samples that must fall within the 'eps' radius to be concidered as a central point.

    Returns
    -------
    Improved predictions.
    """
    df[gt_column] = pred
    df.reset_index(drop=True, inplace=True)
    # Split per image
    unique_id = df["image_id"].unique()

    # For each image
    for image_id in unique_id:
        # Get the points and their classes.
        image = df[df["image_id"] == image_id]
        unique_classes = sorted(image[gt_column].unique())

        for i, layer in enumerate(unique_classes):
            # Get the position of each point classified in 'layer'
            points_in_layer = image[image[gt_column] == layer][
                ["Centroid X µm", "Centroid Y µm"]
            ]

            # Scale the axis to avoid scale differences between images.
            # Mean distance between cells after scaling is ~0.05.
            points_in_layer = (
                MinMaxScaler()
                .set_output(transform="pandas")
                .fit_transform(points_in_layer)
            )

            # Get the main cluster + sub-clusters.
            # Layer 1 is sparser, so we allow for triple the distance.
            cluster = DBSCAN(
                eps=eps * 3 if layer == "Layer 1" else eps, min_samples=min_samples
            ).fit(points_in_layer.to_numpy())

            # Compute average depth of the layer to assign outliers accordingly.
            avg_distance = points_in_layer["Centroid Y µm"].mean()

            # Select the main cluster based on number of cells belonging to it.
            cluster_size = Counter(cluster.labels_)
            # Modify the prediction of points not belonging to the main cluster.
            for index, point in points_in_layer[
                cluster.labels_ != max(cluster_size, key=cluster_size.get)  # type: ignore
            ].iterrows():
                # Assign points that have been found to be outside the main cluster to another layer.
                # Layer 'below' if they're located deeper than the average layer depth.
                if (
                    point["Centroid Y µm"] > avg_distance
                    and layer != unique_classes[-1]
                ):
                    pred[index] = unique_classes[i + 1]
                # Layer 'above' if they're located above the average layer depth.
                if point["Centroid Y µm"] < avg_distance and layer != unique_classes[0]:
                    pred[index] = unique_classes[i - 1]
    return pred


def plot_results(
    image_name: str,
    data_path: str,
    save_path: str,
    classes: list[str],
    models: list[tuple[Any, str]],
    features: list[str],
    gt_column: str = "Expert_layer",
    distinguish_second_layer: bool = False,
    extension: str = "csv",
    clean: bool = False,
    eps: float = 0.06,
    min_samples: int = 3,
    show: bool = False,
) -> tuple[pd.DataFrame, list[Any]]:
    """Plot images of ground truth labels and classification results of all models."""
    validation_dataframe, ground_truth_layers, _ = image_to_df(
        image_names=[image_name],
        data_path=data_path,
        classes=classes,
        filter=False,
        distinguish_second_layer=distinguish_second_layer,
        extension=extension,
        gt_column=gt_column,
    )
    ground_truth_layers = ground_truth_layers.tolist()

    fig, axes = plt.subplots(1, (len(models) + 1), figsize=(5 * (len(models) + 1), 5))

    # Plot ground truth
    plot_cell_by_predicted_layers(
        validation_dataframe,
        ground_truth_layers,
        name="A",
        img_path=save_path,
        distinguishable_layers=distinguish_second_layer,
        sub_ax=axes[0],
    )

    model_predictions = []
    for i, model in enumerate(models):
        if clean:
            predict_layers = clean_predictions(
                validation_dataframe,
                model[0].predict(validation_dataframe[features].values),
                eps=eps,
                min_samples=min_samples,
                gt_column=gt_column,
            )
        else:
            predict_layers = model[0].predict(validation_dataframe[features].values)
        model_predictions.append(predict_layers)
        plot_cell_by_predicted_layers(
            validation_dataframe,
            predict_layers,
            name=chr(ord("@") + i + 2),
            img_path=save_path,
            distinguishable_layers=distinguish_second_layer,
            sub_ax=axes[i + 1],
        )
    scale_formatter = lambda value, unit: ""  # noqa: E731
    scalebar = ScaleBar(
        dx=1,
        units="um",
        box_alpha=0,
        location="lower left",
        scale_formatter=scale_formatter,
    )
    axes[0].add_artist(scalebar)
    handles, labels = axes[-1].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles)))
    plt.legend(handles, labels, loc=(1.1, 0.05))
    plt.subplots_adjust(wspace=-0.2)
    fig.savefig(
        os.path.join(save_path, Path(image_name).stem + ".svg"),
        bbox_inches="tight",
        dpi=150,
    )
    fig.savefig(
        os.path.join(save_path, Path(image_name).stem + ".png"),
        bbox_inches="tight",
        dpi=150,
    )
    if show:
        plt.show()
    return ground_truth_layers, model_predictions


def has_columns(cols: list[str], df: pd.DataFrame) -> bool:
    """Check if the df has certain columns."""
    try:
        _ = df[cols]
    except KeyError as e:
        print(f"Missing columns: {e}")
        return False
    return True


def plot_eval_metrics(
    scores: tuple[
        float | np.ndarray[Any, Any],
        float | np.ndarray[Any, Any],
        float | np.ndarray[Any, Any],
        np.ndarray[Any, Any] | None,
    ],
    per_class_accuracy: list[float],
    confusion_matrix: np.ndarray[Any, Any],
    classes: list[str],
    model_name: str,
    axes: np.ndarray[Any, Any] | None = None,
    show: bool = False,
    save_path: str | None = None,
) -> None:
    """Plot evaluation metrics."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(17, 6))

    df = pd.DataFrame(
        np.stack([per_class_accuracy] + list(scores[:3])).T,
        columns=["Accuracy", "Precision", "Recall", "F1-Score"],
        index=classes,
    )

    cmap = sns.color_palette("YlGnBu")
    axes[0].set_title("Fig. A: Classification metrics")
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=1,
        linecolor="black",
        ax=axes[0],
    )
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

    plt.suptitle(f"Evaluation metrics for {model_name}", fontsize=16)
    plt.xlabel("Metrics")
    plt.ylabel("Classes")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=classes
    )
    disp.plot(ax=axes[1], cmap="viridis")
    axes[1].set_title("Fig. B: Classification counts")
    axes[1].minorticks_off()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".svg")
    if show:
        plt.show()


def plot_crossval_metrics(
    scores: np.ndarray,
    per_class_accuracy: np.ndarray,
    confusion_matrix: np.ndarray,
    classes: list[str],
    model_name: str,
    ax_metrics,
    ax_confusion,
    subplot_names: list[str],
    fontsize=30,
) -> None:
    """Plot cross-validation metrics and confusion matrix.

    Parameters
    ----------
    scores
        Array of cross-validation scores.
    per_class_accuracy
        Array of per-class accuracies.
    confusion_matrix
        Confusion matrix for per-class numbers.
    classes
        List of class names for the Layers.
    model_name
        Name of the model.
    ax_metrics
        Ax to plot cross-validation metrics.
    ax_confusion
        Axe to plot confusion matrix.
    subplot_names
        Names for the subplots of metrics and confusion matrix.
    fontsize
        Font size for the plots.
    """
    df = pd.DataFrame(
        np.stack([per_class_accuracy] + list(scores[:3])).T,
        columns=["Accuracy", "Precision", "Recall", "F1-Score"],
        index=classes,
    )
    cmap = sns.color_palette("YlGnBu")

    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=1,
        linecolor="black",
        ax=ax_metrics,
    )
    if subplot_names:
        ax_metrics.set_title(
            subplot_names[0], loc="left", x=0, fontdict={"fontsize": fontsize}
        )
    else:
        ax_metrics.set_title(f"Classification metrics for {model_name}")
    ax_metrics.set_yticklabels(ax_metrics.get_yticklabels(), rotation=0)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=classes
    )
    disp.plot(ax=ax_confusion, cmap="viridis")
    ax_confusion.set_title(
        subplot_names[1], loc="left", x=0, fontdict={"fontsize": fontsize}
    )
    ax_confusion.minorticks_off()


def plot_overall_metrics(
    metrics: np.ndarray[float, Any] | list[float],
    names: list[str],
    ax_overall,
    subplot_name: str,
    fontsize=30,
) -> None:
    """Plot overall metrics.

    Parameters
    ----------
    metrics : np.ndarray
        Array of metric values.
    names : list[str]
        List of metric names.
    ax_overall
        Axes to plot the overall metrics.
    subplot_name : str
        Name for the subplot.
    fontsize : int, optional
        Font size for the plots, by default 30.
    """
    overall_metrics = pd.DataFrame(
        np.stack(metrics),
        columns=["Accuracy", "Precision", "Recall", "F1-Score"],
        index=names,
    )
    cmap = sns.color_palette("YlGnBu")
    sns.heatmap(
        overall_metrics,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=1,
        linecolor="black",
        ax=ax_overall,
    )
    ax_overall.set_title(subplot_name, loc="left", x=0, fontdict={"fontsize": fontsize})
    ax_overall.set_yticklabels(ax_overall.get_yticklabels(), rotation=0)
