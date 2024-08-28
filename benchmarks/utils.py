# This code is part of the VersesTech Repository `cavi-cmn` (https://github.com/VersesTech/cavi-cmn).
# It is licensed under the VERSES Academic Research License.
#
# For more information, please refer to the license file:
# https://github.com/VersesTech/cavi-cmn/blob/main/license.txt

from matplotlib import pyplot as plt
import seaborn as sns

import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from .data import JaxNumpyLoader, RandomSampler

import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.optimize import curve_fit
import warnings

from numpyro.diagnostics import hpdi


def convolve_fn(x, window):
    return jnp.convolve(x, window, mode="same")


def exponential_decay_fn(x, a, b, c):
    """Exponential decay function

    Arguments:
        x {np.array} -- Independent variable
        a {float} -- Amplitude
        b {float} -- Decay rate
        c {float} -- Offset
    """
    return a * np.exp(-b * x) + c


def check_convergence_expfit(
    losses,
    window_size=100,
    pct_of_maximum_thr=0.15,
    n_iters_truncate=1000,
    smooth=False,
):
    """
    Function that computes the number of iterations it took for each of a set of `n_models` parallel loss curves
    to converge, operationalized as the time at which they reach a certain percentage of the maximum value.
    This is estimated by fitting an exponential decay curve to the loss curve and solving for the iteration at which
    the curve reaches the desired percentage of the maximum value. Because in practice many loss curves are not well described by a
    simple exponential decay in the first few iterations, we truncate the first `n_iters_truncate` iterations from the loss curve
    before fitting the exponential decay curve.

    This assumes losses are in the shape (n_models, n_iters)

    Arguments:
        losses [jnp.array] -- Timeseries of losses over course of iterations per model in the batch
        window_size [int] -- Size of the window to smooth the loss curve
        pct_change_threshold [float] -- The threshold of the percentage change in loss curve to consider convergence
        n_iters_truncate [int] -- Number of iterations to truncate from the beginning of the loss curve. For BBVI losses this is useful for getting rid of the initial
                                    transient phase of the loss curve, after which the loss is (approximately) describable as an exponential decay.
    """

    """
    dmix-cavi: smooth=False, thr=.15, n_iters_truncate=10
    dmix-bbvi: smooth=True, thr=.15, window_size=100, n_iters_truncate=1000
    moe-ml: smooth=False, thr=.15, n_iters_truncate=2000
    """

    """
    goal:
    dmix-cavi: smooth=False, thr=.1, n_iters_truncate=20 (n_iters=500)
    dmix-bbvi: smooth=True, thr=.1, window_size=100, n_iters_truncate=2000
    moe-ml: smooth=False, thr=0.075, n_iters_truncate=2000 (n_iters=40000)
    """
    # note that n_models is now in the 0-th dimension of losses, which is opposite to the assumptions of `check_convergence` above
    n_models, n_iters = losses.shape

    if smooth:
        # smooth each model's loss curve and cast the batch to numpy arays for compatibility with scipy.optimize.curve_fit
        smoothed_losses = np.array(
            vmap(convolve_fn, (0, None), 0)(losses, jnp.ones(window_size) / window_size)
        )

        # baseline each curve so that its minimum value is 0
        smoothed_losses_baselined = smoothed_losses[
            :, n_iters_truncate : -int(window_size / 2)
        ] - smoothed_losses[:, n_iters_truncate : -int(window_size / 2)].min(
            axis=1, keepdims=True
        )

        # normalize each model's loss curve to [0, 1]
        unit_normalized_losses = (
            smoothed_losses_baselined
            / smoothed_losses_baselined.max(axis=1, keepdims=True)
        )
    else:
        # baseline each curve so that its minimum value is 0
        losses_baselined = losses[:, n_iters_truncate:] - losses[
            :, n_iters_truncate:
        ].min(axis=1, keepdims=True)

        # normalize each model's loss curve to [0, 1]
        unit_normalized_losses = losses_baselined / losses_baselined.max(
            axis=1, keepdims=True
        )

    convergence_iter_per_model = np.zeros(n_models)
    x_axis_length = unit_normalized_losses.shape[1]
    for model_i in range(n_models):
        try:
            exp_params, _ = curve_fit(
                exponential_decay_fn,
                np.arange(0, x_axis_length, 1),
                np.array(unit_normalized_losses[model_i]),
            )
            a, b, c = exp_params
            if (c / a) < (pct_of_maximum_thr / (1 - pct_of_maximum_thr)):

                # this is the formula for the value of x* at which an exponential curve of the form y = f(x) = a * exp(-b * x) + c reaches
                # x* = pct_of_maximum_thr * (a + c), where f(x=0) = a + c
                convergence_iter = -(1.0 / b) * np.log(
                    pct_of_maximum_thr - ((1.0 - pct_of_maximum_thr) * c) / a
                )

                # now when reporting convergence_iter_per_model, correct the iteration by adding back in the n_iters_truncate
                convergence_iter_per_model[model_i] = (
                    convergence_iter + n_iters_truncate
                )

            else:
                # if (c/a) >= (pct_of_maximum_thr / (1 - pct_of_maximum_thr)), then
                # convergence will not be reached (curve never decays to pct_of_maximum_thr of the max value)
                warnings.warn(
                    f"Model {model_i} did not converge to {pct_of_maximum_thr} of the max value. "
                )
                convergence_iter_per_model[model_i] = n_iters
        except:
            warnings.warn(
                f"An exponential decay was not able to be fit to the loss for model {model_i}."
            )
            convergence_iter_per_model[model_i] = np.nan

    return convergence_iter_per_model


# Function to plot the dataset
def plot_dataset(
    X, y, X2=None, y2=None, n_classes=2, title="Dataset Visualization", exp_name=None
):
    # Set the aesthetic appearance of the plots
    sns.set(style="whitegrid", rc={"grid.linestyle": "--", "axes.edgecolor": ".8"})

    plt.figure(figsize=(8, 6))

    # Specific starting colors for the main dataset
    specific_colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    main_palette = specific_colors[:n_classes]
    # Main dataset
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,
        palette=main_palette,
        edgecolor="black",
        alpha=0.9,
        s=25,
        linewidth=1,
    )

    # Optional second dataset
    if X2 is not None and y2 is not None:

        second_palette = [sns.desaturate(color, 0.5) for color in main_palette]

        sns.scatterplot(
            x=X2[:, 0],
            y=X2[:, 1],
            hue=y2,
            palette=second_palette,
            edgecolor=None,
            alpha=0.3,
            s=50,
            marker="X",
        )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.gca().set_facecolor("white")

    # Remove labels and tick marks
    plt.xlabel("")
    plt.ylabel("")
    plt.legend().remove()
    plt.xlim([-3.5, 3.5])
    plt.ylim([-3.5, 3.5])
    plt.legend(title="Class", title_fontsize="13", loc="upper right", fontsize="12")
    if exp_name is not None:
        plt.savefig("./examples/benchmarks/plots/" + exp_name + ".png")
    # plt.show()


def grid_of_points(num_points, x_slide, y_slide):
    # Calculate number of points on each dimension

    width = x_slide[1] - x_slide[0]
    height = y_slide[1] - y_slide[0]

    aspect_ratio = width / height  # Aspect ratio of the rectangle
    num_points_x = int(jnp.sqrt(num_points * aspect_ratio))
    num_points_y = int(num_points / num_points_x)

    # Generate grid points using jax.numpy
    x_values = jnp.linspace(x_slide[0], x_slide[1], num_points_x)
    y_values = jnp.linspace(y_slide[0], y_slide[1], num_points_y)
    grid_x, grid_y = jnp.meshgrid(
        x_values, y_values, indexing="ij"
    )  # Note: Default indexing in jax.numpy.meshgrid is 'xy'
    points = jnp.vstack([grid_x.ravel(), grid_y.ravel()]).T
    # breakpoint()
    return points, grid_x.shape, grid_y.shape


def plot_metrics(
    data,
    results,
    train_size_list,
    model_list,
    column_list,
    colors,
    color_cumstomize,
    data_names,
    data_ylims,
    fs=6,
    lw=2.0,
    fontsize=16,
):
    n_metrics = len(column_list)
    fig = plt.figure(figsize=(n_metrics * fs, fs))
    for i, metric in enumerate(column_list):
        ax = fig.add_subplot(1, n_metrics, i + 1)
        for j, model in enumerate(model_list):
            if model == "dmix-nuts" and metric in ["n_steps", "runtime"]:
                continue
            ax.errorbar(
                train_size_list,
                [v.mean() for v in results[metric][model]],
                yerr=[v.std() for v in results[metric][model]],
                marker="o",
                linestyle=":",
                ecolor=colors[color_cumstomize[model]],
                color=colors[color_cumstomize[model]],
                linewidth=lw,
                label=parse_metric_name(model, delimiter="-", allupper=True),
            )
        ax.set_xlabel("Data Size", fontsize=fontsize)
        ax.set_ylabel(
            parse_metric_name(metric, delimiter="_", allupper=False), fontsize=fontsize
        )

        if i == 0:
            ax.legend()
        if metric in ["train_accuracy", "test_accuracy"]:
            ax.set_ylim(data_ylims[data])

    fig.suptitle(parse_metric_name(data_names[data]), fontsize=fontsize)
    plt.tight_layout()
    # plt.savefig(f"./plots/scp/{data}_{column_list}.png")


def load_results(
    dir_name,
    data,
    train_size_list,
    test_size,
    n_components,
    n_classes,
    model_list,
    performance_list,
    runtime_list,
):
    results = defaultdict(dict)

    for model in model_list:
        for m in performance_list:
            results[m][model] = []
        for r in runtime_list:
            results[r][model] = []

    for model in model_list:
        for train_size in train_size_list:
            # load performance results
            if data == "pinwheel":
                filename = f"{data}/metrics/{data}-{model}-layers=1-n_components={n_components}-hidden_dims={n_classes-1}-train_size={train_size}-test_size={test_size}-n_classes={n_classes}-metrics.txt"
            else:
                filename = f"{data}/metrics/{data}-{model}-layers=1-n_components={n_components}-hidden_dims={n_classes-1}-train_size={train_size}-n_classes={n_classes}-metrics.txt"
            df = pd.read_csv(
                dir_name + filename,
                delimiter=", ",
                header=None,
                names=performance_list,
                engine="python",
            )
            for m in performance_list:
                results[m][model].append(
                    df[m].str[int(len(m) + 1) :].to_numpy(dtype=float)
                )
            # load runtime results
            if model != "dmix-nuts":
                if data == "pinwheel":
                    filename = f"{data}/runtimes/{data}-{model}-layers=1-n_components={n_components}-hidden_dims={n_classes-1}-train_size={train_size}-test_size={test_size}-n_classes={n_classes}-runtimes.txt"
                else:
                    filename = f"{data}/runtimes/{data}-{model}-layers=1-n_components={n_components}-hidden_dims={n_classes-1}-train_size={train_size}-n_classes={n_classes}-runtimes.txt"
                df = pd.read_csv(
                    dir_name + filename,
                    delimiter=", ",
                    header=None,
                    names=runtime_list,
                    engine="python",
                )
                for m in runtime_list:
                    results[m][model].append(
                        df[m].str[int(len(m) + 1) :].to_numpy(dtype=float)
                    )

    return results


def parse_metric_name(m, delimiter=" ", allupper=False):
    """
    a helper function that can transfer a str type variable to printable words

    arguments:
        m (str): str to be parsed
        delimiter (str): the symbol that separate words
        allupper (boolean): turn all letters to uppercase if True, otherwise only capitalize words
    """
    return " ".join(
        [v.upper() if allupper else v.capitalize() for v in m.split(delimiter)]
    )
