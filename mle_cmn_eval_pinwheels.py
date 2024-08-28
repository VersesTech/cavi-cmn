# This code is part of the VersesTech Repository `cavi-cmn` (https://github.com/VersesTech/cavi-cmn).
# It is licensed under the VERSES Academic Research License.
#
# For more information, please refer to the license file:
# https://github.com/VersesTech/cavi-cmn/blob/main/license.txt

import jax
from jax import numpy as jnp
from jax import random as jr, vmap

from benchmarks import (
    create_pinwheel_generator,
    fit_cmn_maximum_likelihood,
    check_convergence_expfit,
    grid_of_points,
    plot_dataset,
)

import os
import time
import argparse
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser("Two Layer Conditional Mixture Network-MLE")
    parser.add_argument("--seed", default=1234, type=int)
    ## data config
    parser.add_argument("--train_size", default=200, type=int)
    parser.add_argument("--test_size", default=1000, type=int)
    parser.add_argument("--n_classes", default=5, type=int)
    parser.add_argument("--radial_std", default=0.7, type=float)
    parser.add_argument("--tangential_std", default=0.3, type=float)
    parser.add_argument("--rate", default=0.2, type=float)

    ## model config

    # dimension of discrete latents in the Directed Mixture layer
    parser.add_argument("--n_components", default=20, type=int)

    # number of models to run in parallel (in case you want to average metrics over multiple parallel runs)
    parser.add_argument("--n_models", default=32, type=int)

    # logging config
    parser.add_argument(
        "--log_metrics",
        "-logme",
        action="store_true",
        help="Include this flag if you want to additionally log the metrics of the model training",
    )

    parser.add_argument(
        "--plot_boundary",
        "-pltb",
        action="store_true",
        help="Include this flag if you want to additionally plot the decision boundary by the model",
    )

    parser.add_argument(
        "--log_runtime",
        "-logrt",
        action="store_true",
        help="Include this flag if you want to additionally log the runtimes of the model training",
    )

    # number of iterations of gradient descent on the neg log likelihood loss function
    parser.add_argument("--n_iters", "-n_i", default=20000, type=int)

    # learning rate for the gradient descent steps
    parser.add_argument("--lr", "-lr", default=1e-3, type=float)

    # floating-point precision config
    parser.add_argument(
        "--precision", default="float32", choices=["float32", "float64"], type=str
    )

    # number of m steps used to compute runtime
    parser.add_argument("--n_iters_runtime", default=1e4, type=int)

    args = parser.parse_args()

    args.batch_size = args.train_size

    return args


if __name__ == "__main__":
    args = parse_args()

    # Update precision based on the command line argument
    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)
    elif args.precision == "float32":
        jax.config.update("jax_default_matmul_precision", "float32")

    key = jr.PRNGKey(args.seed)
    data_key, model_key = jr.split(key)
    n_model_keys = jr.split(model_key, args.n_models)

    x_dim = 2  # number of input features (regressor dimension)
    y_dim = args.n_classes  # number of classes in the output (regressand dimension)
    num_layers = 1
    hidden_dim = args.n_classes - 1

    print(
        f"Simulate pinwheel: train_size={args.train_size}, test_size={args.test_size}, n_classes={args.n_classes}, radial_std={args.radial_std}, tangential_std={args.tangential_std}, rate={args.rate}"
    )
    print(
        f"Two Layer MLE-CMN: components={args.n_components}, hidden_dim={hidden_dim}, n_models={args.n_models}, n_iters={args.n_iters}, floating point dtype: {int(args.precision[-2:])}"
    )

    # number of hidden units (continuous latents) in the single hidden (Directed Mixture) layer of the model
    hidden_dims = [hidden_dim] * num_layers

    # number of components (discrete latents) in the single hidden (Directed Mixture) layer of the model
    n_components = [args.n_components] * num_layers

    if not os.path.exists("./logging/"):
        os.makedirs("./logging/")
    exp_name = f"pinwheel-moe-ml-layers={num_layers}-n_components={args.n_components}-hidden_dims={hidden_dim}-train_size={args.train_size}-n_classes={args.n_classes}"

    train_dataloader, test_dataloader = create_pinwheel_generator(
        data_key,
        args.batch_size,
        args.train_size,
        args.test_size,
        args.n_classes,
        radial_std=args.radial_std,
        tangential_std=args.tangential_std,
        rate=args.rate,
    )

    x_train, y_train = next(iter(train_dataloader))
    x_test, y_test = next(iter(test_dataloader))

    if args.plot_boundary:
        grid, _, _ = grid_of_points(5000, [-4, 4], [-4, 4])

    # In order to vmap model generation over keys, first create a partial'd version of `fit_cmn_maximum_likelihood` that
    # only takes the key as an argument, and then vmap this partial'd function over the keys.
    fit_cmn_one_model = partial(
        fit_cmn_maximum_likelihood,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=args.n_classes,
        layer_dims=hidden_dims,
        num_components=n_components,
        lr=args.lr,
        num_iters=args.n_iters,
        prob_type="stick-breaking",
        grid=grid if args.plot_boundary else None,
    )

    # initializes and fits (using gradient descent on the log likelihood of the model) a set of `n_models` independently-initialized Mixture-of-Experts networks on the training data
    train_acc, test_acc, lpd, ece, res, grid_predicted_class = vmap(fit_cmn_one_model)(
        n_model_keys
    )

    print(
        f"Average train / test accuracy: {train_acc.mean():.3f} / {test_acc.mean():.3f}, LPD: {lpd.mean():.3f}, ECE: {ece.mean():.3f}"
    )

    if args.log_metrics:
        fout = open(
            "./logging/" + exp_name + f"-metrics" + ".txt",
            mode="a+",
        )
        for model_i in range(args.n_models):
            print(
                f"model={model_i+1}, train_accuracy={train_acc[model_i]:.3f}, test_accuracy={test_acc[model_i]:.3f}, lpd={lpd[model_i]:.3f}, ece={ece[model_i]:.3f}",
                file=fout,
            )
        fout.close()

    if args.plot_boundary:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
        for model_i in range(args.n_models):
            plot_dataset(
                x_test,
                y_test,
                grid,
                grid_predicted_class[model_i],
                n_classes=args.n_classes,
                exp_name=exp_name + f"-model={model_i+1}",
            )

    if args.log_runtime:

        # compute the number of iterations needed for convergence
        n_iters_convergence = check_convergence_expfit(
            res.losses / args.train_size,
            n_iters_truncate=2000,
            smooth=False,
            pct_of_maximum_thr=7.5e-2,
        )

        fit_cmn_one_model = partial(
            fit_cmn_maximum_likelihood,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            num_classes=args.n_classes,
            layer_dims=hidden_dims,
            num_components=n_components,
            lr=args.lr,
            num_iters=args.n_iters_runtime,
            prob_type="stick-breaking",
            grid=None,
        )

        _, _, _, _, _, _ = fit_cmn_one_model(model_key)

        time_start = time.time()
        _, _, _, _, _, _ = fit_cmn_one_model(model_key)
        time_end = time.time()
        runtime_per_iter = (time_end - time_start) / args.n_iters_runtime

        print(
            f"Average total / per-iter runtime: {runtime_per_iter * jnp.nanmean(n_iters_convergence):.6f} / {runtime_per_iter:.6f}, Convergence step: {jnp.nanmean(n_iters_convergence):.0f}, "
        )

        fout = open(
            "./logging/" + exp_name + f"-runtimes" + ".txt",
            mode="a+",
        )
        for model_i in range(args.n_models):
            print(
                f"model={model_i+1}, n_steps={n_iters_convergence[model_i]:.0f}, runtime={runtime_per_iter * n_iters_convergence[model_i]:.8f}",
                file=fout,
            )
        fout.close()
