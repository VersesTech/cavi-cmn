import jax
from jax import numpy as jnp
from jax import random as jr, config, nn, vmap
from jax.numpy import expand_dims as expand

from benchmarks import (
    create_uci_dataloader,
    find_uci_stats,
    fit_cmn_bbvi,
    check_convergence_expfit,
)

import os
import time
import argparse
import warnings
import numpy as np
from functools import partial
from torchvision import transforms
from collections import defaultdict
import seaborn as sns
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("Two Layer Conditional Mixture Network-BBVI")
    parser.add_argument("--seed", default=1234, type=int)
    ## data config
    parser.add_argument(
        "--data",
        default="rice",
        type=str,
        choices=[
            "rice",
            "waveform",
            "breast_cancer",
            "statlog",
            "banknote",
            "hcv",
            "connectionist_bench",
            "iris",
        ],
    )
    parser.add_argument("--train_size", default=500, type=int)
    parser.add_argument("--test_size", default=100, type=int)
    parser.add_argument("--max_train_size", default=500, type=int)

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
        "--log_runtime",
        "-logrt",
        action="store_true",
        help="Include this flag if you want to additionally log the runtimes of the model training",
    )

    # number of iterations of gradient descent on the neg log likelihood loss function
    parser.add_argument("--n_iters", "-n_i", default=20000, type=int)

    # learning rate for the gradient descent steps
    parser.add_argument("--lr", "-lr", default=5e-3, type=float)

    # type of prior: either gamma or wishart
    parser.add_argument(
        "--prior_type", "-pt", default="gamma", type=str, choices=["gamma", "wishart"]
    )

    # beta scale
    parser.add_argument("--scale_beta", "-sb", default=5.0, type=float)

    # number of nvi samples for prediction
    parser.add_argument("--num_nvi_samples", "-nns", default=64, type=int)

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

    (train_dataloader, test_dataloader), stats = create_uci_dataloader(
        data_key,
        args.data,
        args.train_size,
        args.test_size,
        max_train_size=args.max_train_size,
        not_split=False,
    )

    x_dim = stats["x_dim"]  # number of input features (regressor dimension)
    y_dim = n_classes = stats[
        "n_classes"
    ]  # number of classes in the output (regressand dimension)

    prob_type = "stick-breaking"
    hidden_dim = n_classes - 1
    num_layers = 1

    print(f"Dataset={args.data}: train_size={args.train_size}, n_classes={n_classes}")
    print(
        f"Two Layer BBVI-CMN: components={args.n_components}, hidden_dim={hidden_dim}, n_models={args.n_models}, n_iters={args.n_iters}, n_samples={args.num_nvi_samples}, lr={args.lr}, floating point dtype: {int(args.precision[-2:])}"
    )

    # number of hidden units (continuous latents) in the single hidden (Directed Mixture) layer of the model
    hidden_dims = [hidden_dim] * num_layers

    # number of components (discrete latents) in the single hidden (Directed Mixture) layer of the model
    n_components = [args.n_components] * num_layers

    if not os.path.exists("./examples/benchmarks/logging/"):
        os.makedirs("./examples/benchmarks/logging/")
    exp_name = f"{args.data}-bbvi-cmn-layers={num_layers}-n_components={args.n_components}-hidden_dims={hidden_dim}-train_size={args.train_size}-n_classes={n_classes}"

    x_train, y_train = next(iter(train_dataloader))
    x_test, y_test = next(iter(test_dataloader))

    fit_dmix_bbvi_one_model = partial(
        fit_cmn_bbvi,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=y_dim,
        layer_dims=hidden_dims,
        num_components=n_components,
        prior_type=args.prior_type,
        scale_beta=args.scale_beta,
        lr=args.lr,
        num_iters=args.n_iters,
        prob_type=prob_type,
        grid=None,
    )

    # initializes and fits (using gradient descent on the log likelihood of the model) a set of `n_models` independently-initialized Mixture-of-Experts networks on the training data
    train_acc, test_acc, lpd, ece, res, _ = vmap(fit_dmix_bbvi_one_model)(n_model_keys)

    print(
        f"Average train / test accuracy: {train_acc.mean():.3f} / {test_acc.mean():.3f}, LPD: {lpd.mean():.3f}, ECE: {ece.mean():.3f}"
    )

    if args.log_metrics:
        fout = open(
            "./examples/benchmarks/logging/" + exp_name + f"-metrics" + ".txt",
            mode="a+",
        )
        for model_i in range(args.n_models):
            print(
                f"model={model_i+1}, train_accuracy={train_acc[model_i]:.3f}, test_accuracy={test_acc[model_i]:.3f}, lpd={lpd[model_i]:.3f}, ece={ece[model_i]:.3f}",
                file=fout,
            )
        fout.close()

    if args.log_runtime:

        if args.data == "breast_cancer":
            n_iters_truncate = 2000
            pct_of_maximum_thr = 1.5e-1
        elif args.data == "banknote":
            n_iters_truncate = 2000
            pct_of_maximum_thr = 1.5e-1
        elif args.data == "rice":
            n_iters_truncate = 2000
            if args.train_size <= 320:
                pct_of_maximum_thr = 1.5e-1
            else:
                pct_of_maximum_thr = 5e-2
        elif args.data == "waveform":
            n_iters_truncate = 3000
            if args.train_size <= 960:
                pct_of_maximum_thr = 2e-1
            else:
                pct_of_maximum_thr = 5e-2
        elif args.data == "statlog":
            n_iters_truncate = 3000
            pct_of_maximum_thr = 3e-1
        elif args.data == "connectionist_bench":
            n_iters_truncate = 3000
            pct_of_maximum_thr = 1.5e-1

        else:  # duplicated values for now, but in theory should be done for diff datasets based on our findings
            n_iters_truncate = 2000
            pct_of_maximum_thr = 1.5e-1

        # compute the number of iterations needed for convergence
        n_iters_convergence = check_convergence_expfit(
            res.losses / args.train_size,
            n_iters_truncate=n_iters_truncate,
            smooth=True,
            pct_of_maximum_thr=pct_of_maximum_thr,
        )

        x_train, y_train = next(iter(train_dataloader))
        x_test, y_test = next(iter(test_dataloader))

        fit_dmix_bbvi_one_model = partial(
            fit_cmn_bbvi,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            num_classes=y_dim,
            layer_dims=hidden_dims,
            num_components=n_components,
            prior_type=args.prior_type,
            scale_beta=args.scale_beta,
            lr=args.lr,
            num_iters=args.n_iters_runtime,
            prob_type=prob_type,
            grid=None,
        )
        _, _, _, _, _, _ = fit_dmix_bbvi_one_model(model_key)

        time_start = time.time()
        _, _, _, _, _, _ = fit_dmix_bbvi_one_model(model_key)
        time_end = time.time()
        runtime_per_iter = (time_end - time_start) / args.n_iters_runtime

        print(
            f"Average total / per-iter runtime: {runtime_per_iter * jnp.nanmean(n_iters_convergence):.6f} / {runtime_per_iter:.6f}, Convergence step: {jnp.nanmean(n_iters_convergence):.0f}, "
        )

        fout = open(
            "./examples/benchmarks/logging/" + exp_name + f"-runtimes" + ".txt",
            mode="a+",
        )
        for model_i in range(args.n_models):
            print(
                f"model={model_i+1}, n_steps={n_iters_convergence[model_i]:.0f}, runtime={runtime_per_iter * n_iters_convergence[model_i]:.8f}",
                file=fout,
            )
        fout.close()
