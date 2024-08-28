# This code is part of the VersesTech Repository `cavi-cmn` (https://github.com/VersesTech/cavi-cmn).
# It is licensed under the VERSES Academic Research License.
#
# For more information, please refer to the license file:
# https://github.com/VersesTech/cavi-cmn/blob/main/license.txt

import jax
from jax import numpy as jnp
from jax import random as jr, config, nn, vmap
from jax.numpy import expand_dims as expand

from benchmarks import create_uci_dataloader, find_uci_stats, fit_cmn_nuts

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
    parser = argparse.ArgumentParser("Two Layer Conditional Mixture Network-NUTS")
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

    # dimension of discrete latents in the Conditional Mixture layer
    parser.add_argument("--n_components", default=20, type=int)

    # logging config
    parser.add_argument(
        "--log_metrics",
        "-logme",
        action="store_true",
        help="Include this flag if you want to additionally log the metrics of the model training",
    )

    # MCMC kernel config
    parser.add_argument("--n_warmup", "-n_w", default=800, type=int)
    parser.add_argument("--n_nuts_samples", "-nns", default=64, type=int)

    # n_chains is treated like n_models in other scripts -- metrics are averaged over parallel chains
    parser.add_argument("--n_chains", "-n_c", default=16, type=int)

    # type of prior: either gamma or wishart
    parser.add_argument(
        "--prior_type", "-pt", default="gamma", type=str, choices=["gamma", "wishart"]
    )

    # beta scale
    parser.add_argument("--scale_beta", "-sb", default=5.0, type=float)

    # floating-point precision config
    parser.add_argument(
        "--precision", default="float32", choices=["float32", "float64"], type=str
    )

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

    (train_dataloader, test_dataloader), stats = create_uci_dataloader(
        data_key,
        args.data,
        args.train_size,
        args.test_size,
        max_train_size=args.max_train_size,
        not_split=False,
    )

    x_dim = stats["x_dim"]  # number of input features (regressor dimension)
    # number of classes in the output (regressand dimension)
    y_dim = n_classes = stats["n_classes"]

    num_layers = 1
    hidden_dim = n_classes - 1

    print(f"Dataset={args.data}: train_size={args.train_size}, n_classes={n_classes}")
    print(
        f"Two Layer NUTS-CMN: components={args.n_components}, hidden_dim={hidden_dim}, n_warmup={args.n_warmup}, n_samples={args.n_nuts_samples}, n_chains={args.n_chains}, floating point dtype: {int(args.precision[-2:])}"
    )

    # number of hidden units (continuous latents) in the single hidden (Conditional Mixture) layer of the model
    hidden_dims = [hidden_dim] * num_layers

    # number of components (discrete latents) in the single hidden (Conditional Mixture) layer of the model
    n_components = [args.n_components] * num_layers

    if not os.path.exists("./examples/benchmarks/logging/"):
        os.makedirs("./examples/benchmarks/logging/")
    exp_name = f"{args.data}-nuts-cmn-layers={num_layers}-n_components={args.n_components}-hidden_dims={hidden_dim}-train_size={args.train_size}-n_classes={n_classes}"

    x_train, y_train = next(iter(train_dataloader))
    x_test, y_test = next(iter(test_dataloader))

    train_acc, test_acc, lpd, ece, _ = fit_cmn_nuts(
        model_key,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=n_classes,
        layer_dims=hidden_dims,
        num_components=n_components,
        prior_type=args.prior_type,
        scale_beta=args.scale_beta,
        num_warmup=args.n_warmup,
        num_nuts_samples=args.n_nuts_samples,
        num_chains=args.n_chains,
        prob_type="stick-breaking",
        grid=None,
    )

    print(
        f"Average train / test accuracy: {train_acc.mean():.3f} / {test_acc.mean():.3f}, LPD: {lpd.mean():.3f}, ECE: {ece.mean():.3f}"
    )

    if args.log_metrics:
        fout = open(
            "./examples/benchmarks/logging/" + exp_name + f"-metrics" + ".txt",
            mode="a+",
        )
        for model_i in range(args.n_chains):
            print(
                f"model={model_i+1}, train_accuracy={train_acc[model_i]:.3f}, test_accuracy={test_acc[model_i]:.3f}, lpd={lpd[model_i]:.3f}, ece={ece[model_i]:.3f}",
                file=fout,
            )
        fout.close()
