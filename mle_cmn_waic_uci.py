# Copyright 2024 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the “License”);
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/cavi-cmn/blob/main/license.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
from jax import random as jr, vmap

from benchmarks import create_uci_dataloader, compute_log_likelihood_cmn

import os
import argparse
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser("Two Layer Conditional Mixture Network-MLE")
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

    ## model config

    # dimension of discrete latents in the Directed Mixture layer
    parser.add_argument("--n_components", default=20, type=int)

    # number of models to run in parallel (in case you want to average metrics over multiple parallel runs)
    parser.add_argument("--n_models", default=32, type=int)

    # logging config
    parser.add_argument(
        "--log_waic",
        "-log",
        action="store_true",
        help="Include this flag if you want to additionally log WAIC score to a .txt file",
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

    (x, y), stats = create_uci_dataloader(
        data_key,
        args.data,
        None,
        None,
        not_split=True,
    )

    x_dim = stats["x_dim"]  # number of input features (regressor dimension)
    y_dim = n_classes = stats[
        "n_classes"
    ]  # number of classes in the output (regressand dimension)

    num_layers = 1
    hidden_dim = n_classes - 1

    print(f"Dataset={args.data}: n_classes={n_classes}")
    print(
        f"Two Layer MLE-CMN: components={args.n_components}, hidden_dim={hidden_dim}, n_models={args.n_models}, n_iters={args.n_iters}, floating point dtype: {int(args.precision[-2:])}"
    )

    # number of hidden units (continuous latents) in the single hidden (Directed Mixture) layer of the model
    hidden_dims = [hidden_dim] * num_layers

    # number of components (discrete latents) in the single hidden (Directed Mixture) layer of the model
    n_components = [args.n_components] * num_layers

    if not os.path.exists("./logging/"):
        os.makedirs("./logging/")
    exp_name = f"{args.data}-mle-cmn-layers={num_layers}-n_components={args.n_components}-hidden_dims={hidden_dim}-n_classes={n_classes}"

    log_py_moe_one_model = partial(
        compute_log_likelihood_cmn,
        x=x,
        y=y,
        num_classes=n_classes,
        layer_dims=hidden_dims,
        num_components=n_components,
        lr=args.lr,
        num_iters=args.n_iters,
        prob_type="stick-breaking",
    )

    log_py_all_models = vmap(log_py_moe_one_model)(n_model_keys)

    waic = log_py_all_models.mean()

    if args.log_waic:
        fout = open("./logging/" + exp_name + f"-waic" + ".txt", mode="a+")
        print(
            f"waic={waic}",
            file=fout,
        )
        fout.close()
