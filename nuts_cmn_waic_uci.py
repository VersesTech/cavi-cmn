# Copyright 2024 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the “License”);
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/cavi-cmn/blob/main/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
from jax import numpy as jnp
from jax import random as jr

from benchmarks import create_uci_dataloader, compute_log_likelihood_nuts

import os
import argparse

import xarray as xr
import arviz as az


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

    # dimension of discrete latents in the Directed Mixture layer
    parser.add_argument("--n_components", default=20, type=int)

    # logging config
    parser.add_argument(
        "--log_waic",
        "-log",
        action="store_true",
        help="Include this flag if you want to additionally log the WAIC score to a .txt file",
    )

    # MCMC kernel config
    parser.add_argument("--n_warmup", "-n_w", default=800, type=int)
    parser.add_argument("--n_nuts_samples", "-nns", default=64, type=int)

    # metrics will be averaged over parallel MCMC chains
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

    prob_type = "stick-breaking"
    num_layers = 1
    hidden_dim = n_classes - 1

    print(f"Dataset={args.data}: n_classes={n_classes}")
    print(
        f"Two Layer NUTS-CMN: components={args.n_components}, hidden_dim={hidden_dim}, n_warmup={args.n_warmup}, n_samples={args.n_nuts_samples}, n_chains={args.n_chains}, floating point dtype: {int(args.precision[-2:])}"
    )

    # number of hidden units (continuous latents) in the single hidden (Directed Mixture) layer of the model
    hidden_dims = [hidden_dim] * num_layers

    # number of components (discrete latents) in the single hidden (Directed Mixture) layer of the model
    n_components = [args.n_components] * num_layers

    if not os.path.exists("./logging/"):
        os.makedirs("./logging/")
    exp_name = f"{args.data}-nuts-cmn-layers={num_layers}-n_components={args.n_components}-hidden_dims={hidden_dim}-n_classes={n_classes}"

    log_pY_all_chains = compute_log_likelihood_nuts(
        model_key,
        x,
        y,
        num_classes=n_classes,
        layer_dims=hidden_dims,
        num_components=n_components,
        prior_type=args.prior_type,
        scale_beta=args.scale_beta,
        num_warmup=args.n_warmup,
        num_nuts_samples=args.n_nuts_samples,
        num_chains=args.n_chains,
        prob_type="stick-breaking",
    )

    dataset = xr.Dataset(
        {
            "obs": (["chain", "draw", "n"], log_pY_all_chains["obs"]),
        },
        coords={
            "chain": (["chain"], jnp.arange(args.n_chains)),
            "draw": (["draw"], jnp.arange(args.n_nuts_samples)),
            "n": (["n"], jnp.arange(x.shape[0])),
        },
    )

    data_nuts = az.InferenceData(log_likelihood=dataset)
    waic = az.waic(data_nuts, pointwise=True).elpd_waic / x.shape[0]

    if args.log_waic:
        fout = open("./logging/" + exp_name + f"-waic" + ".txt", mode="a+")
        print(
            f"waic={waic}",
            file=fout,
        )
        fout.close()
