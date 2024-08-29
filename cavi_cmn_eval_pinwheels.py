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
from jax import random as jr, nn
from jax.scipy.special import logsumexp

from cavi_cmn.model_utils import initialize_network
from benchmarks import (
    create_pinwheel_generator,
    check_convergence_expfit,
    grid_of_points,
    plot_dataset,
)

import os
import time
import argparse
import warnings
from tensorflow_probability.substrates import jax as tfp


warnings.simplefilter(action="ignore", category=FutureWarning)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser("Two Layer Conditional Mixture Network-CAVI")
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
    parser.add_argument("--n_components", default=10, type=int)

    # number of models to run in parallel (in case you want to average metrics over multiple parallel runs)
    parser.add_argument("--n_models", default=32, type=int)

    # degrees of freedom offset for the Gamma or Wishart prior
    parser.add_argument("--dof_offset", default=1.0, type=float)

    # scale of V^{-1} of the MatrixNormal prior
    parser.add_argument("--inv_v_scale", default=1e-1, type=float)

    # number of variational bayes iterations for the MNLR output layer
    parser.add_argument("--n_vb_iters_mnlr", nargs="+", type=int)

    # scale of U^{-1} of the inverse Wishart (Gamma) prior within the MNIW (MNG) of the conditional mixture layer
    parser.add_argument("--scale_likelihood", default=1.0, type=float)

    # scale argument of the mnlr (standard deviation of the `self.prior_inv_sigma_mu` term)
    parser.add_argument("--scale_mnlr_betas", nargs="+", type=float)

    # scale of the initially-sampled posterior betas of the MNLR layers
    parser.add_argument("--init_posterior_scale", nargs="+", type=float)

    # whether to sample the initial betas of the MNLR layers
    parser.add_argument("--sample_initial_betas", nargs="+", type=str2bool)

    # number of backward iterations to perform in the backward smoothing pass
    parser.add_argument("--n_backwards_iters", default=8, type=int)

    # whether to remove the final MNLR layer of the network
    parser.add_argument(
        "--remove_mnlr_output",
        action="store_true",
        help="Include this flag to remove the MNLR output layer from the network",
    )

    # what type of prior to use for the likelihood of the Directed Mixture layer
    parser.add_argument("--likelihood_type", default="mng", type=str)

    # whether to fix the precision (i.e., \Sigma_{yy}^{-1}) of the likelihood of the Directed Mixture layer
    parser.add_argument(
        "--fixed_precision",
        action="store_true",
        help="Fix the precision of the Linear likelihood",
    )

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
    # number of M steps taken within the VBEM loop
    parser.add_argument("--n_m_steps", "-n_m", default=200, type=int)

    # for every M step, how many E steps are taken to update the latents across the network
    parser.add_argument("--n_e_steps", "-n_e", default=1, type=int)

    # learning rate for the linear components of the Directed Mixture layer
    parser.add_argument("--lr_linear", "-lr_l", default=1.0, type=float)

    # learning rate for the MNLR output layer
    parser.add_argument("--lr_mnlr", "-lr_m", default=1.0, type=float)

    # batch decay for the linear components of the Directed Mixture layer
    parser.add_argument("--beta_linear", "-b_l", default=0.0, type=float)

    # batch decay for the MNLR output layer
    parser.add_argument("--beta_mnlr", "-b_m", default=0.0, type=float)
    parser.add_argument("--data_seed", default=0, type=int)

    # floating-point precision config
    parser.add_argument(
        "--precision", default="float32", choices=["float32", "float64"], type=str
    )

    # number of m steps used to compute runtime
    parser.add_argument("--n_iters_runtime", default=1e4, type=int)

    args = parser.parse_args()

    args.add_mnlr_output = not args.remove_mnlr_output

    # turn the n_vb_iters_mnlr, scale_mnlr_betas, init_posterior_scale, and sample_initial_betas arguments into tuples
    # If no data was passed in (in the form of that +nargs setting of the parser), set them to the default values
    if args.n_vb_iters_mnlr is None:
        args.n_vb_iters_mnlr = (1, 1)
    else:
        args.n_vb_iters_mnlr = tuple(args.n_vb_iters_mnlr)

    if args.scale_mnlr_betas is None:
        args.scale_mnlr_betas = (5.0, 5.0)
    else:
        args.scale_mnlr_betas = tuple(args.scale_mnlr_betas)

    if args.init_posterior_scale is None:
        args.init_posterior_scale = (1e2, 1e2)
    else:
        args.init_posterior_scale = tuple(args.init_posterior_scale)

    if args.sample_initial_betas is None:
        args.sample_initial_betas = (False, False)
    else:
        args.sample_initial_betas = tuple(args.sample_initial_betas)

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
    cmix_key, mnlr_key = jr.split(model_key)
    x_dim = 2  # number of input features (regressor dimension)
    y_dim = args.n_classes  # number of classes in the output (regressand dimension)
    num_cmix_layers = 1
    num_layers = num_cmix_layers + args.add_mnlr_output
    hidden_dim = args.n_classes - 1

    print(
        f"Simulate pinwheel: train_size={args.train_size}, test_size={args.test_size}, n_classes={args.n_classes}, radial_std={args.radial_std}, tangential_std={args.tangential_std}, rate={args.rate}"
    )
    print(
        f"Two Layer CAVI-CMN: components={args.n_components}, hidden_dim={hidden_dim}, likelihood={args.likelihood_type}, fixed_precision={args.fixed_precision}, n_models={args.n_models}, m_steps={args.n_m_steps}, floating point dtype: {int(args.precision[-2:])}"
    )

    # if we are running multiple models in parallel, we need to add a batch dimension to the parameters
    batch_shape = () if args.n_models == 1 else (args.n_models,)
    # number of hidden units (continuous latents) in the single hidden (Directed Mixture) layer of the model
    hidden_dims = [hidden_dim] * num_layers

    # number of components (discrete latents) in the single hidden (Directed Mixture) layer of the model
    n_components = [args.n_components] * num_cmix_layers

    if not os.path.exists("./logging/"):
        os.makedirs("./logging/")
    exp_name = f"pinwheel-cavi-cmn-layers={num_cmix_layers}-n_components={args.n_components}-hidden_dims={hidden_dim}-train_size={args.train_size}-n_classes={args.n_classes}"

    cmix_optim_args = {
        "learning_rate_linear": args.lr_linear,
        "batch_decay_linear": args.beta_linear,
        "learning_rate_mnlr": args.lr_mnlr,
        "batch_decay_mnlr": args.beta_mnlr,
    }
    mnlr_optim_args = {
        "iters": args.n_vb_iters_mnlr[-1],
        "lr": args.lr_mnlr,
        "beta": args.beta_mnlr,
    }

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

    mnlr_keys = jr.split(mnlr_key, num_layers)
    linear_keys = jr.split(cmix_key, num_cmix_layers)

    # initialize the model
    model = initialize_network(
        mnlr_keys,
        linear_keys,
        n_components,
        hidden_dims,
        batch_shape,
        args.dof_offset,
        args.inv_v_scale,
        x_dim,
        y_dim,
        args.add_mnlr_output,
        args.n_vb_iters_mnlr,
        args.scale_likelihood,
        args.scale_mnlr_betas,
        args.init_posterior_scale,
        args.sample_initial_betas,
        args.fixed_precision,
        args.likelihood_type,
        args.n_backwards_iters,
        cmix_optim_args,
        mnlr_optim_args,
        compute_elbo=(True if args.log_runtime else False),
    )

    x_train, y_train = next(iter(train_dataloader))
    x_test, y_test = next(iter(test_dataloader))

    # if batch_shape is non-empty (i.e., we're running more than one model), then append a trivial batch dimension
    # to the inputs and labels (for both training and testing sets)
    batch_dims_to_expand = (
        () if batch_shape == () else tuple(range(1, len(batch_shape) + 1))
    )
    x_train_expanded = jnp.expand_dims(x_train, batch_dims_to_expand + (-1,))
    y_train_expanded = (
        nn.one_hot(y_train, args.n_classes)
        if batch_shape == ()
        else jnp.expand_dims(nn.one_hot(y_train, args.n_classes), batch_dims_to_expand)
    )

    x_test_expanded = jnp.expand_dims(x_test, batch_dims_to_expand + (-1,))
    y_test_expanded = (
        y_test if batch_shape == () else jnp.expand_dims(y_test, batch_dims_to_expand)
    )

    # train the model while storing the ELBO, the training accuracy, and the test accuracy over the course of M steps
    elbo_over_iters, train_acc_over_iters, test_acc_over_iters = model.fit_vmp(
        x_train_expanded,
        y_train_expanded,
        n_m_steps=args.n_m_steps,
        compute_accuracy=True,
        x_test=x_test_expanded,
        y_test=y_test_expanded,
    )

    train_acc_over_iters = jnp.array(train_acc_over_iters)
    test_acc_over_iters = jnp.array(test_acc_over_iters)

    # Get the predictions of the model on the test set
    logits = model.predict(x_test_expanded).logits
    logits = logits - logsumexp(logits, -1, keepdims=True)

    # Print the test accuracy, ECE, and LPD of each model
    # loop over the predictions of each model and use them to compute (and store) the train accuracy, test accuracy, ECE, and LPD of each
    train_acc = train_acc_over_iters[-1, :]
    test_acc = test_acc_over_iters[-1, :]
    lpd = []
    ece = []

    for model_i in range(args.n_models):
        logits_model_i = logits[:, model_i, :]
        pred_y_i = logits_model_i.argmax(-1)

        # Compute the expected calibration error (ECE) for the model's predictions
        ece_i = tfp.stats.expected_calibration_error(
            20, logits=logits_model_i, labels_true=y_test, labels_predicted=pred_y_i
        )

        # Compute the log-predictive density (LPD) of the model's predictions
        lpd_i = jnp.sum(logits_model_i * nn.one_hot(y_test, args.n_classes), -1).mean()

        ece.append(ece_i)
        lpd.append(lpd_i)

    ece = jnp.array(ece)
    lpd = jnp.array(lpd)

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
        grid, _, _ = grid_of_points(5000, [-4, 4], [-4, 4])
        grid_expanded = jnp.expand_dims(grid, batch_dims_to_expand + (-1,))
        predicted_class = model.predict(grid_expanded).logits.argmax(-1)
        for model_i in range(args.n_models):
            plot_dataset(
                x_test,
                y_test,
                grid,
                predicted_class[:, model_i],
                n_classes=args.n_classes,
                exp_name=exp_name + f"-model={model_i+1}",
            )

    if args.log_runtime:

        elbo_over_iters = jnp.array(elbo_over_iters) / args.train_size

        # compute the number of iterations needed for convergence
        n_iters_convergence = check_convergence_expfit(
            -1 * elbo_over_iters.T,
            n_iters_truncate=20,
            smooth=False,
            pct_of_maximum_thr=1e-1,
        )

        # initialize the model with n_model=1 for computing runtime
        batch_shape = ()
        x_train, y_train = next(iter(train_dataloader))
        x_test, y_test = next(iter(test_dataloader))

        batch_dims_to_expand = ()
        x_train_expanded = jnp.expand_dims(x_train, batch_dims_to_expand + (-1,))
        y_train_expanded = nn.one_hot(y_train, args.n_classes)

        x_test_expanded = jnp.expand_dims(x_test, batch_dims_to_expand + (-1,))
        y_test_expanded = y_test

        model = initialize_network(
            mnlr_keys,
            linear_keys,
            n_components,
            hidden_dims,
            batch_shape,
            args.dof_offset,
            args.inv_v_scale,
            x_dim,
            y_dim,
            args.add_mnlr_output,
            args.n_vb_iters_mnlr,
            args.scale_likelihood,
            args.scale_mnlr_betas,
            args.init_posterior_scale,
            args.sample_initial_betas,
            args.fixed_precision,
            args.likelihood_type,
            args.n_backwards_iters,
            cmix_optim_args,
            mnlr_optim_args,
        )

        time_start = time.time()
        # train the model while storing the ELBO, the training accuracy, and the test accuracy over the course of M steps
        _, _, _ = model.fit_vmp(
            x_train_expanded,
            y_train_expanded,
            n_m_steps=args.n_iters_runtime,
            compute_accuracy=False,
            x_test=x_test_expanded,
            y_test=y_test_expanded,
        )
        time_end = time.time()
        runtime_per_iter = (time_end - time_start) / args.n_iters_runtime

        print(
            f"Average total / per-iter runtime: {runtime_per_iter * jnp.nanmean(n_iters_convergence):.6f} / {runtime_per_iter:.6f}, Convergence step: {n_iters_convergence.mean():.0f}, "
        )

        fout = open(
            "./logging/" + exp_name + f"-runtimes" + ".txt",
            mode="a+",
        )
        for model_i in range(args.n_models):
            print(
                f"model={model_i+1}, n_steps={n_iters_convergence[model_i]:.0f}, runtime={runtime_per_iter * n_iters_convergence[model_i]:.3f}",
                file=fout,
            )
        fout.close()
