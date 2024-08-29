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

def get_default_args() -> dict:
    """
    Returns the default optimization arguments.

    Returns:
        dict: The default optimization arguments.
    """
    return {"iters": 1, "lr": 1.0, "beta": 0.0}


def check_optim_args(optim_args: dict) -> dict:
    """
    Checks the optim_args dict for the correct keys and values.

    Args:
        optim_args (dict): The dictionary of optimization arguments.

    Returns:
        dict: The dictionary of optimization arguments.
    """
    if not isinstance(optim_args, dict):
        raise TypeError(f"optim_args must be a dict, got {type(optim_args)}")

    if "iters" not in optim_args:
        optim_args["iters"] = get_default_args()["iters"]

    if "lr" not in optim_args:
        optim_args["lr"] = get_default_args()["lr"]

    if "beta" not in optim_args:
        optim_args["beta"] = get_default_args()["beta"]
    elif optim_args["beta"] < 0.0:
        raise ValueError(f"beta must be non-negative, got {optim_args['beta']}")
    elif optim_args["beta"] == 1.0 and optim_args["iters"] != 1:
        # raise a warning and set iters to 1
        optim_args["iters"] = 1
        print("WARNING: beta = 1.0, setting iters = 1")

    return optim_args
