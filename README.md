# cavi-cmn


This repository contains code accompanying the preprint: ["Gradient-free variational learning with conditional mixture networks"](https://arxiv.org)

## Installation

To be able to run the code, first clone the repository and then change directory into the root directory.

```
$ git clone https://github.com/VersesTech/cavi-cmn.git
$ cd cavi-cmn
```

Install necessary dependencies using `pip`:

```
$ pip install -r requirements.txt
```

## Model Training and Evaluation

The code in this repository allows you to train the 4 different inference methods described in the paper (CAVI, BBVI, NUTS, or MLE) on 7 different UCI datasets, and additionally on the Pinwheel dataset.

#### Pinwheel Dataset
To run the pinwheels benchmarks, run the following commands that correspond to fitting each of the different inference algorithms:

CAVI
```
python cavi_cmn_eval_pinwheels.py --train_size=[desired_training_set_size] --test_size=[desired_test_set_size] --<cavi_specific_param>=[param_value] ... 
```

BBVI
```
python bbvi_cmn_eval_pinwheels.py --train_size=[desired_training_set_size] --test_size=[desired_test_set_size]  --<bbvi_specific_param>=[param_value] ... 
```

NUTS
```
python nuts_cmn_eval_pinwheels.py --train_size=[desired_training_set_size] --test_size=[desired_test_set_size]  --<nuts_specific_param>=[param_value] ... 
```

MLE
```
python mle_cmn_eval_pinwheels.py --train_size=[desired_training_set_size] --test_size=[desired_test_set_size]  --<mle_specific_param>=[param_value] ... 
```

Some hyperparameters / arguments are specific to the inference method and dataset (e.g. the `--num_chains` argument is specific to NUTS, and `--n_iters` is specific to BBVI and MLE).


#### UCI Datasets: Performance Plots
To reproduce the UCI performance and runtime results from the paper, run the following commands that correspond to each of the models:


CAVI
```
python cavi_cmn_eval_uci.py --data=[dataset_name] --train_size=[desired_training_set_size] --test_size=[desired_test_set_size] --<cavi_specific_param>=[param_value] ...
```

BBVI
```
python bbvi_cmn_eval_uci.py --data=[dataset_name] --train_size=[desired_training_set_size] --test_size=[desired_test_set_size]  --<bbvi_specific_param>=[param_value] ... 
```

NUTS
```
python nuts_cmn_eval_uci.py --data=[dataset_name] --train_size=[desired_training_set_size] --test_size=[desired_test_set_size]  --<nuts_specific_param>=[param_value] ...  
```

MLE
```
python mle_cmn_eval_uci.py --data=[dataset_name] --train_size=[desired_training_set_size] --test_size=[desired_test_set_size]  --<mle_specific_param>=[param_value] ... 
```

Replace [dataset_name] with one of the choices: `rice`, `waveform`, `statlog`, `breast_cancer`, `banknote`, `connectionist_bench`, or `iris`. The optional flags --log_metrics and --log_runtime can be used to log the performance or runtime results, respectively, to a .txt file in a `logging/` directory.

For instance, to evaluate and record the test accuracy, log predictive density, and expected calibration error of CAVI-CMN fit on 250 training examples from the Breast Cancer Diagnosis Dataset, where there are 20 linear experts in the conditional mixture layer, you can run

```
python cavi_cmn_eval_uci.py --data=breast_cancer --train_size=250 --max_train_size=400 --test_size=169 --num_components=20 --log_metrics
 ```

N.B.: the parameter `max_train_size` should be greater than or equal to the largest value of `train_size` you are using (and smaller than the total size of the dataset), and `max_train_size + test_size` must be upper bounded by the size of the dataset. The argument `max_train_size` determines (through the `train_size` parameter of `sklearn.model_selection.train_test_split`) a train/test split of the full dataset that ensures the same class frequencies that are in the original full dataset, are maintained across both the train and test sets. 

In order to compare the performance of differently trained algorithms on the same test set, we set max_train_size to be the same, dataset-specific value regardless of the free parameter `train_size` (which is upper bounded by `max_train_size`).

```
breast_cancer: max_train_size = 400
rice: max_train_size = 2560
waveform: max_train_size = 3840
statlog: max_train_size = 640
banknote: max_train_size = 1024
connectionist_bench: max_train_size = 128
```

Since we did not generate performance plots for the Iris dataset, but only computed its WAIC score for the different models, we have no suggested max_train_size for this dataset.

#### UCI Datasets: WAIC Table
To calculate the WAIC scores evaluated on each dataset, use the following commands, which are similar to the scripts used to evaluate the performance and runtime metrics in the previous section.


CAVI
```
python cavi_cmn_waic_uci.py --data=[dataset_name] --<cavi_specific_param>=[param_value] ...
```

BBVI
```
python bbvi_cmn_waic_uci.py --data=[dataset_name]  --<bbvi_specific_param>=[param_value] ... 
```

NUTS
```
python nuts_cmn_waic_uci.py --data=[dataset_name]  --<nuts_specific_param>=[param_value] ...  
```

MLE
```
python mle_cmn_waic_uci.py --data=[dataset_name] --<mle_specific_param>=[param_value] ... 
```

Each script trains and evaluates the WAIC score using the entirety of the chosen UCI dataset. Therefore, there are no arguments like `train_size` or `test_size` for the WAIC scripts, but there are of course model-specific hyperparameters.

