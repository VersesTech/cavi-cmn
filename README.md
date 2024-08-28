# cavi-cmn

This repository contains the companion code for the paper [Gradient-free variational learning with conditional mixture networks][hyperlink to arxiv preprint].

## Installation

To be able to run the code, you will need to setup the dependent python packages by running the following command:

```
$ pip install -r requirements.txt
```

## Model Training and Evaluation

In this paper we evaluate 4 different models on 6 UCI datasets. To reproduce the results from the paper, run the following commands that correspond to each of the models:
: rice, waveform, breast_cancer, statlog, banknote, 

cavi
```
python cavi_cmn_eval_uci.py --data=[dataset_name]
```

bbvi
```
python bbvi_cmn_eval_uci.py --data=[dataset_name]
```

nuts
```
python nuts_cmn_eval_uci.py --data=[dataset_name]
```

mle
```
python mle_cmn_eval_uci.py --data=[dataset_name]
```

Replace the [dataset_name] with one of the choices: rice, waveform, statlog, breast_cancer, banknote, connectionist_bench.
