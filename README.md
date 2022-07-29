## Seq2Seq with Diverse Decoding Algorithm

This is a Seq2Seq model with multiple diverse decoding algorithms.

Datasets:

* `dataset1`: [Quora](https://data.deepai.org/quora_question_pairs.zip)

Models:

* `model1`: Seq2Seq

### Data Process

```shell
PYTHONPATH=. python dataprocess/process.py
```

### Unit Test

* for loader

```shell
PYTHONPATH=. python loaders/loader1.py
```

* for module

```shell
PYTHONPATH=. python modules/module1.py
```

### Main Process

```shell
python main.py
```

You can change the config either in the command line or in the file `utils/parser.py`

Here are the examples:

```shell
# train
python main.py \
    --mode train
```

```shell
# test with temperature_sampling
python main.py \
    --mode test \
    --decoding_algorithm temperature_sampling \
    --T 1e-13
```

```shell
# test with top_k_sampling
python main.py \
    --mode test \
    --decoding_algorithm top_k_sampling \
    --K 10
```

```shell
# test with top_p_sampling
python main.py \
    --mode test \
    --decoding_algorithm top_p_sampling \
    --P 0.3
```
