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
