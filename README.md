# Data Science - Assignment 1
### Sean Herman, Daniel Reidler

## Setup
This project was developed in Python (2.7) and Spark 1.6.0 built with Hadoop 2.6. First, install the project's Python dependencies with pip:

    $ pip install -r requirements.txt

### Criteo Data
This project analyzes data from the [Kaggle's Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge). `config.py` must be updated to point to a directory, `DAC_FILES_PATH`, which includes the `train.txt` file included in the Criteo archive. `SPLIT_FILES_PATH`, `RESULTS_PATH`, `MODELS_PATH` should also be updated to storage locations for the train file splits, the results charts, and saved training models.

Run the `split.py` script to split Criteo's `train.txt` dataset into a `test.txt` set (approx. 38mm rows) and training set (approx. 10mm rows). This training set is further divided into `train_5m.txt`, `test_3m.txt`, and `validation_2m.txt`.

    $ ./split.py

## Summary Statistics Instructions

Open the hw1_summary_statistics.ipyn notebook.

Running the notebook will calculate the histograms for the integer and category features. Further, the notebook will also calculate the summary statistics for the integer features (mean, std, skewness, kurtosis).


## Classification Instructions
Once setup and Criteo Data splits are completed, the data analysis can be initiated through `classify.py`.

Train on `train_5m.txt` and make predictions for `test_3m.txt`:

    $ ./classify.py

Train on `train_5m.txt` and make predictions for `validation_2m.txt`:

    $ PY_ENV=validate ./classify.py

Train on `train_5m.txt` and make predictions for `test.txt` (38mm rows):

    $ PY_ENV=production ./classify.py
