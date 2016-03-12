#!/usr/bin/env python2

import time
import random
import config
import evaluate

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from criteodata import CriteoDataSets
from etl import transform_train, transform_test
from summary import cat_column_counts_iter, integer_column_mean_iter


CAT_COLUMNS = ["C8", "C9"]
LR_MAX_ITER = 10
LR_REG_PARAM = 0.01
random.seed(time.time())
config.maybe_make_path(config.MODELS_PATH)


def prepare(data, cat_columns):
    cat_counts = {
        name: counts for (name, counts)
        in cat_column_counts_iter(data, cat_columns)
    }

    # TODO: cleanup mean/other stat funcs
    int_means = {name: mean for (name, mean) in integer_column_mean_iter(data)}

    return transform_train(data, int_means, cat_columns, cat_counts)


def train_logistic(df):
    lr = LogisticRegression(maxIter=LR_MAX_ITER, regParam=LR_REG_PARAM)
    return lr, lr.fit(df)


def train_random_forest(df):
    stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
    si_model = stringIndexer.fit(df)
    td = si_model.transform(df)
    rf = RandomForestClassifier(numTrees=3, maxDepth=2, labelCol="indexed",
                                seed=int(random.random()))
    return rf, rf.fit(td)


def evaluate_predictions(predictions, model_name=""):
    labels = evaluate.labels_array(predictions)
    scores = evaluate.scores_array(predictions)
    fpr, tpr, thresholds = evaluate.roc(labels, scores)
    roc_auc = evaluate.roc_auc(labels, scores)
    evaluate.plot_roc(fpr, tpr, roc_auc, model_name=model_name)


def save_model(model, name="model"):
    model.save(os.path.join(config.MODELS_PATH, name))


def train_validate(train, validation):
    raise NotImplementedError()


def train_models(df):
    lr, lr_model = train_logistic(df)
    rf, rf_model = train_random_forest(df)

    return lr_model, rf_model


def train_predict(test, model, model_name="Model"):
    predictions = model.transform(transformed)
    predictions.select(["features", "label", "prediction"]).show()
    evaluate_predictions(predictions, model_name)


def prod_predict(test):
    raise NotImplementedError()


# Given Global Variables that represent the feature columns we wish to include
# in the dataset.
# CAT_COLUMNS = ["C8", "C9"]
# LR_MAX_ITER = 10
# LR_REG_PARAM = 0.01
#
# /1/ - Prepare training data
# /1.1/ - Compute aggregates, mean, and rates for each distinct value in each
#           category
# /1.2/ - Integer Columns, replaces nulls with the column average.
#       - Categorical Columns, we create a count of each value, and compute
#           its rate within that column
#       - Creates, a column that consists of a vector that consists of all the
#           feature columns.
#
def main():
    sc = config.SPARK_CONTEXT
    sqlc = config.SPARK_SQL_CONTEXT

    data = CriteoDataSets(sc, sqlc)

    if config.DEBUG:
        train = data.debug
    else:
        train = data.train_5m

    df, int_means, cat_rates, scaler = prepare(train, CAT_COLUMNS)
    lr_model, rf_model = train_models(df)

    if config.DEBUG:
        df = transform_test(data.debug, int_means, cat_rates, scaler)
        train_predict(df, lr_model, "LogisticRegression-validation")
        train_predict(df, rf_model, "RandomForestClassifier-validation")

    elif config.VALIDATE:
        df = transform_test(data.validation_2m, int_means, cat_rates, scaler)
        train_predict(df, lr_model, "LogisticRegression-validation")
        train_predict(df, rf_model, "RandomForestClassifier-validation")

    elif config.TEST:
        df = transform_test(data.test_3m, int_means, cat_rates, scaler)
        train_predict(df, lr_model, "LogisticRegression-test")
        train_predict(df, rf_model, "RandomForestClassifier-test")

        df = transform_test(data.validation_2m, int_means, cat_rates, scaler)
        train_predict(df, lr_model, "LogisticRegression")
        train_predict(df, rf_model, "RandomForestClassifier")

    elif config.PROD:
        prod_predict(data.test)

    sc.stop()


if __name__ == '__main__':
    main()
