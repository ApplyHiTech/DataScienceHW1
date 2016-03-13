#!/usr/bin/env python2

import time
import random
import config
import evaluate

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.linalg import Vectors
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


def evaluate_charts(predictions, model_name=""):
    labels = evaluate.labels_array(predictions)
    scores = evaluate.scores_array(predictions)
    fpr, tpr, thresholds = evaluate.roc(labels, scores)
    roc_auc = evaluate.roc_auc(labels, scores)
    evaluate.plot_roc(fpr, tpr, roc_auc, model_name=model_name)


def save_model(model, name="model"):
    model.save(os.path.join(config.MODELS_PATH, name))


def train_validate(train, validation):
    raise NotImplementedError()


def train_predict(test, model, model_name="Model"):
    predictions = model.transform(test)
    predictions.select(["features", "label", "prediction"]).show()
    return predictions


def evaluate_roc_auc(predictions, sqlc):
    raw = scores_and_labels(predictions, sqlc)
    evaluator = BinaryClassificationEvaluator()
    return evaluator.evaluate(raw)


def scores_and_labels(predictions, sqlc):
    raw = predictions.map(lambda r: (Vectors.dense(1.0 - r["prediction"],
                                                   r["prediction"]),
                                     r["label"]))
    return sqlc.createDataFrame(raw, ["rawPrediction", "label"])


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

    # Only the RandomForestClassifier is used for production scoring
    if not config.PROD:
        _, lr_model = train_logistic(df)

    _, rf_model = train_random_forest(df)

    if config.DEBUG:
        df = transform_test(data.debug, int_means, cat_rates, scaler)
        predictions = train_predict(df, lr_model, "LogisticRegression-debug")
        predictions.show(100)
        auc = evaluate_roc_auc(predictions, sqlc)
        print("\n\nArea Under ROC Curve: %0.3f\n\n" % auc)
        predictions = train_predict(df, rf_model,
                                    "RandomForestClassifier-debug")
        predictions.show(100)
        auc = evaluate_roc_auc(predictions, sqlc)
        print("\n\nArea Under ROC Curve: %0.3f\n\n" % auc)

    elif config.VALIDATE:
        df = transform_test(data.validation_2m, int_means, cat_rates, scaler)
        predictions = train_predict(df, lr_model,
                                    "LogisticRegression-validation_2m")
        evaluate_charts(predictions, "LogisticRegression-validation_2m")
        predictions = train_predict(df, rf_model,
                                    "RandomForestClassifier-validation_2m")
        evaluate_charts(predictions, "RandomForestClassifier-validation_2m")

    elif config.TEST:
        df = transform_test(data.test_3m, int_means, cat_rates, scaler)
        predictions = train_predict(df, lr_model,
                                    "LogisticRegression-test_3m")
        evaluate_charts(predictions, "LogisticRegression-test_3m")
        predictions = train_predict(df, rf_model,
                                    "RandomForestClassifier-test_3m")
        evaluate_charts(predictions, "LogisticRegression-test_3m")

        df = transform_test(data.validation_2m, int_means, cat_rates, scaler)
        predictions = train_predict(df, lr_model,
                                    "LogisticRegression-validation_2m")
        evaluate_charts(predictions, "LogisticRegression-validation_2m")
        predictions = train_predict(df, rf_model,
                                    "RandomForestClassifier-validation_2m")
        evaluate_charts(predictions, "RandomForestClassifier-validation_2m")

    elif config.PROD:
        df = transform_test(data.test, int_means, cat_rates, scaler)
        predictions = train_predict(df, rf_model, "RandomForestClassifier-test")
        predictions.show(100)
        auc = evaluate_roc_auc(predictions, sqlc)
        print("\n\nArea Under ROC Curve: %0.3f\n\n" % auc)

    sc.stop()


if __name__ == '__main__':
    main()
