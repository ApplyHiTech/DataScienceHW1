#!/usr/bin/env python2

import config

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from criteodata import CriteoDataSets
from etl import transform_train, transform_test
from summary import cat_column_counts_iter, integer_column_mean_iter


CAT_COLUMNS = ["C8", "C9"]
LR_MAX_ITER = 10
LR_REG_PARAM = 0.01


def prepare(data, cat_columns):
    cat_counts = {
        name: counts for (name, counts)
        in cat_column_counts_iter(data, cat_columns)
    }

    # TODO: cleanup mean/other stat funcs
    int_means = {
        name: mean for (name, mean)
        in integer_column_mean_iter(data)
    }

    return transform_train(data, int_means, cat_columns, cat_counts)


def train_model(df):
    lr = LogisticRegression(maxIter=LR_MAX_ITER, regParam=LR_REG_PARAM)
    return lr, lr.fit(df)


def predict(df, model):
    return model.transform(df)


#Given Global Variables that represent the feature columns we wish to include
#in the dataset.
# CAT_COLUMNS = ["C8", "C9"]
# LR_MAX_ITER = 10
# LR_REG_PARAM = 0.01
#
# /1/ - Prepare training data
# /1.1/ - Compute aggregates, mean, and rates for each distinct value in each category
# /1.2/ - Integer Columns, replaces nulls with the column average.
#       - Categorical Columns, we create a count of each value, and compute its rate within that column
#       - Creates, a column that consists of a vector that consists of all the feature columns.
#
def main():
    sc = config.SPARK_CONTEXT
    sqlc = config.SPARK_SQL_CONTEXT

    data = CriteoDataSets(sc, sqlc)

    if config.DEBUG:
        train = data.debug
        validation = data.debug
        test = data.debug
    elif config.TEST:
        train = data.train_5m
        validation = data.validation_2m
        test = data.test_3m
    elif config.PROD:
        raise NotImplementedError()

    # /1/
    df, int_means, cat_rates, scaler = prepare(train, CAT_COLUMNS)
    df.show()

    lr, model = train_model(df)

    # test instead of validation?
    validation_transformed = transform_test(validation, int_means,
                                            cat_rates, scaler)

    predictions = predict(validation_transformed, model)
    predictions.select(
        ["features", "label", "prediction", "probability"]
    ).show(100)

    sc.stop()


if __name__ == '__main__':
    main()
