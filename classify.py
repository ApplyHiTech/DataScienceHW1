#!/usr/bin/env python2

import config
import summary
import etl

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from dataset import CriteoDataSets

sc = config.SPARK_CONTEXT
sqlc = config.SPARK_SQL_CONTEXT
CAT_COLUMNS = ["C8", "C9", "C14", "C17", "C20", "C22", "C23", "C25"]
LR_MAX_ITER = 10
LR_REG_PARAM = 0.01


def prepare(data, cat_columns):
    cat_counts = {
        name: counts for (name, counts)
        in summary.cat_column_counts_iter(data, cat_columns)
    }

    # TODO: cleanup mean/other stat funcs
    int_means = {
        name: mean for (name, mean)
        in summary.integer_column_mean_iter(data)
    }

    return etl.transform_train(data, int_means, cat_columns, cat_counts)


def train_model(df):
    lr = LogisticRegression(maxIter=LR_MAX_ITER, regParam=LR_REG_PARAM)
    return lr, lr.fit(df)


def predict(test, int_means, cat_rates, scaler, model):
    df = etl.transform_test(test, int_means, cat_rates, scaler)
    return model.transform(df)


def evaluate(df):
    lr = LogisticRegression()
    grid = (
        ParamGridBuilder()
        .baseOn({lr.labelCol: "label"})
        .baseOn([lr.predictionCol, "prediction"])
        .addGrid(lr.regParam, [1.0, 2.0])
        .addGrid(lr.maxIter, [1, 10])
        .build())
    evaluator = BinaryClassificationEvaluator()
    # cv = CrossValidator(estimator=lr, evaluator=evaluator)
    cv = CrossValidator(estimator=lr, estimatorParamMaps=grid,
                        evaluator=evaluator)
    cv_model = cv.fit(df)
    print evaluator.evaluate(cv_model.transform(df))


def main():
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

    df, int_means, cat_rates, scaler = prepare(train, CAT_COLUMNS)
    df.show()

    lr, model = train_model(df)

    # test instead of validation?
    validation_transformed = etl.transform_test(validation, int_means,
                                                cat_rates, scaler)
    # predictions = predict(validation, int_means, cat_rates, scaler, model)
    # predictions.select(
    #     ["features", "label", "prediction", "probability"]
    # ).show()

    evaluate(validation_transformed)


if __name__ == '__main__':
    main()
