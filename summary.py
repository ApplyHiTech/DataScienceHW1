#!/usr/bin/env python2

from pyspark.sql.functions import kurtosis, skewness, avg, stddev


def label_histogram(data):
    return data.rdd.map(lambda row: row[0]).histogram([0, 1, 2])


def is_integer_col_num(col_num):
    return col_num > 1 and col_num < 15


def is_label_col_num(col_num):
    return col_num == 1


def is_categorical_col_num(col_num):
    return col_num > 15


def int_column_histogram(column, buckets=10):
    return column.histogram(buckets)


def rdd_column_iter(data):
    column_count = len(data.rdd.take(1)[0])
    for column_num in range(column_count):
        yield data.rdd.map(lambda row: row[column_num])


def int_columns_histograms_iter(data):
    for i, col in enumerate(rdd_column_iter(data)):
        col_num = i + 1
        if is_integer_col_num(col_num):
            yield col_num, int_column_histogram(col)
    # TODO: display graph of histogram
    # TODO: better buckets for histogram (smart sub-dividing)
        # sum the counts
        # max of the counts
        # if  > 25%


def cat_column_key_counts(data, col_name):
    import config

    df = data.df.groupBy(col_name).count()
    return df if not config.DEBUG else df.orderBy("count", ascending=False)
    # return data.df.groupBy(col_name).count().orderBy("count", ascending=False)


def cat_column_counts_iter(data, col_names=None):
    col_names = (
        col_names if col_names is not None else
        data.categorical_column_names)
    for col_name in col_names:
        yield col_name, cat_column_key_counts(data, col_name)


def make_col_func(col_func):
    def wrapped(data, col):
        result_col = col_func(col)
        result = data.df.select(result_col.alias("calc")).collect()[0]
        return result.calc
    return wrapped


calc_kurtosis = make_col_func(kurtosis)
calc_skewness = make_col_func(skewness)
calc_mean = make_col_func(avg)
calc_stddev = make_col_func(stddev)
calc_keys = ["kurtosis", "skewness", "mean", "stddev"]


def integer_column_stats_iter(data):
    for col_name in data.integer_column_names:
        col = data.df[col_name]
        calcs = (calc_kurtosis(data, col), calc_skewness(data, col),
                 calc_mean(data, col), calc_stddev(data, col))
        yield col_name, dict(zip(calc_keys, calcs))


def integer_column_mean_iter(data):
    for col_name in data.integer_column_names:
        col = data.df[col_name]
        yield col_name, calc_mean(data, col)


def column_distinct_count(data, column_name):
    return data.df.select(column_name).distinct().count()


def row_count(data):
    return data.df.count()
