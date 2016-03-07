#!/usr/bin/env python2

import config
import summary
import etl

from dataset import CriteoDataSets

sc = config.SPARK_CONTEXT
sqlc = config.SPARK_SQL_CONTEXT
CATEGORICAL_COLUMNS = ["C9", "C20"]


def prepare(data, cat_columns):
    # TODO: only calculate counts for cat_columns
    categorical_counts = {
        name: counts for (name, counts)
        in summary.cat_column_counts_iter(data, cat_columns)
    }
    # b_categorical_counts = sc.broadcast(categorical_counts)

    # TODO: cleanup mean/other stat funcs
    integer_column_means = {
        name: mean for (name, mean)
        in summary.integer_column_mean_iter(data)
    }
    # b_integer_column_means = sc.broadcast(integer_column_means)

    total_rows_num = summary.row_count(data)
    # b_total_rows_num = sc.broadcast(total_rows_num)

    return etl.transform_train(data, integer_column_means, cat_columns,
                               categorical_counts, total_rows_num)
    # return etl.transform_train(data, b_integer_column_means.value, cat_columns,
    #                            b_categorical_counts.value,
    #                            b_total_rows_num.value)


def train():
    pass


def test():
    pass


def main():
    data = CriteoDataSets(sc, sqlc)

    if config.DEBUG:
        train = data.debug
    else:
        train = data.train_5m

    df, sscaler, cat_rates = prepare(train, CATEGORICAL_COLUMNS)
    df.show()
    # broadcast_cat_rates = sc.broadcast(cat_rates)
    # broadcast_sscaler = sc.broadcast(sscaler)

    # train()
    # test()


if __name__ == '__main__':
    main()
