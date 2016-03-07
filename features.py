from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf


def categorical_counts_rate(categorical_counts, total_row_count):
    for col_name, count_df in categorical_counts.iteritems():
        pass
        # count_df.map(lambda row: row.count)
        # pyspark.sql.functions.udf(f, returnType=StringType)


# Join categorical columns with value counts
def join_column_counts(data, categorical_counts):
    cat_columns_names = []
    df = data.df
    for col_name in data.categorical_column_names:
        count_col_name = "%s_count" % col_name
        cat_columns_names.append(cat_columns_names)
        df = (
            df.join(categorical_counts[col_name], col_name, "outer")
            .withColumnRenamed("count", count_col_name))
    return df


# calculate categorical proportion columns
def count_proportions(counts_df, rows_num):
    pass

# replace nulls with 0? average?

# replace integer column nulls with average or 0?
# make feature vectors
