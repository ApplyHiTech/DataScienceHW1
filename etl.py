from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from summary import row_count

# RAW_FEATURES_COL = "features"
RAW_FEATURES_COL = "raw_features"
FEATURES_COL = "features"


def make_value_rate_udf(denominator):
    return udf(lambda num: num / denominator if num else 0.0, DoubleType())


# dictionary of {value: rate}
def make_resolve_rate_udf(rates):
    return udf(lambda hash_value: rates.get(hash_value, 0.0), DoubleType())


def cat_rate_dict(df):
    return {row.value: row.rate for row in df.collect()}
    # return {row.value: float(row.rate) for row in df.collect()}


def cat_rate_col_name(col_name):
    return "%s_rate" % col_name


def cat_value_rate_col(df, col_name, val_rates):
    val_rate_udf = make_resolve_rate_udf(val_rates)
    rate_col_name = cat_rate_col_name(col_name)
    return df.withColumn(rate_col_name, val_rate_udf(df[col_name]))


# calculate categorical proportion columns
# category_count is a DataFrame with columns ([COLUMN_NAME], count)
def cat_rate_df(cat_count, rate_udf):
    col_name = cat_count.columns[0]
    return cat_count.select([
        cat_count[col_name].alias("value"),
        rate_udf(cat_count["count"].cast(DoubleType())).alias("rate")
    ])


def cat_value_rates(column_names, cat_counts, total_rows):
    rate_udf = make_value_rate_udf(total_rows)
    return {
        col_name: cat_rate_dict(cat_rate_df(cat_counts[col_name], rate_udf))
        for col_name in column_names
    }


def assemble_vector(df, inputCols, outputCol):
    assembler = VectorAssembler(inputCols=inputCols, outputCol=outputCol)
    return assembler.transform(df)


def train_scaler(df, inputCol, outputCol):
    scaler = MinMaxScaler(inputCol=inputCol, outputCol=outputCol)
    return scaler.fit(df)


def fill_null_ints(df, int_means):
    return df.fillna(int_means)


def feature_col_names(data, cat_column_names):
    return (
        data.integer_column_names +
        [cat_rate_col_name(col_name) for col_name in cat_column_names]
    )


def convert_label(df):
    return df.select([df["label"].cast(DoubleType()), df["features"]])


def transform_train(data, int_means, cat_column_names, cat_counts):
    feature_cols = feature_col_names(data, cat_column_names)
    df = data.df

    rows_num = row_count(data)
    cat_rates = cat_value_rates(cat_column_names, cat_counts, rows_num)

    # Add rate columns for each categorical column
    for col_name, rates in cat_rates.items():
        df = cat_value_rate_col(df, col_name, rates)

    df = fill_null_ints(df, int_means)
    df = assemble_vector(df, feature_cols, RAW_FEATURES_COL)
    scaler = train_scaler(df, RAW_FEATURES_COL, FEATURES_COL)
    df = scaler.transform(df)
    df = convert_label(df)

    return (df, int_means, cat_rates, scaler)


# def scale_test(df, col_names, scaler):
#     return scaler.transform(assemble_vector(df, col_names))


def transform_test(data, int_means, cat_rates, scaler):
    cat_column_names = cat_rates.keys()
    feature_cols = feature_col_names(data, cat_column_names)

    # Replace categorical values with rate values
    df = data.df
    for col_name in cat_column_names:
        df = cat_value_rate_col(df, col_name, cat_rates[col_name])

    # Replace null integer values with mean values
    df = fill_null_ints(df, int_means)
    df = assemble_vector(df, feature_cols, RAW_FEATURES_COL)
    df = scaler.transform(df)
    df = convert_label(df)

    return df
