import summary

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler

RAW_FEATURES_COL = "raw_features"
FEATURES_COL = "features"


def make_value_rate_udf(denominator):
    return udf(lambda num: num / denominator if num else 0.0, DoubleType())


# dictionary of {value: rate}
def make_resolve_rate_udf(rates):
    return udf(lambda hash_value: rates.get(hash_value, 0.0), DoubleType())


def cat_rate_dict(df):
    return {row.value: float(row.rate) for row in df.collect()}


def cat_rate_col_name(col_name):
    return "%s_rate" % col_name


def df_with_rate(df, col_name, val_rates):
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


def cat_features(data, column_names, cat_counts, total_row_count):
    df = data.df
    rate_udf = make_value_rate_udf(total_row_count)
    cat_rates = {}
    for col_name in column_names:
        cat_count = cat_counts[col_name]
        rate_df = cat_rate_df(cat_count, rate_udf)
        # Convert list of [Row(value, rate), ...] results to dict keyed on value
        cat_rates[col_name] = cat_rate_dict(rate_df)
        # Add category rates for the corresponding category values
        df = df_with_rate(df, col_name, cat_rates[col_name])
    return df, cat_rates


def assemble_vector(df, col_names):
    assembler = VectorAssembler(inputCols=col_names, outputCol=RAW_FEATURES_COL)
    return assembler.transform(df)


def scale_train(df, col_names):
    df = assemble_vector(df, col_names)
    # df.select(RAW_FEATURES_COL).show()

    scaler = StandardScaler(inputCol=RAW_FEATURES_COL, outputCol=FEATURES_COL)
    scaler_model = scaler.fit(df)

    df = scaler_model.transform(df)
    # df.select([RAW_FEATURES_COL, FEATURES_COL]).show()

    return df, scaler_model


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
    row_count = summary.row_count(data)

    df, cat_rates = cat_features(data, cat_column_names, cat_counts, row_count)

    df, scaler = scale_train(fill_null_ints(df, int_means),
                             feature_col_names(data, cat_column_names))

    return (
        convert_label(df),
        int_means,
        cat_rates,
        scaler
    )


def scale_test(df, col_names, scaler):
    return scaler.transform(assemble_vector(df, col_names))


def transform_test(data, int_means, cat_rates, scaler):
    cat_column_names = cat_rates.keys()
    column_names = feature_col_names(data, cat_column_names)

    # Replace categorical values with rate values
    df = data.df
    for col_name in cat_column_names:
        df = df_with_rate(df, col_name, cat_rates[col_name])

    # Replace null integer values with mean values
    df = fill_null_ints(df, int_means)

    return convert_label(scale_test(df, column_names, scaler))
