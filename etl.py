from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler


def make_value_rate_udf(denominator):
    return udf(lambda count: count / denominator if count else 0.0, DoubleType())


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


def standard_scale_column(df, col_name):
    output_col_name = "%s%s" % (col_name, "_scaled")
    scaler = StandardScaler(inputCol=col_name, outputCol=output_col_name)
    return scaler.fit(df)


def scale_train(df, column_names):
    # Vectorize
    assembler = VectorAssembler(inputCols=column_names, outputCol="features")
    df = assembler.transform(df)

    # Scale
    scaler = StandardScaler(inputCol="features", outputCol="features_scaled")
    scaler_model = scaler.fit(df)

    return scaler_model.transform(df), scaler_model


def feature_col_names(data, cat_column_names):
    return (
        data.integer_column_names +
        [cat_rate_col_name(col_name) for col_name in cat_column_names]
    )


# data = CriteoData
# cat_features = cat columns to include
def transform_train(data, int_means, cat_column_names, cat_counts, total_row_count):
    df, cat_rates_map = cat_features(data, cat_column_names,
                                               cat_counts, total_row_count)

    col_names = feature_col_names(data, cat_column_names)

    df, scaler = scale_train(df.fillna(int_means), col_names)

    return (
        df.select(["label", "features", "features_scaled"]),
        scaler,
        cat_rates_map
    )
