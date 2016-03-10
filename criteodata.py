import config

# Label - Target variable that indicates if an ad was clicked (1) or not (0).
# I1-I13 - A total of 13 columns of integer features (mostly count features).
# C1-C26 - A total of 26 columns of categorical features. The values of these
# features have been hashed onto 32 bits for anonymization purposes.
FIRST_CAT_INDEX = 14  # Label + I13 = index 14 for first categorical feature


class CriteoData(object):
    COLUMN_NAMES = (
        ["label"] +
        ["%s%d" % ("I", i) for i in xrange(1, 14)] +
        ["%s%d" % ("C", i - 13) for i in xrange(14, 40)]
    )

    def __init__(self, spark_cotext, sql_context, file=None, df=None, rdd=None):
        self._sc = spark_cotext
        self._sqlc = sql_context
        self._file_path = file
        self._rdd = rdd
        self._df = df
        if self._file_path is None and self._df is None and self._rdd is None:
            raise ValueError("One of file, df, or rdd must be provided")
        self._integer_column_names = None
        self._categorical_column_names = None

    @staticmethod
    def _convert_value(index, value):
        if index < FIRST_CAT_INDEX:
            return int(value) if value else None
        else:
            return value if value else None

    @classmethod
    def _convert_line(cls, line):
        return [cls._convert_value(i, value)
                for i, value in enumerate(line.split("\t"))]

    @property
    def rdd(self):
        if self._rdd is None and self._file_path:
            self._rdd = (
                self._sc.textFile(self._file_path)
                .map(self._convert_line))
        elif self._rdd and self._df is not None:
            self._rdd = self._df.rdd
        elif self._rdd is None:
            raise ValueError("One of rdd or file must be provided")
        return self._rdd

    @property
    def df(self):
        if self._df is None:
            self._df = self._sqlc.createDataFrame(self.rdd,
                                                  CriteoData.COLUMN_NAMES)
        return self._df

    @property
    def integer_column_names(self):
        if self._integer_column_names is None:
            self._integer_column_names = [col_name for col_name
                                          in self.df.columns
                                          if col_name[0] == "I"]
        return self._integer_column_names

    @property
    def categorical_column_names(self):
        if self._categorical_column_names is None:
            self._categorical_column_names = [col_name for col_name
                                              in self.df.columns
                                              if col_name[0] == "C"]
        return self._categorical_column_names

    @property
    def integer_columns(self):
        return self.df.select(self.integer_column_names())

    @property
    def categorical_columns(self):
        return self.df.select(self.categorical_column_names())


class CriteoDataSets(object):
    def __init__(self, spark_cotext, sql_context):
        self.sc = spark_cotext
        self.sqlc = sql_context

        # Full, original training set
        self.train = CriteoData(self.sc, self.sqlc, config.FULL_TRAIN_PATH)

        # Training set splits
        self.test = CriteoData(self.sc, self.sqlc,
                               config.SPLIT_TEST_PATH)
        self.test_3m = CriteoData(self.sc, self.sqlc,
                                  config.SPLIT_TRAIN_TEST_PATH)
        self.train_5m = CriteoData(self.sc, self.sqlc, config.SPLIT_TRAIN_PATH)
        self.validation_2m = CriteoData(self.sc, self.sqlc,
                                        config.SPLIT_VALIDATION_PATH)
        self.debug = CriteoData(self.sc, self.sqlc, config.DEBUG_PATH)
