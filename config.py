import os
import findspark

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

try:
    ENV = os.environ["PY_ENV"]
except KeyError:
    ENV = "debug"

DEBUG = (ENV == "debug")
TEST = (ENV == "test")
VALIDATE = (ENV == "validate")
PROD = (ENV == "production")

findspark.init()
conf = SparkConf().setAppName("dshw1")
SPARK_CONTEXT = SparkContext(conf=conf)
SPARK_SQL_CONTEXT = SQLContext(SPARK_CONTEXT)

DAC_FILES_PATH = "dac"
FULL_TRAIN_PATH = os.path.join(DAC_FILES_PATH, "train.txt")
DEBUG_PATH = os.path.join(DAC_FILES_PATH, "small-train.txt")
# DEBUG_PATH = os.path.join(DAC_FILES_PATH, "very-small-train.txt")

SPLIT_FILES_PATH = os.path.join(DAC_FILES_PATH, "split")
RESULTS_PATH = os.path.join("results")
MODELS_PATH = os.path.join("models")

SPLIT_TEST_PATH = os.path.join(SPLIT_FILES_PATH, "test.txt")
SPLIT_TRAIN_TEST_PATH = os.path.join(SPLIT_FILES_PATH, "test_3m.txt")
SPLIT_TRAIN_PATH = os.path.join(SPLIT_FILES_PATH, "train_5m.txt")
SPLIT_VALIDATION_PATH = os.path.join(SPLIT_FILES_PATH, "validation_2m.txt")


def maybe_make_path(path):
    if not os.path.exists(path):
        print("Creating directory %s" % path)
        os.mkdir(config.SPLIT_FILES_PATH)
