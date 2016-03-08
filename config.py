import os
import findspark
import pyspark

try:
    ENV = os.environ["PY_ENV"]
except KeyError:
    ENV = "debug"

DEBUG = (ENV == "debug")
TEST = (ENV == "test")
PROD = (ENV == "production")

findspark.init()
SPARK_CONTEXT = pyspark.SparkContext()
SPARK_SQL_CONTEXT = pyspark.sql.SQLContext(SPARK_CONTEXT)

DAC_FILES_PATH = "dac"
FULL_TRAIN_PATH = os.path.join(DAC_FILES_PATH, "train.txt")
DEBUG_PATH = os.path.join(DAC_FILES_PATH, "small-train.txt")
# DEBUG_PATH = os.path.join(DAC_FILES_PATH, "very-small-train.txt")

SPLIT_FILES_PATH = os.path.join(DAC_FILES_PATH, "split")

if not os.path.exists(SPLIT_FILES_PATH):
    print("Creating split files directory %s" % SPLIT_FILES_PATH)
    os.mkdir(SPLIT_FILES_PATH)

SPLIT_TEST_PATH = os.path.join(SPLIT_FILES_PATH, "test.txt")
SPLIT_TRAIN_TEST_PATH = os.path.join(SPLIT_FILES_PATH, "test_3m.txt")
SPLIT_TRAIN_PATH = os.path.join(SPLIT_FILES_PATH, "train_5m.txt")
SPLIT_VALIDATION_PATH = os.path.join(SPLIT_FILES_PATH, "validation_2m.txt")
