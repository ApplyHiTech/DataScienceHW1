import sys
import os.path
import time
import random
import findspark
import pyspark
from pyspark.sql import SQLContext

findspark.init()
sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)

DATA_DIR = "dac/split/"

HELP_PROMPT = """split.py
Usage:
    split.py FILENAME
"""

FULL_TEST_FILE = os.path.join(DATA_DIR, "test.txt")
TRAIN_5M_FILE = os.path.join(DATA_DIR, "train_5m.txt")
VALIDATION_2M_FILE = os.path.join(DATA_DIR, "validation_2m.txt")
TEST_3M_FILE = os.path.join(DATA_DIR, "test_3m.txt")

def train_split(train_dataset, seed=None):
    print("Train split")
    train_5m, validation_2m, test_3m = (
        train_dataset.randomSplit([0.5, 0.2, 0.3], seed=seed))

    print("Saving train 5M file to %s" % TRAIN_5M_FILE)
    train_5m.saveAsTextFile(TRAIN_5M_FILE)
    print("Saving validation 2M file to %s" % VALIDATION_2M_FILE)
    validation_2m.saveAsTextFile(VALIDATION_2M_FILE)
    print("Saving test 3M file to %s" % TEST_3M_FILE)
    test_3m.saveAsTextFile(TEST_3M_FILE)


def test_split(raw_dataset, seed=None):
    print("Test split")
    train, test = raw_dataset.randomSplit([0.2083333, 0.7916667], seed=seed)
    print("Saving test file to %s" % FULL_TEST_FILE)
    test.saveAsTextFile(FULL_TEST_FILE)
    return train, test

def main(filename):
    raw_dataset = sc.textFile(filename)
    random.seed(time.time())
    seed = int(random.random())

    full_train, test = test_split(raw_dataset, seed=seed)
    train_split(full_train, seed=seed)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Splitting %s" % sys.argv[1])
        main(sys.argv[1])
    else:
        print(HELP_PROMPT)
        sys.exit(0)