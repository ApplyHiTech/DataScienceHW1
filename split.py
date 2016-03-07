#!/usr/bin/env python2

import sys
import os.path
import time
import random
import config

COMPRESSION = "org.apache.hadoop.io.compress.GzipCodec"
sc = config.SPARK_CONTEXT


def train_split(train_dataset, seed=None):
    print("Splitting train into train, validation, test files")
    train_5m, validation_2m, test_3m = (
        train_dataset.randomSplit([0.5, 0.2, 0.3], seed=seed))

    print("Saving train file to %s" % config.SPLIT_TRAIN_PATH)
    train_5m.saveAsTextFile(config.SPLIT_TRAIN_PATH,
                            compressionCodecClass=COMPRESSION)

    print("Saving validation file to %s" % config.SPLIT_VALIDATION_PATH)
    validation_2m.saveAsTextFile(config.SPLIT_VALIDATION_PATH,
                                 compressionCodecClass=COMPRESSION)

    print("Saving test file to %s" % config.SPLIT_TRAIN_TEST_PATH)
    test_3m.saveAsTextFile(config.SPLIT_TRAIN_TEST_PATH,
                           compressionCodecClass=COMPRESSION)

    return train_5m, validation_2m, test_3m


def test_split(raw_dataset, seed=None):
    print("Splitting source into train and test files")
    train, test = raw_dataset.randomSplit([0.2083333, 0.7916667], seed=seed)

    print("Saving test file to %s" % config.SPLIT_TEST_PATH)
    test.saveAsTextFile(config.SPLIT_TEST_PATH,
                        compressionCodecClass=COMPRESSION)

    return train, test


HELP_PROMPT = ("""split.py

Usage:
    split.py [FILE]

Options:
FILE    The source file to split (default=%s)
""" % config.FULL_TRAIN_PATH)


def main(filename):
    print("Splitting %s" % filename)

    raw_dataset = sc.textFile(filename)
    random.seed(time.time())
    seed = int(random.random())

    full_train, test = test_split(raw_dataset, seed=seed)
    train_split(full_train, seed=seed)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(config.FULL_TRAIN_PATH)
    elif sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(HELP_PROMPT)
        sys.exit(0)
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print(HELP_PROMPT)
        sys.exit(1)
