#!/usr/bin/env bash

PYTHON_MODULES="config.py,etl.py,summary.py,evaluate.py,criteodata.py"
DATA_FILES="dac/small-train.txt,dac/split/test_3m.txt,dac/split/train_5m.txt,dac/split/validation_2m.txt"

spark-submit --master "spark://jazzhands:7077" \
  --py-files $PYTHON_MODULES \
  --files $DATA_FILES \
  ./classify.py