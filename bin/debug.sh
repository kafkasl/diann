#!/bin/bash -e

python3 ../src/main_diann.py \
--debug True \
--language english \
--threads 1 \
--folds 2 \
--input_dir ../data/English_Training/Annotated/
