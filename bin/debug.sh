#!/bin/bash -e

python3 ../src/bio_nltk.py \
--debug True \
--language english \
--threads 1 \
--input_dir ../data/English_Training/Annotated/
