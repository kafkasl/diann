#!/bin/bash -e

#--debug True \
python3 ../src/main_diann.py \
--language english \
--threads 1 \
--folds 3 \
--model /home/hydra/projects/diann/results/trained_modelCRF \
--input_dir ../data/English_Training/debug/
