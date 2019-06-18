#!/bin/bash
# Prepare dataset
#wget -O data/val2017.zip http://images.cocodataset.org/zips/val2017.zip
#unzip data/val2017.zip -d data/MSCOCO

# Clean files
rm -rf ./output
rm -f output.tar

mkdir output
mkdir output/pretrain
mkdir output/pruned
mkdir output/retrain

# Run weight pruning
python weight_pruning_adain.py

tar -czvf output.tar output

