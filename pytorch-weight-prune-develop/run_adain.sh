#!/bin/bash
# Prepare dataset
#wget -O data/val2017.zip http://images.cocodataset.org/zips/val2017.zip
#unzip data/val2017.zip -d data/MSCOCO
#wget -O data/wikiart.zip https://doc-10-3k-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/vsp71mk9i5qso0cgpl6vbe90dtoaa74d/1560866400000/01686551976389763091/*/182-pFiKvXPB25DbTfAYjJ6gDE-ZCRXz0?e=download
#unzip data/wikiart.zip -d data/wikiart

# Move all images in subfolder to parent folder
#cd data/wikiart
#find . -mindepth 2 -type f -print -exec mv {} . \;
# also delete all subfolder(for now I did it manually)

# Clean files
rm -rf ./output
rm -f output.tar

mkdir output
mkdir output/pretrain
mkdir output/pruned
mkdir output/retrain
mkdir output/poster

# Run weight pruning
python weight_pruning_adain.py

#tar -czvf output.tar output

