#!/bin/bash
rm -rf pretrain/
rm -rf pruned/
rm -f pretrained_benchmark.txt  
rm -f pruned_benchmark.txt
rm -f pruned.tar
rm -f pretrained.tar

mkdir pretrain
mkdir pruned

python weight_pruning_adain.py

tar -czvf pruned.tar pruned
tar -czvf pretrain.tar pretrain

