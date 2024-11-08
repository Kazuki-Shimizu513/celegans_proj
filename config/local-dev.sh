#!/bin/bash

python ./src/celegans_proj/run/run.py \
  --exp_name exp_20241106 \
  --in_dir /mnt/e/WDDD2_AD \
  --out_dir  /mnt/c/Users/compbio/Desktop/shimizudata/ \
  --batch 1 \
  --resolution 256 \
  --model_names \
              "Patchcore" \
              "ReverseDistillation" \
  --pseudo_anomaly_modes  \
              "wildType" \
              "patchBlack" \
              "gridBlack" \
              "zoom" \
              "shrink" \
              "oneCell" \
  --anomaly_gene_list \
              "wildType" \
              "F10E9.8" \
              "F54E7.3" \

