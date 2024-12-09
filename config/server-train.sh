#!/bin/bash

python ./src/celegans_proj/run/run.py \
  --exp_name exp_20241209 \
  --in_dir /home/skazuki/data/WDDD2_AD \
  --out_dir  /home/skazuki/skazuki/result \
  --batch 1 \
  --resolution 256 \
  --model_names \
      "Patchcore"\
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





