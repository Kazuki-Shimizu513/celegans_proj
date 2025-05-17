
# SAPAG: Self-Attention Parts Guidance for Anomaly Detection in C. elegans Images with Latent Diffusion Models

## What is SAPAG

Detecting phenotypic anomalies in microscopy images, those caused by genetic mutations or environmental factors, is an important task in biomedical image analysis. We propose a novel unsupervised anomaly detection method, Self-Attention Parts Guidance (SAPAG), which leverages Latent Diffusion Models (LDMs) trained on normal images. SAPAG operates in two steps: 
 1.  segmenting self-attention maps into semantically distinct regions using an unsupervised segmentation model, DiffSeg, and
 2. performing region-wise reconstruction during the reverse diffusion process, guided by Self-Attention Guidance(SAG).

We validated SAPAG using microscopy images of C. elegans embryos,where wild-type embryo images were used as normal data, RNAi-treated embryo images – which may contain phenotypic anomalies – serve as test data. Experiments show that SAPAG outperforms representative anomaly detection methods,
including PatchCore, RD4AD, and THOR, especially in detecting anomalies in embryo shape and size. Ablation studies further confirm that both SAG and DiffSeg contribute to improved detection performance. Although subtle small structural anomalies such as nuclei remain challenging, SAPAG demonstrates strong potential for phenotypic screening in microscopy-based   analysis.

## Getting Started

### Installation

1. Clone this Repo

```
gh repo clone Kazuki-Shimizu513/celegans_proj
cd celegans_proj
```

2. Install rye via https://rye.astral.sh/guide/installation/
3. Sync all dependencies 

```
rye sync
```

### Dataset Prepareraion
All dataset we use made from https://wddd.riken.jp/
please fetch and expand data using https://github.com/Kazuki-Shimizu513/celegans_proj/blob/main/src/celegans_proj/generate/fetch.py
and, make WDDD2_AD dataset using https://github.com/Kazuki-Shimizu513/celegans_proj/blob/main/src/celegans_proj/generate/transfer.py
note: make sure implement argments which fits your environment

### Training 
training code is https://github.com/Kazuki-Shimizu513/celegans_proj/blob/main/src/celegans_proj/run/train.py
please modify arguments to your environment and just run 
```
cd celegans_proj/run
python train.py
```

### Inference

also inference code is https://github.com/Kazuki-Shimizu513/celegans_proj/blob/main/src/celegans_proj/run/predict.py
please modify arguments to your environment and just run 
```
cd celegans_proj/run
python predict.py
```


### Citation
@article{kazuki2025sapag,
      title={SAPAG: Self-Attention Parts Guidance for Anomaly Detection in C. elegans Images with Latent Diffusion Models}, 
      author={Kazuki Shimizu and, Yukako Tohsato},
      journal={},
      year={2025}
}
