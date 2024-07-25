## CBDAE

This repository implements the Contrastive Blind Denoising Autoencoder described in the ["Contrastive Blind Denoising Autoencoder for Real Time Denoising of Industrial IoT Sensor Data"](https://arxiv.org/pdf/2004.06806.pdf) by Saúl Langarica et al.

## Paper Information

Title: Contrastive blind denoising autoencoder for real time denoising of industrial IoT sensor data

Authors: Saúl Langarica, Felipe Núñez

Publication Year: 2023

Journal: Engineering Applications of Artificial Intelligence

Volume: 120

Pages: 105838

DOI: https://doi.org/10.1016/j.engappai.2023.105838

## Project introduction

With the widespread use of data science, there have been many attempts to apply it to the industrial field. However, data-driven approaches have limitations due to heavily corrupted data in some plants. In certain situations, conventional denoising methods like wavelet, FFT or PCA enhance the performance of data-driven techniques by partially eliminating data noise. However, this type of filtering often disregards the multivariate or dynamic (unsteady or time-varying) properties of the data. 

Therefore, it is necessary to recover accurate observations by analyzing corrupted data while taking into account its multivariate and dynamic behavior.

The main idea behind CBDAE (Contrastive Blind Denoising Autoencoder) is to introduce **contrast loss** into time series data.

```math
L=L_{AE}+\beta L_{NCE}
```

The purpose of this project is to implement the CBDAE introduced in the paper and enable experimentation on a variety of datasets.

## Pytorch version

```makefile
torch==2.1.1
```

