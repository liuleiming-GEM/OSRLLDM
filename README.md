# OSRLLDM
OSRLLDM: Omnidirectional Image Super-Resolution with Latitude-Aware Latent Diffusion Models

# Requirements
* Python 3.10, Pytorch 2.3.0, pytorch-lightning 1.9.4
* More detail (See [environment.yml](environment.yml))
A suitable conda environment named `OSRLLDM` can be created and activated with:
```
conda env create -n OSRLLDM python=3.10
conda activate OSRLLDM
pip install -r requirements.txt
```
or
```
conda env create -f environment.yml
conda activate OSRLLDM
```

# Data Preparation


# Training
Training VQ-LAE

Training Denoising U-Net

# Testing
Pretrained models (X4) and SR ODIs (X2) can be downloaded [here](https://pan.baidu.com/s/1zrW_TL0c4iUw8_CIN8u3nQ). 提取码：1234

Testing Denoising UNet

## Acknowledgement

This project is based on [LDM](https://github.com/CompVis/latent-diffusion), [ResShift](https://github.com/zsyOAOA/ResShift.git), and [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for their awesome works.
