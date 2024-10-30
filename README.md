# DeformableTST (NeurIPS 2024)
This is an official implementation of paper: DeformableTST: Transformer for Time Series Forecasting without Over-reliance on Patching

## Our Paper
Donghao Luo and Xue Wang. DeformableTST: Transformer for Time Series Forecasting without Over-reliance on Patching. In 38th Conference on Neural Information Processing Systems (NeurIPS 2024), 2024.

We expose, explore and solve the issue of over-reliance on patching in previous Transformer-based models.
+ We observe a new problem that the recent Transformer-based models are overly reliant on patching to achieve ideal performance, which limits their applicability to some tasks unsuitable for patching.
+ We propose **DeformableTST** as an effective solution to this emerging issue. Specifically, we propose deformable attention, a sparse attention mechanism that can better focus on the important time points by itself, to get rid of the need of
patching. And we also adopt a hierarchical structure to alleviate the efficiency issue caused by the removal of patching.
+ Our DeformableTST can successfully reduce the reliance on patching and broaden the applicability of Transformer-based models.


## Structure Overview

Structure overview of DeformableTST. 
+ The input time series is embedded variate-independently. 
+ The local perception unit (LPU) is used to learn the local temporal information.
+ The proposed deformable attention is adopted to learn the global temporal information. 
+ The feed-forward network injected with a depth-wise convolution (ConvFFN) is used to learn the local temporal information and the new feature representation.

![Structure Overview](fig/fig_structure.png)


## Deformable Attention
Deformable Attention.
+ Sample the important time points based on a set of learnable coordinates called sampling points. Specifically, the sampling points are calculated by a set of uniformly sparse reference points and their learnable offsets.
+ Calculate attention outputs with the selected important time points to learn non-trivial representation.

![Deformable Attention](fig/fig_attention.png)

## Main Results

Main Results. 
+ DeformableTST achieves consistent state-of-the-art performance in a broader range of time series tasks.
+ DeformableTST can flexibly adapt to multiple input lengths and achieve excellent performance in tasks unsuitable for patching, which is a great improvement than previous Transformer-based models.
+ DeformableTST can successfully reduce the reliance on patching and broaden the applicability of Transformer-based models.

![Main Results](fig/fig_mainresult.png)

## Get Started

1. Install Python 3.7 and necessary dependencies.
```
pip install -r requirements.txt
```
2. Download data. You can obtain all datasets from [[Times-series-library](https://github.com/thuml/Time-Series-Library)].

3. Train the model. We provide the experiment scripts under the folder `./scripts`. To run the code on ETTh2, just run the following command:

```
sh ./scripts/ETTh2.sh
```

## Contact
If you have any question or want to use the code, please contact [ldh21@mails.tsinghua.edu.cn](mailto:ldh21@mails.tsinghua.edu.cn).

## Citation

If you find this repo useful, please cite our paper. 
```
@inproceedings{
Luo2024deformabletst,
title={Deformable{TST}: Transformer for Time Series Forecasting without Over-reliance on Patching},
author={Donghao Luo and Xue Wang},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
}
```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/luodhhh/ModernTCN

https://github.com/ts-kim/RevIN

https://github.com/PatchTST/PatchTST

https://github.com/thuml/Time-Series-Library

https://github.com/LeapLabTHU/DAT
