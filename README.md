# A hybrid PCNet and STC method for noisy plant PET image classification
# Abstract
{\it Objective}. Plant Positron Emission Tomography (PET) is a new and efficient imaging technique which aims at providing a quantitative analysis of plant stress, enabling personalized crop management and maximizing productivity. However, a highly performant classification system for noisy dynamic plant PET images faces the challenge of retrieving noise-free datasets and encoding both spatial and temporal representations within a unified model.

{\it Approach}. To overcome these limitations, we introduce an innovative hybrid model that combines denoising and classification for dynamic plant PET images. Initially, we compute a precise solution for the denoising problem of noisy dynamic plant PET images using a modified optimization method coupled with deep convolutional neural networks. Subsequently, this solution is unfolded into a deep network known as the Predictor-Corrector Network (PCNet). To optimize the PCNet without requiring a noise-free dynamic training set, we propose a novel unsupervised learning method. Finally, the sequence of noise-reduced dynamic plant PET images is further fed into a unique classification system, encoding spatial representations of images and temporal representations of multivariate time series into a unified spatiotemporal representation and generating a prediction.

{\it Main results}. The algorithm was tested on a set of 2-[$^{18}$F]-Fluorodeoxyglucose time-dynamic PET images acquired on a set of 24 control and saline-stressed  zucchini sprouts. Notably, the classification performance between the two classes achieves an averaged accuracy of 0.852, an averaged precision of 0.838, an averaged recall of 0.959, and an averaged F1-score of 0.880.

{\it Significance}. The generality and modularity of the proposed hybrid model make it well-suited for addressing broader denoising and classification challenges involving multimodal and heterogeneous data.
# Results
## Denoising Visualization ##
<img src="https://github.com/weikechang/PCNet-with-STC/Results/Denoising_Visualization.png" width="551" height="309" alt="111"/><br/>
## Denoising and Classification Results ##
![ezcv logo](https://github.com/weikechang/PCNet-with-STC/Results/Classification.png)
