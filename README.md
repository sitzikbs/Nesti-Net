***Nesti-Net***: Normal Estimation for Unstructured 3D Point Clouds using Convolutional Neural Networks
---
Created by [Yizhak (Itzik) Ben-Shabat](http://www.itzikbs.com), [Michael Lindenbaum](http://www.cs.technion.ac.il/people/mic/index.html), and [Anath Fischer](https://meeng.technion.ac.il/members/anath-fischer/) from [Technion, I.I.T](https://www.technion.ac.il/en/).

![Nesti-Net_pipeline](https://github.com/sitzikbs/Nesti-Net/blob/master/doc/NestiNet_pipeline.png)

### Introduction
This is the code for estimating normal vectors for unstructured 3D point clouds using Nesti-Net. It allows to train, test and evaluate our different normal estimation models. We provide the option to train a model or use a pretrained model. Please follow the installation instructions below.

Here is a short [YouTube](https://www.youtube.com/watch?v=E7PudeA4XvM) video providing a brief overview of the methods.

Abstract:

We propose a normal estimation method for unstructured 3D point clouds. This method, called Nesti-Net, builds on a new local point cloud representation which consists of multi-scale point statistics (MuPS), estimated on a local coarse Gaussian grid. This representation is a suitable input to a CNN architecture. The normals are estimated using a mixture-of-experts (MoE) architecture, which relies on a data-driven approach for selecting the optimal scale around each point and encourages sub-network specialization. Interesting insights into the network's resource distribution are provided. The scale prediction significantly improves robustness to different noise levels, point density variations and different levels of detail. We achieve state-of-the-art results on a benchmark synthetic dataset and present qualitative results on real scanned scenes. 

### Citation
If you find our work useful in your research, please cite our [paper](https://arxiv.org/abs/1812.00709):

Preprint:

    @article{ben_shabat2018nestinet,
      title={Nesti-Net: Normal Estimation for Unstructured 3D Point Clouds using Convolutional Neural Networks},
      author={Ben-Shabat, Yizhak and Lindenbaum, Michael and Fischer, Anath},
      journal={arXiv preprint arXiv:1812.00709},
      year={2018}
    }

### Installation
Install [Tensorflow](https://www.tensorflow.org) and [scikit-learn](http://scikit-learn.org/stable/).
 You will also need `torch`, and `torchvision` for the PCPNet data loader.
 
The code was tested with Python 2.7, TensorFlow 1.12, torch 0.4.1, torchvision 0.2.1, CUDA 9.2.148, and cuDNN 7201 on Ubuntu 16.04.


Download the PCPNet data from this [link](http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip) and place it in the `data` directory.
Alternatively, download the data used to train and test Nesti-Net from this [link](https://technionmail-my.sharepoint.com/:u:/g/personal/cadlab_technion_ac_il/EX9hmT6gUuhOlO5039nLaroB-bkUEObPOy1BHtUBNPKnjg?e=YZCKna) and place it in the `data` directory.

Download all trained models from this [link](https://technionmail-my.sharepoint.com/:u:/g/personal/cadlab_technion_ac_il/ERcpW34CYzNIvHAP7f1OSpcBgePbFF1XAPrWNtn_fXdeLg?e=4b4Zt6) and place them in the `models` directory.
alternatively, dowload just the mixture of experts model (Nesti-Net)  from this [link](https://technionmail-my.sharepoint.com/:u:/g/personal/cadlab_technion_ac_il/ETmIRKAIjZdEoYq0d1L_h0EBYbNu95jN1HYiIf3zT_ztXg?e=EDUtUV)


### Train
This repository allows to train a Nesti-Net multi-scale mixture of experts network for normal estimation.
Simply run `train_n_est_w_experts.py`. 
 

This repository allows to train additional three vatiations: 
Single-scale / multi-scale models can be trained by running `train_n_est.py`. 
Multi-scale with switching - i.e. estimating the noise and switching between small scale network and large scale network (note that for this you will need a `.txt` file specifying the noise for each point cloud). It can be trained by running `train_n_est_w_switching.py`.

### Test
To test Nesti-Net run `test_n_est_w_experts.py` and input the desired model log directory. 

In order to test on your own data, place your point cloud directory in the `data` directory. Make sure that your point cloud directory includes a test set `.txt` file, that lists the files you wish to include in the test.  

For the other models run the corresponding `test_...py` file.

Testing the different models will generate a `results` directory inside the trained model `log` directory. The results will be saved as separate `.normals` files containing the estimated normals. 

### Evaluate
To compute the RMS error, PGP5 or PGP10 evaluation metrics run `evaluate.py` for the desired `results` directory. The evaluation results will be saved in a `summary` directory within the `results` directory.

### License
See LICENSE file.


