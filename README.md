# MUN: Image Forgery Localization Based on M<sup>3</sup> Encoder and UN Decoder

This is the PyTorch implementation for **MUN**.

### Abstract

Image forgeries can entirely change the semantic information of an image, and can be used for unscrupulous purposes. In this paper, we propose a novel image forgery localization network named as MUN, which consists of an M<sup>3</sup> encoder and a UN decoder. Firstly, the M<sup>3</sup> encoder is constructed based on a Multi-scale Max-pooling query module to extract Multi-clue forged  features. Noiseprint++ is adopted to assist the RGB clue, and  its deployment methodology is  discussed. A Multi-scale Max-pooling Query (MMQ) module is proposed to integrate RGB and noise features. Secondly, a novel UN decoder is proposed to extract hierarchical features from both top-down and bottom-up directions, reconstructing both high-level and low-level features at the same time. Thirdly, we formulate an  IoU-recalibrated Dynamic Cross-Entropy (IoUDCE) loss to dynamically adjust the weights on forged regions according to IoU which can adaptively balance the influence of authentic and forged regions. Last but not least, we propose a data augmentation method, i.e., Deviation Noise Augmentation (DNA), which acquires accessible prior knowledge of RGB distribution to improve the generalization ability. Extensive experiments on publicly available datasets show that MUN outperforms the state-of-the-art works.

### Environment

------

We develop our codes in the following environment:

- Python==3.8.18
- Pytorch==2.0.1
- MMSemgmentation==1.2.2
- MMPretrain==1.0.0

### Experimental Results



### How to run the codes

1. Install [MMSegmentaion](https://github.com/open-mmlab/mmsegmentation) and [MMPretrain](https://github.com/open-mmlab/mmpretrain) follow the official tutorial.
2. Move the files provided here to the folder corresponding to MMSegmentation.
3. Run the codes in demo/test.ipynb.

### Datasets

You can download the training dataset from [link](https://github.com/free1dom1/TBFormer).

### Citation
