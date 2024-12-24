# MUN: Image Forgery Localization Based on M<sup>3</sup> Encoder and UN Decoder

**MUN** is accepted by **AAAI 2025**!

---

### Abstract

Image forgeries can entirely change the semantic information of an image, and can be used for unscrupulous purposes. In this paper, we propose a novel image forgery localization network named as MUN, which consists of an M<sup>3</sup> encoder and a UN decoder. Firstly, the M<sup>3</sup> encoder is constructed based on a Multi-scale Max-pooling query module to extract Multi-clue forged  features. Noiseprint++ is adopted to assist the RGB clue, and  its deployment methodology is  discussed. A Multi-scale Max-pooling Query (MMQ) module is proposed to integrate RGB and noise features. Secondly, a novel UN decoder is proposed to extract hierarchical features from both top-down and bottom-up directions, reconstructing both high-level and low-level features at the same time. Thirdly, we formulate an  IoU-recalibrated Dynamic Cross-Entropy (IoUDCE) loss to dynamically adjust the weights on forged regions according to IoU which can adaptively balance the influence of authentic and forged regions. Last but not least, we propose a data augmentation method, i.e., Deviation Noise Augmentation (DNA), which acquires accessible prior knowledge of RGB distribution to improve the generalization ability. Extensive experiments on publicly available datasets show that MUN outperforms the state-of-the-art works.



---

### Environment

We develop our codes in the following environment:

- Python==3.8.18
- Pytorch==2.0.1
- MMSemgmentation==1.2.2
- MMPretrain==1.0.0
- MMDetection==3.3.0

---

### Pretrained models

Please download the pretrained MUN model from [Google Drive](https://drive.google.com/file/d/1Lww1Y3BX-DwzybwehSgdOv-yDNF1Pgm_/view?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/1LA0Uh72qDUhe9De4aJczUQ?pwd=5qjs) (password: 5qjs), and put it in the demo folder.

---

### Experimental results

|          Methods          |    NIST16    |  CASIA v1.0  |   IMD2020    |  CocoGlide   |     Wild     |
| :-----------------------: | :----------: | :----------: | :----------: | :----------: | :----------: |
|    **RGB-N**(CVPR’18)     |    0.764     |    0.795     |      -       |      -       |      -       |
|  **ManTra-Net**(CVPR’19)  |    0.795     |    0.817     |    0.748     |    0.778     |    0.677     |
|     **SPAN**(ECCV’20)     |    0.840     |    0.797     |    0.750     |    0.475     |      -       |
|   **MVSS-Net**(ICCV’21)   |      -       |    0.815     |    0.814     |    0.654     |    0.768     |
|  **PSCC-Net**(TCSVT’22)   |    0.855     |    0.829     |    0.806     |    0.777     |    0.745     |
| **ObjectFormer**(CVPR’22) |    0.872     |    0.843     |    0.821     |      -       |      -       |
|    **TANet**(TCSVT’23)    | <u>0.898</u> |    0.853     |    0.849     |      -       | <u>0.832</u> |
|   **TBFormer**(SPL’23)    |    0.847     |    0.955     |    0.863     |    0.747     |    0.783     |
|     **HiFi**(CVPR’23)     |    0.869     |    0.866     |    0.834     |      -       |      -       |
|    **TruFor**(CVPR’23)    |    0.839     |    0.833     |    0.818     |    0.752     |      -       |
|   **CSR-Net**(AAAI’24)    |    0.883     |    0.881     |    0.854     |      -       |      -       |
|   **NRL-Net**(AAAI’24)    |  **0.900**   |    0.872     |    0.852     |      -       |      -       |
|  **MGQFormer**(AAAI'24)   |    0.862     |    0.886     |    0.883     |      -       |      -       |
|       **MUN**(Ours)       |    0.857     |  **0.967**   | <u>0.885</u> | <u>0.811</u> |    0.805     |
|      **MUN\***(Ours)      |    0.861     | <u>0.962</u> |  **0.897**   |  **0.815**   |  **0.843**   |


The bold entities denote the best results per column and the underlined ones denote the second best results. * denotes
that the DNA data augmentation method is performed. AUC scores are reported.

---

### How to run the codes

1. Install [MMSegmentaion](https://github.com/open-mmlab/mmsegmentation), [MMPretrain](https://github.com/open-mmlab/mmpretrain) and [MMDetection](https://github.com/open-mmlab/mmdetection) follow the official tutorial.
2. Move the files provided here to the folder corresponding to MMSegmentation.
3. Run the codes in demo/test.ipynb.

---

### Datasets

You can download the training dataset from [link](https://www.kaggle.com/datasets/hanhan0104/mydata).

---

### Citation

If you find our code useful, please generously cite our paper.

```
@inproceedings{
Liu2025mun,
title={{MUN}: Image Forgery Localization Based on M^3 Encoder and {UN} Decoder},
author={Liu, Yaqi and Chen, Shuhuan and Shi, Haichao and Zhang, Xiao-Yu and Xiao, Song and Cai, Qiang},
booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
year={2025}
}
```


