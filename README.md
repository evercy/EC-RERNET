# EC-RERNET
The PyTorch implementation for the paper 'EC-RFERNet: an edge computing-oriented real-time facial expression recognition network'.
If this work is useful to you, please cite:
> Sun, Q., Chen, Y., Yang, D. X., Wen, J., Yang, J. J., Li, Y. L.: EC-RFERNet: an edge computing-oriented real-time facial expression recognition network. Signal, Image and Video Processing. https://doi.org/10.1007/s11760-023-02832-4

# Overview
![image](https://github.com/evercy/EC-RERNET/blob/main/images/Fig.%201.jpg)
Abstract:Edge computing has shown significant successes in addressing the security and privacy issues related to facial expression recognition (FER) tasks. Although several lightweight networks have been proposed for edge computing, the computing demands and memory access cost (MAC) imposed by these networks hinder their deployment on edge devices. Thus, we propose an edge computing-oriented real-time facial expression recognition network, called EC-RFERNet. Specifically, to improve the inference speed, we devise a mini-and-fast (MF) block based on the partial convolution operation. The MF block effectively reduces the MAC and parameters by processing only a part of the input feature maps and eliminating unnecessary channel expansion operations. To improve the accuracy, the squeeze-and-excitation (SE) operation is introduced into certain MF blocks, and the MF blocks at different levels are selectively connected by the harmonic dense connection. SE operation is used to complete the adaptive channel weighting, and the harmonic dense connection is used to exchange information between different MF blocks to enhance the feature learning ability. The MF block and the harmonic dense connection together constitute the harmonic-MF module, which is the core component of EC-RFERNet. This module achieves a balance between accuracy and inference speed. Five public datasets are used to test the validity of EC-RFERNet and to demonstrate its competitive performance, with only 2.25 MB and 0.55 million parameters. Furthermore, one human-robot interaction system is constructed with a humanoid robot equipped with the Raspberry Pi. The experimental results demonstrate that EC-RFERNet can provide an effective solution for practical FER applications.

# Dependencies
+ Python 3.6
+ PyTorch 1.7.1
+ torchvision 0.8.2

# Confusion matrix
![image](https://github.com/evercy/EC-RERNET/blob/main/images/Fig.%206.jpg)
EC-RFERNet corresponds to the confusion matrix for (a) RAF-DB (b) FER2013 (c) CK+ (d) AffectNet (e) SFEW dataset.

# One Human-Robot Interaction Case 
![image](https://github.com/evercy/EC-RERNET/blob/main/images/Fig.%207%20.jpg)
Schematic diagram of the HCI system based on Raspberry Pi 4B

# Citation
[1] Huang, Z., Yang, S. Z., Zhou, M. C., Gong, Z., Abusorrah, A., Lin, C., Huang, Z.: Making accurate object detection at the edge: review and new approach. Artif. Intell. Rev. 55(3), 2245–2274 (2022)

[2] Chao, P., Kao, C. Y., Ruan, Y., Huang, C. H., Lin, Y. L.: HarDNet: a low memory traffic network. In: The 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3551–3560 (2019)
