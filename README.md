# EC-RFERNet: an edge computing-oriented real-time facial expression recognition network
The PyTorch implementation for the paper 'EC-RFERNet: an edge computing-oriented real-time facial expression recognition network'.
Qiang Sun<sup>1,2</sup>,Yuan Chen<sup>1</sup>, Dongxu Yang<sup>1</sup>, Jing Wen<sup>1</sup>, Jiaojiao Yang<sup>1</sup>,Yonglu Li<sub>3</sub>
1.	Department of Communication Engineering, School of Automation and Information Engineering, Xi’an University of Technology, Xi’an 710048, China
2.	Xi’an Key Laboratory of Wireless Optical Communication and Network Research, Xi’an 710048, China
3.	Xi’an Founder Robot Co., LTD, Xi’an 710068, China

# Overview
![image](https://github.com/evercy/EC-RERNET/blob/main/images/Fig.%201.jpg)
<p align="center">Architecture of the EC-RFERNet </p>

Abstract: Edge computing has shown significant successes in addressing the security and privacy issues related to facial expression recognition (FER) tasks. Although several lightweight networks have been proposed for edge computing, the computing demands and memory access cost (MAC) imposed by these networks hinder their deployment on edge devices. Thus, we propose an edge computing-oriented real-time facial expression recognition network, called EC-RFERNet. Specifically, to improve the inference speed, we devise a mini-and-fast (MF) block based on the partial convolution operation. The MF block effectively reduces the MAC and parameters by processing only a part of the input feature maps and eliminating unnecessary channel expansion operations. To improve the accuracy, the squeeze-and-excitation (SE) operation is introduced into certain MF blocks, and the MF blocks at different levels are selectively connected by the harmonic dense connection. SE operation is used to complete the adaptive channel weighting, and the harmonic dense connection is used to exchange information between different MF blocks to enhance the feature learning ability. The MF block and the harmonic dense connection together constitute the harmonic-MF module, which is the core component of EC-RFERNet. This module achieves a balance between accuracy and inference speed. Five public datasets are used to test the validity of EC-RFERNet and to demonstrate its competitive performance, with only 2.25 MB and 0.55 million parameters. Furthermore, one human-robot interaction system is constructed with a humanoid robot equipped with the Raspberry Pi. The experimental results demonstrate that EC-RFERNet can provide an effective solution for practical FER applications.

# Dependencies
+ Python 3.6
+ PyTorch 1.7.1
+ torchvision 0.8.2

# Confusion matrix
![image](https://github.com/evercy/EC-RERNET/blob/main/images/Fig.%206.jpg)
<p align="center">EC-RFERNet corresponds to the confusion matrix for (a) RAF-DB (b) FER2013 (c) CK+ (d) AffectNet (e) SFEW dataset.</p>

# Data availability
All datasets are freely available in public repositories. 
+ RAF-DB: http://www.whdeng.cn/raf/model1.html
+ FER2013: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
+ CK+ : http://www.jeffcohn.net/Resources/
+ AffectNet: http://mohammadmahoor.com/affectnet/
+ SFEW: https://cs.anu.edu.au/few/AFEW.html

# Contact
If you have any questions, please feel free to reach me out at qsun@xaut.edu.cn and 2210321182@stu.xaut.edu.cn.

# Acknowledgements
This project is built upon [CNN-RIS](https://github.com/yangshunzhi1994/CNN-RIS) and [Pytorch-HarDNet](https://github.com/PingoLH/Pytorch-HarDNet). Thanks for their great codebase.

# Citation
If this work is useful to you, please cite:
> Sun, Q., Chen, Y., Yang, D. X., Wen, J., Yang, J. J., Li, Y. L.: EC-RFERNet: an edge computing-oriented real-time facial expression recognition network. Signal, Image and Video Processing. https://doi.org/10.1007/s11760-023-02832-4

