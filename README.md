# A-frequency-domain-neural-network-for-fast-image-super-resolution
This is the implementation of paper "A frequency domain neural network for fast image super-resolution".

File directory:<br />
Network model implememtation:<br />
cnn_fft_train.m     This file is for training a new network model.<br />
cnn_init.m          The regular-net defined in here.<br />
cnn_fft_test_regular_batch.m   This is the testing file for our model.<br />

Hartely transformation:<br />
hartleyTrans.m    Hartely transformation in 2D version.<br />
hartleyTrans3D.m    Hartely transformation in 3D version.<br />

Loss function:<br />
vl_nnloss_l3.m     <br />
vl_nnloss_sqrt.m<br />
vl_nnloss_expl2.m<br />
vl_nnloss_l2.m<br />

Training data generation:<br />
./generate_training_data/generate_aug_data.m<br />

Testing data:<br />
./testing/BSDS100<br />
./testing/Set5<br />
./testing/Set14<br />
./testing/Set19<br />

Testing code(Pre-processing, only for testing):<br />
./testing_building_batch/imtobatch.m    To divid a testing image into several 360*480 batch. So that it can be process by network.<br />
./testing_building_batch/batchtoim.m    To integate many 360*480 batchs into the original image.<br />


Post-processing. Only for testing. (All the state of art in image super-resolution have this post-processing.)<br />
modcrop.m    To crop and shave image when testing. <br />
shave.m    To crop and shave image when testing. <br />


Journal version's addition:<br />
The below files are used in our Journal version, but not in the arKiv and conference version.
File directory:<br />

cnn_init_equalnet.m   The equal-net defined in here.<br />

HTmatrix.m   Compute the spatial kernel's frequency representation.<br />
computeHT.m<br />




Before runing the code. You need to first finish the compilation of MatConvNet. <br />
Do not re-download a new MatConvNet, since I have modified some files in the library. Directly run: "run matlab/vl_compilenn;" in the matlab will be enough. For more infomation about MatConvNet, please visit here: http://www.vlfeat.org/matconvnet/


# MatConvNet: CNNs for MATLAB

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images. Please visit
the [homepage](http://www.vlfeat.org/matconvnet) to know more.

In case of compilation issues, please read first the
[Installation](http://www.vlfeat.org/matconvnet/install/) and
[FAQ](http://www.vlfeat.org/matconvnet/faq/) section before creating an GitHub
issue. For general inquiries regarding network design and training
related questions, please use the
[Discussion forum](https://groups.google.com/d/forum/matconvnet).