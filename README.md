# I’m Something of a Painter Myself

## Problem Description
A GAN consists of at least two neural networks: a generator model and a discriminator model. The generator is a neural network that creates the images. For our project, we will be generating images in the style of Monet. This generator is trained using a discriminator.

The two models will work against each other, with the generator trying to trick the discriminator, and the discriminator trying to accurately classify the real vs. generated images.


## Previous Work
While researching on this topic we looked at a lot of previous work done around GANs and decided the best way to approach this problem would be by using CycleGANs. 
CycleGANs are a type of GAN architecture that involves the automatic training of image-to-image translation models without paired examples. The models are trained in an unsupervised manner using a collection of images from the source and target domain that do not need to be related in any way. For our project we would be putting the real images through our network to create a Fake Monet image and then re feeding it in the reverse order to try and get the original image back. We then uses these two Real(ish) images to calculate our losses and train the network.

## Approach
### Discriminator
We had two discriminators: Photo and Monet
 - Architecture: Downsample from 256x256x3 to 30x30x1 using multiple 2D convolutional layers
 - Optimizer: Adam Optimizer with a learning rate=2e-4 and beta=0.5
 - Activation: Leaky ReLU after every layer
### Generator
We had two generators: Photo and Monet
 - Architecture: 
 - Encoder - Downsample from 256x256x3 to 1x1x512 using multiple 2D convolutional layers
 - Decoder - Upsample from 1x1x512 to 256x256x3 using multiple 2D convolutional layers
 - Optimizer: Adam Optimizer with a learning rate=2e-4 and beta=0.5
 - Activation: Leaky ReLU after every layer

### General framework
PIL - Python Imaging Library
Keras
TensorFlow

For more information refrer to slide 2 of [Demo Slides](https://docs.google.com/presentation/d/1w4oLIag3NXQzoHTgMCSeGJepoC8b0b8ABqd699oeiXk/edit?usp=sharing
)

## Dataset
For our approach we used the dataset from Kaggle's : I’m Something of a Painter Myself Competition which has a collection of 300 Monet images and over 7000 real images. Our networks were made in TensorFlow using Keras models. 


## Results
 - Epoch 1 : monet_gen_loss: 12.1780 - photo_gen_loss: 12.8266 - monet_disc_loss: 0.6873 - photo_disc_loss: 0.6885
 - Epoch 2 : monet_gen_loss: 3.3638 - photo_gen_loss: 3.5896 - monet_disc_loss: 0.6303 - photo_disc_loss: 0.5723
 - Epoch 3 : monet_gen_loss: 3.1253 - photo_gen_loss: 3.2091 - monet_disc_loss: 0.6159 - photo_disc_loss: 0.5543
 - Epoch 4 : monet_gen_loss: 2.7849 - photo_gen_loss: 2.8058 - monet_disc_loss: 0.6284 - photo_disc_loss: 0.5973
 - Epoch 5 : monet_gen_loss: 2.5382 - photo_gen_loss: 2.6347 - monet_disc_loss: 0.6032 - photo_disc_loss: 0.6083
 - Epoch 6 : monet_gen_loss: 2.4919 - photo_gen_loss: 2.5578 - monet_disc_loss: 0.6163 - photo_disc_loss: 0.6047

## Video
[![Demo Video](https://img.youtube.com/vi/RI8qSyGP864/0.jpg)](https://youtu.be/RI8qSyGP864)

## Presentation
https://docs.google.com/presentation/d/1w4oLIag3NXQzoHTgMCSeGJepoC8b0b8ABqd699oeiXk/edit?usp=sharing

