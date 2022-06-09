# I’m Something of a Painter Myself

## Problem Description
A GAN consists of at least two neural networks: a generator model and a discriminator model. The generator is a neural network that creates the images. For our project, we will be generating images in the style of Monet. This generator is trained using a discriminator.

The two models will work against each other, with the generator trying to trick the discriminator, and the discriminator trying to accurately classify the real vs. generated images.


## Previous Work
While researching on this topic we looked at a lot of previous work done around GANs and decided the best way to approach this problem would be by using CycleGANs. 
CycleGANs are a type of GAN architecture that involves the automatic training of image-to-image translation models without paired examples. The models are trained in an unsupervised manner using a collection of images from the source and target domain that do not need to be related in any way. For our project we would be putting the real images through our network to create a Fake Monet image and then re feeding it in the reverse order to try and get the original image back. We then uses these two Real(ish) images to calculate our losses and train the network.

## Approach

## Dataset
For our approach we used the dataset from Kaggle's : I’m Something of a Painter Myself Competition which has a collection of 300 Monet images and over 7000 real images. Our networks were made in TensorFlow using Keras models. 
