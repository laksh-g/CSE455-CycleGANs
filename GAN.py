import tensorflow as tf
import os
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

BATCH_SIZE = 20

class GAN:
    def __init__(self):
        self.discriminator = self.make_discriminator_model()
        self.generator = self.make_generator_model()
        self.dataset = None
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.checkpoint = None
        self.checkpoint_prefix = None

    def run(self):
        self.load_data()
        self.save_ckpt()
        EPOCHS = 200
        # noise = tf.random.normal([1, 100])
        # generated_image = self.generator(noise, training=False)
        # print(generated_image)
        # plt.imshow(generated_image[0, :, :, 0], cmap='gray')
        # decision = self.discriminator(generated_image)
        # print(decision)
        self.train(self.dataset, EPOCHS)

    def save_ckpt(self):
        checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)


    def train(self, dataset, epochs):
        noise_dim = BATCH_SIZE
        num_examples_to_generate = 16

        # You will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        seed = tf.random.normal([num_examples_to_generate, noise_dim])
        gen_losses = []
        disc_losses = []
        for epoch in range(epochs):
            print(f'Epoch : {epoch}')
            start = time.time()
            # print(self.dataset.list_builders())
            gen_loss_total = 0
            disc_loss_total = 0
            num_batches = 0
            for image_batch in dataset:
                # print(f"image batch shape {tf.shape(image_batch[0])}")
                gen_loss, disc_loss = self.train_step(image_batch[0])
                gen_loss_total += gen_loss
                disc_loss_total += disc_loss
                num_batches+=1
            gen_losses.append(gen_loss_total / num_batches)
            disc_losses.append(disc_loss_total / num_batches)
            print(f'gen_loss [{epoch}] = {gen_losses[-1]}')
            print(f'disc_loss [{epoch}] = {disc_losses[-1]}')


            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator,
                                     epoch + 1,
                                     seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

            # Generate after the final epoch
        display.clear_output(wait=True)
        self.generate_and_save_images(self.generator,
                                 epochs,
                                 seed)
        print(f'Final Gen Loss : {gen_losses[-1]}')
        print(f'Final Disc Loss : {disc_losses[-1]}')

    @tf.function
    def train_step(self, images):
        noise_dim = BATCH_SIZE
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            print(f'shape of imgs {images.shape}')
            print(f'shape of img {images[0].shape}')
            real_output = self.discriminator(images, training=True)
            print(type(real_output))
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss



    def load_data(self):
        data_dir = os.path.join(os.getcwd(), 'data')
        self.dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir, labels='inferred', batch_size=BATCH_SIZE)
        print(self.dataset)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return (total_loss + 1e-2)*0.5

    def generator_loss(self, fake_output):
        return self.loss(tf.ones_like(fake_output), fake_output)

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(32 * 32 * 512, use_bias=False, input_shape=(BATCH_SIZE,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((32, 32, 512)))
        assert model.output_shape == (None, 32, 32, 512)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 64, 64, 256)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 128, 128, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 128, 128, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 128, 128, 32)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(16, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 128, 128, 16)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 256, 256, 3)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[256, 256, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='brg')
            plt.axis('off')

        plt.savefig('training_checkpoints/imgs/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()







g = GAN()
g.run()