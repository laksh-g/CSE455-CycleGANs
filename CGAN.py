import tensorflow as tf
from tensorflow import keras
import os


class CGan(keras.Model):
    def __init__(self, monet_generator, photo_generator, monet_discriminator, photo_discriminator, dataset, epochs, steps_per_ep, lambda_cycle=10):
        super(CGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle
        self.dataset = dataset
        self.epochs = epochs
        self.steps_per_epoch = steps_per_ep
        self.version = 1.5

    def compile(self):
        super(CGan, self).compile()
        self.m_gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.p_gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.m_disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.p_disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.gen_loss_fn = self.generator_loss
        self.disc_loss_fn = self.discriminator_loss
        self.cycle_loss_fn = self.calc_cycle_loss
        self.identity_loss_fn = self.identity_loss

    def discriminator_loss(self, real, generated):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)
        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image, LAMBDA):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return LAMBDA * loss1

    def identity_loss(self, real_image, same_image, LAMBDA):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss

    def train_step(self, batch_data):
        real_monet, real_photo = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(fake_monet, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(
                real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet,
                                                                                             self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo,
                                                                                             self.lambda_cycle)

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        # Calculate the gradients for generator and discriminator
        m_gen_grads, p_gen_grads, m_disc_grads, p_disc_grads = self.calc_grads(tape, total_monet_gen_loss,
                                                                               total_photo_gen_loss, monet_disc_loss,
                                                                               photo_disc_loss)

        # Apply the gradients to the optimizer
        self.apply_grads(m_gen_grads, p_gen_grads, m_disc_grads, p_disc_grads)

        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }

    def calc_grads(self, tape, m_gen_loss, p_gen_loss, m_disc_loss, p_disc_loss):
        m_gen_grads = tape.gradient(m_gen_loss, self.m_gen.trainable_variables)
        p_gen_grads = tape.gradient(p_gen_loss, self.p_gen.trainable_variables)

        m_disc_grads = tape.gradient(m_disc_loss, self.m_disc.trainable_variables)
        p_disc_grads = tape.gradient(p_disc_loss, self.p_disc.trainable_variables)

        return m_gen_grads, p_gen_grads, m_disc_grads, p_disc_grads

    def apply_grads(self, m_gen_grads, p_gen_grads, m_disc_grads, p_disc_grads):
        self.m_gen_optimizer.apply_gradients(zip(m_gen_grads, self.m_gen.trainable_variables))
        self.p_gen_optimizer.apply_gradients(zip(p_gen_grads, self.p_gen.trainable_variables))
        self.m_disc_optimizer.apply_gradients(zip(m_disc_grads, self.m_disc.trainable_variables))
        self.p_disc_optimizer.apply_gradients(zip(p_disc_grads, self.p_disc.trainable_variables))

    def fit_model(self):
        self.fit(
            self.dataset,
            epochs=self.epochs,
            callbacks=[self.checkpoint()],
            steps_per_epoch=self.steps_per_epoch,
        )

    def checkpoint(self):
        checkpoint_path = f'training_checkpoints/v{self.version}/cp.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)

        latest = tf.train.latest_checkpoint(f'training_checkpoints\\v1.4')
        self.load_weights(latest)
        print("model restored")

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True)
        return cp_callback
