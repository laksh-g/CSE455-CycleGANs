import tensorflow as tf
import numpy as np
import re
from CycleGAN import CycleGan
import PIL
import dataset
import main

IMAGE_SIZE = [256, 256]

GCS_PATH_MONET = '/git_test/data/monet_tfrec'
GCS_PATH = '/git_test/data'

PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))
version = 1.2
latest = tf.train.latest_checkpoint(f'training_checkpoints\\v{version}')
print(f'restoring {latest}')

model = CycleGan(
        main.Generator(), main.Generator(), main.Discriminator(), main.Discriminator()
    )
model.compile(
        m_gen_optimizer=main.monet_generator_optimizer,
        p_gen_optimizer=main.photo_generator_optimizer,
        m_disc_optimizer=main.monet_discriminator_optimizer,
        p_disc_optimizer=main.photo_discriminator_optimizer,
        gen_loss_fn=main.generator_loss,
        disc_loss_fn=main.discriminator_loss,
        cycle_loss_fn=main.calc_cycle_loss,
        identity_loss_fn=main.identity_loss
    )
model.load_weights(latest)
print("model restored")
print(model.m_gen)


def predict_and_save(input_ds, generator_model, output_path):
    i = 1
    print(f'predicting and saving {input_ds}')
    for img in input_ds:
        print(f'saving img {i}')
        if 50 < i < 100:
            prediction = generator_model(img, training=False)[0].numpy() # make predition
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)   # re-scale
            im = PIL.Image.fromarray(prediction)
            im.save(f'{output_path}{str(i)}.jpg')
            i += 1
        else:
            return


predict_and_save(main.load_dataset(PHOTO_FILENAMES).batch(1), model.m_gen, "Results\\")

