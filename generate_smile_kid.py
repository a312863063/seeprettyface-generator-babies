# Thanks to StyleGAN provider —— Copyright (c) 2019, NVIDIA CORPORATION.
#
# This work is trained by Copyright(c) 2018, seeprettyface.com, BUPT_GWY.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image as Image
import dnnlib.tflib as tflib

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

def text_save(file, data):  # save generate code, which can be modified to generate customized style
    for i in range(len(data[0])):
        s = str(data[0][i])+'\n'
        file.write(s)


def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    model_path = 'model/generator_kids.pkl'
    classifier_path = 'model/classifier-smiling.pkl'

    # Prepare result folder
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/generate_code', exist_ok=True)

    with open(model_path, "rb") as f:
        _G, _D, Gs = pickle.load(f, encoding='latin1')

    with open(classifier_path, "rb") as f:
        discriminator = pickle.load(f)

    # Print network details.
    Gs.print_layers()

    # Generate pictures
    generate_num = 20  # 需要生成的图片数目
    count = 0
    while count != generate_num:

        # Generate latent.
        latents = np.random.randn(1, Gs.input_shape[1])

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

        img_ori = Image.fromarray(images[0], 'RGB')
        img = img_ori.resize((256, 256), Image.ANTIALIAS)
        img = np.asarray(img)
        img_chw = np.transpose(img, (2, 0, 1))
        img_chw = img_chw[np.newaxis, :, :, :]
        answer = discriminator.run(img_chw, None)[0][0]

        if answer < -500:
            # Save latent.
            txt_filename = os.path.join(result_dir, 'generate_code/' + str(count).zfill(4) + '.txt')
            file = open(txt_filename, 'w')
            text_save(file, latents)
            file.close()

            # Save image.
            png_filename = os.path.join(result_dir, str(count).zfill(4) + '.png')
            img_ori.save(png_filename)
            count += 1


if __name__ == "__main__":
    main()
