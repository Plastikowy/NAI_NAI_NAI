"""
==========================================
Program to compose image using neural style transfer.
Creators:
Tomasz Samól(Plastikowy)
Sebastian Lewandowski(SxLewandowski)
==========================================
Prerequisites:
Before you run program, you need to install: Numpy, matplotlib, Pillow
IPython, OpenCV and TensorFlow  packages.
You can use for example use PIP package manager do to that:
pip install numpy
pip install Pillow
pip install ipython(we recommend v7.31.1)
pip install opencv-python
pip install tensorflow
==========================================
"""
import cv2
import cv2 as cv
import numpy as np
from PIL import Image
import time

import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models

import IPython.display

# Set up some global values here
#style examples to use for learning
style_great_waves = 'The_Great_Wave_off_Kanagawa.jpg'
style_comic = 'comic_style.jpg'
style_gouache = 'Gouache-style.jpg'

content_path = 'baldo_paralotnia.jpg'
style_path = style_gouache

# if we wanna force size of image and fasten learning process
smallSizeEnabled = 1
smallSize = 350

# amount of learning iterations for our AI
number_of_iterations = 100
every_which_iteration_save_to_file = 20

#rgb to bgr shift values
RtoB = 103.939
GtoG = 116.779
BtoR = 123.68

def load_img(path_to_img):
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)))

    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    # it converts the input image from RGB to BGR and zero-center each color channel with respect to the ImageNet
    # dataset, without scaling
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # zero-centering values because of color conversion
    x[:, :, 0] += RtoB
    x[:, :, 1] += GtoG
    x[:, :, 2] += BtoR
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_model():
    # we use VGG19 model which is pretrained on data from ImageNet
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    # print("style output:", style_outputs)
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    # print("content_outputs:", content_outputs)
    model_outputs = style_outputs + content_outputs
    return models.Model(vgg.input, model_outputs)


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))


def get_feature_representations(model, content_path, style_path):
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights

    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    # Compute gradients wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

#content weight must be larger than style weight, because we want content to stay the same
#but change the style of content, they're values which describes what we want to be changed
def run_style_transfer(content_path,
                       style_path,
                       content_weight=1e3,
                       style_weight=1e-2):
    model = get_model()

    # we set trainable to false, cause our layers dont know what to do
    # if we would have gone for second training, then we would set it to True
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    # Store our best result
    best_loss, best_img = float('inf'), None

    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    global_start = time.time()

    # zero-centering values because of color conversion
    norm_means = np.array([RtoB, GtoG, BtoR])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    for i in range(number_of_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

        if i % every_which_iteration_save_to_file == 0:
            save_result_to_file(best_img, i)

    print('Total time: {:.4f}s'.format(time.time() - global_start))
    IPython.display.clear_output(wait=True)

    return best_img, best_loss


def save_result_to_file(generated_img, iteration=0):
    print(f"Generating output image, iteration {iteration}")
    conv_generated_img = cv.cvtColor(generated_img, cv2.COLOR_BGR2RGB)
    cv.imwrite(f"generated_{iteration}_{style_path.split('.')[0]}.png", conv_generated_img)


def calculateMaxDim():
    style_image = Image.open(style_path)
    content_image = Image.open(content_path)
    if smallSizeEnabled:
        return smallSize
    if style_image.width > content_image.width:
        return style_image.width
    else:
        return content_image.width


max_dim = calculateMaxDim()
print("max dim = ", max_dim)
# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

best, best_loss = run_style_transfer(content_path, style_path)

Image.fromarray(best)

save_result_to_file(best, number_of_iterations)
