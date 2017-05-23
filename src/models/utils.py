import tensorflow as tf
import math

def lrelu(x, leak=0.2, name="lrelu"):
    '''Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    '''
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def generate_patches(image, patch_h, patch_w, name='patch'):
    '''Splits an image into patches of size patch_h x patch_w
    Input: image of shape [image_h, image_w, image_ch]
    Output: batch of patches shape [n, patch_h, patch_w, image_ch]
    '''
    assert image.shape.ndims == 3

    pad = [[0, 0], [0, 0]]
    image_h = image.shape[0].value
    image_w = image.shape[1].value
    image_ch = image.shape[2].value
    p_area = patch_h * patch_w

    with tf.variable_scope(name):
        patches = tf.space_to_batch_nd([image], [patch_h, patch_w], pad)
        patches = tf.split(patches, p_area, 0)
        patches = tf.stack(patches, 3)
        patches = tf.reshape(patches, [-1, patch_h, patch_w, image_ch])

    return patches

def reconstruct_image(patches, image_h, image_w, name='reconstruct'):
    '''Reconstructs an image from patches of size patch_h x patch_w
    Input: batch of patches shape [n, patch_h, patch_w, patch_ch]
    Output: image of shape [image_h, image_w, patch_ch]
    '''
    assert patches.shape.ndims == 4

    pad = [[0, 0], [0, 0]]
    patch_h = patches.shape[1].value
    patch_w = patches.shape[2].value
    patch_ch = patches.shape[3].value
    p_area = patch_h * patch_w
    h_ratio = image_h // patch_h
    w_ratio = image_w // patch_w

    with tf.variable_scope(name):
        image = tf.reshape(patches, [1, h_ratio, w_ratio, p_area, patch_ch])
        image = tf.split(image, p_area, 3)
        image = tf.stack(image, 0)
        image = tf.reshape(image, [p_area, h_ratio, w_ratio, patch_ch])
        image = tf.batch_to_space_nd(image, [patch_h, patch_w], pad)

    return image[0]

def time_taken(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%dh %dm %.3fs' % (hours, minutes, seconds)

