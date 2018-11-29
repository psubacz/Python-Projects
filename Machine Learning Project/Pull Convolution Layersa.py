# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 14:35:44 2018

@author: Eastwind
"""

import numpy as np
import time
from keras.preprocessing.image import save_img
from keras import backend as K
from keras.models import model_from_json

# dimensions of the generated pictures for each filter.
img_width = 150
img_height = 200

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
layer_name = 'block2_conv1'

filters = 64
# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

print('Loading model')
json_file = open('model_6.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_6.h5")
print("Loaded model from disk")
print('Model Loaded')    
model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

kept_filters = []

filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, :, :, filter_index])
# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])

# we start from a gray image with some noise
input_img_data = np.random.random((1, img_width, img_height,1)) * 20 + 128.

step = 1
# run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step
    
img = input_img_data[0]
img = deprocess_image(img)
save_img('%s_filter_%d.png' % (layer_name, filter_index), img)
    
    
#for filter_index in range(filters):
#    # we only scan through the first 200 filters,
#    # but there are actually 512 of them
#    print('Processing filter %d' % filter_index)
#    start_time = time.time()
#
#    # we build a loss function that maximizes the activation
#    # of the nth filter of the layer considered
#    layer_output = layer_dict[layer_name].output
#    if K.image_data_format() == 'channels_first':
#        loss = K.mean(layer_output[:, filter_index, :, :])
#    else:
#        loss = K.mean(layer_output[:, :, :, filter_index])
#
#    # we compute the gradient of the input picture wrt this loss
#    grads = K.gradients(loss, input_img)[0]
#
#    # normalization trick: we normalize the gradient
#    grads = normalize(grads)
#
#    # this function returns the loss and grads given the input picture
#    iterate = K.function([input_img], [loss, grads])
#
#    # step size for gradient ascent
#    step = 1.
#
#    # we start from a gray image with some random noise
#    if K.image_data_format() == 'channels_first':
#        input_img_data = np.random.random((1, 3, img_width, img_height))
#    else:
#        input_img_data = np.random.random((1, img_width, img_height, 3))
#    input_img_data = (input_img_data - 0.5) * 20 + 128
#
#    # we run gradient ascent for 20 steps
#    for i in range(20):
#        loss_value, grads_value = iterate([input_img_data])
#        input_img_data += grads_value * step
#
#        print('Current loss value:', loss_value)
#        if loss_value <= 0.:
#            # some filters get stuck to 0, we can skip them
#            break
#
#    # decode the resulting input image
#    if loss_value > 0:
#        img = deprocess_image(input_img_data[0])
#        kept_filters.append((img, loss_value))
#    end_time = time.time()
#    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
#
## we will stich the best 64 filters on a 8 x 8 grid.
#n = 4
#
## the filters that have the highest loss are assumed to be better-looking.
## we will only keep the top 64 filters.
#kept_filters.sort(key=lambda x: x[1], reverse=True)
#kept_filters = kept_filters[:n * n]
#
## build a black picture with enough space for
## our 8 x 8 filters of size 128 x 128, with a 5px margin in between
#margin = 9
#width = n * img_width + (n - 1) * margin
#height = n * img_height + (n - 1) * margin
#stitched_filters = np.zeros((width, height, 3))
#
## fill the picture with our saved filters
#for i in range(n):
#    for j in range(n):
#        img, loss = kept_filters[i * n + j]
#        width_margin = (img_width + margin) * i
#        height_margin = (img_height + margin) * j
#        stitched_filters[
#            width_margin: width_margin + img_width,
#            height_margin: height_margin + img_height, :] = img
#
## save the result to disk
#save_img(layer_name+'_stitched_filters_%dx%d.png' % (n, n), stitched_filters)