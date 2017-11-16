import numpy as np
import tensorflow as tf
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Scale

#hyperparameters
#anything bigger than that would give me OOM straight away
filter_size = []
num_filters = []
#Convolution layer #1
filter_size.append(5)
num_filters.append(32)
#Convolution layer #2
filter_size.append(5)
num_filters.append(32)
#Convolution layer #2
filter_size.append(5)
num_filters.append(64)
#Dense layer
dense_layer_size = 128

#Image dimensions
img_w, img_h = 400, 300
img_size_flat = img_w * img_h
img_shape = (img_w, img_h)
color_channels = 3

train_data = ImageFolder('train_0', transform=Compose([Scale((img_w, img_h)), ToTensor()]))

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=20,
                                          shuffle=True, num_workers=8)

num_classes = 2 #cat 0 or dog 1

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def new_biases(length):
    return tf.Variable(tf.constant(0.1, shape=[length]))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def new_conv_layer(input, #the previous layer
                   num_input_channels, #channels in the previous layer
                   filter_size, #width and height of each filter
                   num_filters, #number of filters
                   max_pooled=True): #use 2x2 max-pooling
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases  = new_biases(length=num_filters)
    layer = conv2d(input, weights)
    layer += biases
    if max_pooled:
        layer = max_pool_2x2(layer)
    layer = tf.nn.relu(layer)
    return layer

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

x = tf.placeholder(tf.float32, shape=[None, color_channels, img_h, img_w], name="x")
#image tensor should be shaped as [None, img_width, img_height, channels]
image = tf.reshape(x, shape=[-1, img_w, img_h, color_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y")
y_true_class = tf.argmax(y_true, axis=1)

layer_conv1 = new_conv_layer(input=image, num_input_channels=color_channels,
                                        filter_size=filter_size[0],
                                        num_filters=num_filters[0],
                                        max_pooled=True)

layer_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters[0],
                                        filter_size=filter_size[1],
                                        num_filters=num_filters[1],
                                        max_pooled=True)

layer_conv3 = new_conv_layer(input=layer_conv2, num_input_channels=num_filters[1],
                                        filter_size=filter_size[2],
                                        num_filters=num_filters[2],
                                        max_pooled=True)

layer_flatten, num_features = flatten_layer(layer_conv3)

layer_dense1 = new_fc_layer(layer_flatten, num_features, dense_layer_size)
layer_dense2 = new_fc_layer(layer_dense1, dense_layer_size, num_classes, False)

y_pred = tf.nn.softmax(layer_dense2)
y_pred_class = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_dense2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_class, y_true_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(0,5):
        for i, data in enumerate (train_data_loader,0):
            images, labels = data
            images, labels = images.numpy(), labels.numpy() #images shape [batch_size, color_channels, img_h, img_w]
            #images = np.transpose(images, (0,3,2,1))
            labels = np.eye(2)[labels] #labels shape [batch_size, 2]
            feed_dict_train = {x: images, y_true: labels}
            sess.run(optimizer, feed_dict_train)
            if i % 10 == 0 and i != 0:
                acc = sess.run(accuracy, feed_dict=feed_dict_train)
                # Message for printing.
                msg = "Optimization Iteration: {}, Accuracy: {}"
                # Print it.
                print(msg.format(i + 1, acc))
                #save model
                save_path = saver.save(sess, f"model/model_epoch{epoch}_itr{i}.ckpt")
